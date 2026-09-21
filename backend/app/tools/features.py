import re
import numpy as np
import pandas as pd
from sqlalchemy import inspect
from app.database.models import Dataset
from app.tools.database import dataframe

ALLOWED={"family_size","is_alone","title_from_name","fare_per_person","age_group","date_parts","ratio","bin","log1p","frequency_encode","one_hot"}

def recommend_features(db,dataset,target=None):
    df=dataframe(db,dataset); names={c.lower():c for c in df.columns}; ops=[]
    if {"sibsp","parch"}<=set(names): ops += [{"operation":"family_size","sibsp":names["sibsp"],"parch":names["parch"]},{"operation":"is_alone","from":"family_size"}]
    if "name" in names: ops.append({"operation":"title_from_name","column":names["name"]})
    if "fare" in names and ({"sibsp","parch"}<=set(names)): ops.append({"operation":"fare_per_person","fare":names["fare"],"family":"family_size"})
    if "age" in names: ops.append({"operation":"age_group","column":names["age"],"bins":[0,18,30,45,60,120]})
    for c in df.select_dtypes(include="datetime").columns: ops.append({"operation":"date_parts","column":c})
    return {"dataset":dataset.name,"target":target,"transformations":ops,"requires_approval":True}

def _apply(df,ops):
    out=df.copy()
    for op in ops:
        kind=op["operation"]
        if kind not in ALLOWED: raise ValueError(f"Unsupported feature operation: {kind}")
        if kind=="family_size": out["family_size"]=out[op["sibsp"]].fillna(0)+out[op["parch"]].fillna(0)+1
        elif kind=="is_alone": out["is_alone"]=(out[op["from"]]==1).astype(int)
        elif kind=="title_from_name": out["title"]=out[op["column"]].astype(str).str.extract(r",\s*([^.]*)\.",expand=False).fillna("Unknown")
        elif kind=="fare_per_person": out["fare_per_person"]=out[op["fare"]]/out[op["family"]].replace(0,1)
        elif kind=="age_group": out["age_group"]=pd.cut(out[op["column"]],bins=op["bins"],include_lowest=True).astype("string")
        elif kind=="date_parts":
            d=pd.to_datetime(out[op["column"]],errors="coerce"); out[f"{op['column']}_year"]=d.dt.year; out[f"{op['column']}_month"]=d.dt.month; out[f"{op['column']}_dow"]=d.dt.dayofweek
        elif kind=="ratio": out[op["name"]]=out[op["numerator"]]/out[op["denominator"]].replace(0,np.nan)
        elif kind=="bin": out[op["name"]]=pd.cut(out[op["column"]],bins=op["bins"],labels=op.get("labels")).astype("string")
        elif kind=="log1p": out[op.get("name",f"{op['column']}_log")]=np.log1p(out[op["column"]].clip(lower=0))
        elif kind=="frequency_encode": out[op.get("name",f"{op['column']}_frequency")]=out[op["column"]].map(out[op["column"]].value_counts(normalize=True))
        elif kind=="one_hot": out=pd.get_dummies(out,columns=op["columns"],drop_first=op.get("drop_first",False),dtype=int)
    return out

def preview_feature_engineering(db,dataset,transformations):
    df=dataframe(db,dataset); out=_apply(df,transformations); added=[c for c in out if c not in df]
    return {"dataset":dataset.name,"transformations":transformations,"added_columns":added,"before_columns":len(df.columns),"after_columns":len(out.columns),"sample":out[added].head(5).where(pd.notna(out[added]),None).to_dict("records") if added else [],"requires_approval":True}

def apply_feature_engineering(db,dataset,transformations,session_id=None):
    df=dataframe(db,dataset); out=_apply(df,transformations); base=re.sub(r"_(raw|clean_v\d+|features_v\d+)$","",dataset.table_name); n=db.query(Dataset).filter(Dataset.name.like(f"{dataset.name}%"),Dataset.version.like("features_v%")).count()+1; used=set(inspect(db.bind).get_table_names()); table=f"{base}_features_v{n}"
    while table in used: n+=1; table=f"{base}_features_v{n}"
    version=f"features_v{n}"; out.to_sql(table,db.bind,if_exists="fail",index=False)
    child=Dataset(name=f"{dataset.name}_{version}",display_name=f"{dataset.display_name} — features v{n}",table_name=table,description=dataset.description,source=dataset.source,license=dataset.license,row_count=len(out),column_count=len(out.columns),task_candidates=dataset.task_candidates,target_suggestions=dataset.target_suggestions,column_descriptions=dataset.column_descriptions,parent_id=dataset.id,version=version,transformations=transformations); db.add(child); db.commit(); db.refresh(child)
    return {"dataset_id":child.id,"name":child.name,"display_name":child.display_name,"table":table,"version":version,"added_columns":[c for c in out if c not in df],"lineage":{"parent_id":dataset.id}}
