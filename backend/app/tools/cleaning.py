import re
import pandas as pd
from sqlalchemy import inspect
from app.database.models import Dataset
from app.tools.database import dataframe
from app.tools.profiling import profile_dataset, quality_issues

ALLOWED={"drop_duplicates","impute_numeric","impute_categorical","standardize_categories","convert_type","cap_outliers","drop_columns"}

def recommend_cleaning_steps(db,dataset):
    p=profile_dataset(db,dataset); ops=[]
    if p["duplicate_rows"]: ops.append({"operation":"drop_duplicates"})
    for c in p["columns"]:
        if c["missing"]:
            if "int" in c["dtype"] or "float" in c["dtype"]: ops.append({"operation":"impute_numeric","column":c["name"],"strategy":"median"})
            elif c["missing_pct"]<=70: ops.append({"operation":"impute_categorical","column":c["name"],"strategy":"mode"})
        if c["constant"]: ops.append({"operation":"drop_columns","columns":[c["name"]],"reason":"constant"})
    return {"dataset":dataset.name,"issues":quality_issues(p),"operations":ops,"requires_approval":True}

def _apply(df,operations):
    out=df.copy(); effects=[]
    for op in operations:
        kind=op.get("operation")
        if kind not in ALLOWED: raise ValueError(f"Unsupported cleaning operation: {kind}")
        before=out.shape
        if kind=="drop_duplicates": out=out.drop_duplicates()
        elif kind=="impute_numeric":
            c=op["column"]; value=out[c].median() if op.get("strategy","median")=="median" else out[c].mean(); out[c]=out[c].fillna(value)
        elif kind=="impute_categorical":
            c=op["column"]; mode=out[c].mode(dropna=True); out[c]=out[c].fillna(mode.iloc[0] if len(mode) else "Unknown")
        elif kind=="standardize_categories":
            c=op["column"]; out[c]=out[c].astype("string").str.strip().str.lower()
        elif kind=="convert_type": out[op["column"]]=out[op["column"]].astype(op["dtype"])
        elif kind=="cap_outliers":
            c=op["column"]; q1,q3=out[c].quantile([.25,.75]); iqr=q3-q1; out[c]=out[c].clip(q1-1.5*iqr,q3+1.5*iqr)
        elif kind=="drop_columns": out=out.drop(columns=op["columns"],errors="ignore")
        effects.append({"operation":kind,"before":{"rows":before[0],"columns":before[1]},"after":{"rows":len(out),"columns":len(out.columns)},"missing_after":int(out.isna().sum().sum())})
    return out,effects

def preview_cleaning(db,dataset,operations):
    df=dataframe(db,dataset); out,effects=_apply(df,operations)
    return {"dataset":dataset.name,"operations":operations,"effects":effects,"before":{"rows":len(df),"columns":len(df.columns),"missing":int(df.isna().sum().sum())},"after":{"rows":len(out),"columns":len(out.columns),"missing":int(out.isna().sum().sum())},"requires_approval":True}

def apply_cleaning(db,dataset,operations,session_id=None):
    df=dataframe(db,dataset); out,effects=_apply(df,operations)
    base=re.sub(r"_(raw|clean_v\d+|features_v\d+)$","",dataset.table_name); used=set(inspect(db.bind).get_table_names()); existing=db.query(Dataset).filter(Dataset.name.like(f"{dataset.name}%"),Dataset.version.like("clean_v%")).count(); n=existing+1; table=f"{base}_clean_v{n}"
    while table in used: n+=1; table=f"{base}_clean_v{n}"
    version=f"clean_v{n}"
    out.to_sql(table,db.bind,if_exists="fail",index=False)
    child=Dataset(name=f"{dataset.name}_{version}",display_name=f"{dataset.display_name} — cleaned v{n}",table_name=table,description=dataset.description,source=dataset.source,license=dataset.license,row_count=len(out),column_count=len(out.columns),task_candidates=dataset.task_candidates,target_suggestions=dataset.target_suggestions,column_descriptions=dataset.column_descriptions,parent_id=dataset.id,version=version,transformations=operations)
    db.add(child); db.commit(); db.refresh(child)
    return {"dataset_id":child.id,"name":child.name,"display_name":child.display_name,"table":table,"version":version,"effects":effects,"lineage":{"parent_id":dataset.id,"parent_version":dataset.version}}
