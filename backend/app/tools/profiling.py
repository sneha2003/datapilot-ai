import numpy as np
import pandas as pd
from app.tools.database import dataframe

def _json(value):
    if pd.isna(value): return None
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,)): return float(value)
    return value

def profile_dataset(db,dataset):
    df=dataframe(db,dataset); n=max(len(df),1); columns=[]
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    for c in df.columns:
        s=df[c]; missing=int(s.isna().sum()); unique=int(s.nunique(dropna=True)); item={"name":c,"dtype":str(s.dtype),"missing":missing,"missing_pct":round(missing*100/n,2),"unique":unique,"constant":unique<=1,"high_cardinality":unique>50 and unique/n>.3,"potential_id":unique==len(df) and ("id" in c.lower() or unique>100),"top_values":{str(k):int(v) for k,v in s.value_counts(dropna=False).head(8).items()}}
        if c in numeric:
            clean=s.dropna(); q=clean.quantile([.25,.5,.75]) if len(clean) else pd.Series(dtype=float); iqr=(q.get(.75,0)-q.get(.25,0))
            item["statistics"]={"min":_json(clean.min()),"max":_json(clean.max()),"mean":_json(clean.mean()),"median":_json(clean.median()),"std":_json(clean.std()),"q25":_json(q.get(.25)),"q75":_json(q.get(.75)),"skew":_json(clean.skew())}
            item["outliers_iqr"]=int(((clean<q.get(.25,0)-1.5*iqr)|(clean>q.get(.75,0)+1.5*iqr)).sum()) if len(clean) else 0
        columns.append(item)
    leakage=[{"column":c,"target":dataset.target_suggestions[0] if dataset.target_suggestions else None,"reason":description,"score":None} for c,description in dataset.column_descriptions.items() if c in df and "leakage" in str(description).lower()]
    for target in dataset.target_suggestions:
        if target in df:
            for c in numeric:
                if c!=target and pd.api.types.is_numeric_dtype(df[target]):
                    corr=df[[c,target]].corr().iloc[0,1]
                    if pd.notna(corr) and abs(corr)>.98: leakage.append({"column":c,"target":target,"reason":"near-perfect correlation","score":round(float(corr),4)})
    return {"dataset":dataset.name,"version":dataset.version,"shape":{"rows":len(df),"columns":len(df.columns)},"duplicate_rows":int(df.duplicated().sum()),"missing_cells":int(df.isna().sum().sum()),"missing_pct":round(float(df.isna().mean().mean()*100),2),"columns":columns,"potential_leakage":leakage}

def dataset_overview(db,dataset):
    """Return a compact, computed business overview for broad beginner questions."""
    df=dataframe(db,dataset); profile=profile_dataset(db,dataset)
    target=next((x for x in dataset.target_suggestions if x in df.columns),None)
    target_distribution=[]; target_statistics=None
    if target:
        series=df[target].dropna(); unique=series.nunique()
        continuous_target="regression" in dataset.task_candidates and "classification" not in dataset.task_candidates
        if pd.api.types.is_numeric_dtype(series) and (continuous_target or unique>12):
            target_statistics={"mean":round(float(series.mean()),2),"median":round(float(series.median()),2),"minimum":_json(series.min()),"maximum":_json(series.max())}
        elif unique<=12:
            counts=df[target].value_counts(dropna=False)
            target_distribution=[{"value":str(value),"count":int(count),"percentage":round(float(count*100/len(df)),1)} for value,count in counts.items()]
    numeric=[]
    for column in df.select_dtypes(include=np.number).columns[:6]:
        series=df[column].dropna()
        if len(series): numeric.append({"column":column,"mean":round(float(series.mean()),2),"median":round(float(series.median()),2),"minimum":_json(series.min()),"maximum":_json(series.max())})
    categorical=[]
    for column in df.select_dtypes(exclude=np.number).columns:
        if column==target: continue
        counts=df[column].value_counts(dropna=False).head(3)
        categorical.append({"column":column,"most_common":[{"value":str(value),"count":int(count)} for value,count in counts.items()]})
        if len(categorical)>=6: break
    return {
        "dataset":dataset.name,"display_name":dataset.display_name,"description":dataset.description,
        "source":dataset.source,"license":dataset.license,"rows":len(df),"columns":len(df.columns),
        "column_names":df.columns.tolist(),"target":target,"target_description":dataset.column_descriptions.get(target,"") if target else "",
        "target_distribution":target_distribution,"target_statistics":target_statistics,"numeric_highlights":numeric,"categorical_highlights":categorical,
        "missing_pct":profile["missing_pct"],"duplicate_rows":profile["duplicate_rows"],"potential_leakage":profile["potential_leakage"],
    }

def missing_value_report(db,dataset):
    p=profile_dataset(db,dataset); return {"dataset":dataset.name,"columns":[x for x in p["columns"] if x["missing"]],"missing_cells":p["missing_cells"]}

def calculate_correlations(db,dataset,method="pearson"):
    df=dataframe(db,dataset).select_dtypes(include=np.number); corr=df.corr(method=method).round(4)
    return {"method":method,"columns":corr.columns.tolist(),"matrix":corr.where(pd.notna(corr),None).values.tolist()}

def quality_issues(profile):
    issues=[]
    if profile["duplicate_rows"]: issues.append({"severity":"medium","type":"duplicates","message":f"{profile['duplicate_rows']} exact duplicate rows"})
    for c in profile["columns"]:
        if c["missing_pct"]: issues.append({"severity":"high" if c["missing_pct"]>30 else "medium","type":"missing","column":c["name"],"message":f"{c['missing_pct']}% missing"})
        if c["constant"]: issues.append({"severity":"medium","type":"constant","column":c["name"],"message":"Column has one or no distinct values"})
        if c["high_cardinality"]: issues.append({"severity":"low","type":"cardinality","column":c["name"],"message":f"{c['unique']} distinct values"})
        if c.get("statistics",{}).get("skew") and abs(c["statistics"]["skew"])>2: issues.append({"severity":"low","type":"skew","column":c["name"],"message":f"Strong skew ({c['statistics']['skew']:.2f})"})
    issues += [{"severity":"high","type":"leakage",**x} for x in profile["potential_leakage"]]
    return issues
