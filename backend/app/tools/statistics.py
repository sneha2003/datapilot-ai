import numpy as np
import pandas as pd
from scipy import stats
from app.tools.database import dataframe

POSITIVE_VALUES={"yes","true","1","1.0","survived","subscribed","positive"}

def _binary_target(series):
    """Return a deterministic positive-class mask and its human-readable label."""
    clean=series.dropna()
    values=clean.astype(str).str.strip()
    unique=values.unique().tolist()
    if len(unique)!=2:
        raise ValueError("Customer segment analysis requires a target with exactly two outcomes")
    normalized=values.str.lower()
    recognized=[value for value in unique if str(value).strip().lower() in POSITIVE_VALUES]
    if recognized:
        positive_label=recognized[0]
    else:
        # When labels are unfamiliar, the less frequent outcome is normally the event
        # a business wants to understand (conversion, churn, fraud, and so on).
        positive_label=values.value_counts().idxmin()
    return series.astype(str).str.strip().eq(str(positive_label)),str(positive_label)

def target_driver_analysis(db,dataset,target=None,limit=8):
    """Compare real outcome rates across safe, human-readable customer segments."""
    df=dataframe(db,dataset)
    target=target or (dataset.target_suggestions[0] if dataset.target_suggestions else None)
    if not target or target not in df.columns:
        raise ValueError("A valid outcome column is required to compare customer groups")
    valid=df[target].notna()
    if valid.sum()<8:
        raise ValueError("There are too few known outcomes for a reliable segment comparison")
    positive,positive_label=_binary_target(df.loc[valid,target])
    work=df.loc[valid].copy(); work["_positive"]=positive.astype(int).to_numpy()
    baseline=float(work["_positive"].mean())
    if baseline<=0:
        raise ValueError("The selected outcome has no positive examples")

    descriptions=dataset.column_descriptions or {}
    excluded={target}
    excluded_reasons={target:"This is the outcome being measured"}
    for column in work.columns:
        name=column.lower(); description=str(descriptions.get(column,"" )).lower()
        if column=="_positive": excluded.add(column); continue
        if any(term in description for term in ["leakage","after the outcome","post-outcome"]):
            excluded.add(column); excluded_reasons[column]="Recorded during or after the outcome, so it could mislead the analysis"
        elif name in {"id","customer_id","client_id","row_id","index"} or name.endswith("_id"):
            excluded.add(column); excluded_reasons[column]="Identifier rather than a reusable customer characteristic"

    min_count=max(2 if len(work)<100 else 20,int(len(work)*.005))
    segments=[]; field_summaries=[]
    for column in [c for c in work.columns if c not in excluded]:
        series=work[column]
        unique=int(series.nunique(dropna=True))
        if unique<2: continue
        grouped_source=None
        if pd.api.types.is_numeric_dtype(series) and unique>12:
            try:
                grouped_source=pd.qcut(series,q=min(5,unique),duplicates="drop")
            except (ValueError,TypeError):
                continue
        elif unique<=30:
            grouped_source=series.astype("string").fillna("Missing")
        else:
            excluded_reasons[column]="Too many unique values for a useful customer-group comparison"
            continue
        summary=(work.assign(_segment=grouped_source)
                 .groupby("_segment",observed=True,dropna=False)["_positive"]
                 .agg(["mean","size","sum"]).reset_index())
        summary=summary[summary["size"]>=min_count]
        if summary.empty: continue
        rates=summary["mean"]
        field_summaries.append({
            "feature":column,
            "highest_rate_pct":round(float(rates.max()*100),1),
            "lowest_rate_pct":round(float(rates.min()*100),1),
            "rate_gap_points":round(float((rates.max()-rates.min())*100),1),
        })
        for _,row in summary.iterrows():
            segments.append({
                "feature":column,
                "segment":str(row["_segment"]),
                "customers":int(row["size"]),
                "positive_customers":int(row["sum"]),
                "positive_rate_pct":round(float(row["mean"]*100),1),
                "lift":round(float(row["mean"]/baseline),2),
            })
    segments.sort(key=lambda item:(item["lift"],item["customers"]),reverse=True)
    field_summaries.sort(key=lambda item:item["rate_gap_points"],reverse=True)
    return {
        "dataset":dataset.display_name,
        "target":target,
        "positive_label":positive_label,
        "baseline_rate_pct":round(baseline*100,1),
        "total_customers":int(len(work)),
        "minimum_segment_size":min_count,
        "top_segments":segments[:max(1,min(int(limit),15))],
        "strongest_fields":field_summaries[:5],
        "excluded_fields":excluded_reasons,
        "interpretation":"These are patterns in historical data. They show association, not proof that a characteristic caused the outcome.",
    }

def grouped_analysis(db,dataset,group_by,metric=None,aggregation="count",filters=None,bins=None):
    df=dataframe(db,dataset)
    for f in filters or []:
        if f["op"]=="eq": df=df[df[f["column"]]==f["value"]]
        elif f["op"]=="gt": df=df[df[f["column"]]>f["value"]]
        elif f["op"]=="lt": df=df[df[f["column"]]<f["value"]]
    if bins and isinstance(group_by,str):
        label=f"{group_by}_group"; df[label]=pd.cut(df[group_by],bins=bins,include_lowest=True).astype("string"); group_by=label
    grouped=df.groupby(group_by,dropna=False)
    if aggregation=="count": result=grouped.size().rename("count").reset_index()
    elif aggregation=="mean": result=grouped[metric].agg(["mean","size"]).reset_index().rename(columns={"mean":f"mean_{metric}","size":"sample_size"})
    elif aggregation in {"sum","median","min","max"}: result=getattr(grouped[metric],aggregation)().reset_index(name=f"{aggregation}_{metric}")
    elif aggregation=="positive_rate":
        positive=df[metric].astype(str).str.lower().isin({"yes","true","1","1.0","survived","subscribed"}).astype(float); result=df.assign(_positive=positive).groupby(group_by,dropna=False)["_positive"].agg(["mean","size"]).reset_index().rename(columns={"mean":"positive_rate","size":"sample_size"}); result["positive_rate"]=(result["positive_rate"]*100).round(3)
    else: raise ValueError("Unsupported aggregation")
    return {"columns":result.columns.tolist(),"rows":result.where(pd.notna(result),None).to_dict("records")}

def trend_analysis(db,dataset,date_column,metric=None,frequency="M",aggregation="count"):
    df=dataframe(db,dataset); dates=pd.to_datetime(df[date_column],errors="coerce"); work=df.assign(_period=dates.dt.to_period(frequency).astype(str)).dropna(subset=["_period"])
    if aggregation=="count": result=work.groupby("_period").size()
    else: result=getattr(work.groupby("_period")[metric],aggregation)()
    return {"period":frequency,"rows":[{"period":str(k),"value":float(v)} for k,v in result.items()]}

def statistical_test(db,dataset,test,columns):
    df=dataframe(db,dataset); result={"test":test,"columns":columns}
    if test in {"pearson","spearman"}:
        clean=df[columns].dropna(); fn=stats.pearsonr if test=="pearson" else stats.spearmanr; stat,p=fn(clean.iloc[:,0],clean.iloc[:,1]); result|={"statistic":float(stat),"p_value":float(p),"sample_sizes":[len(clean)],"effect_size":float(stat)}
    elif test=="chi_square":
        table=pd.crosstab(df[columns[0]],df[columns[1]]); stat,p,dof,_=stats.chi2_contingency(table); n=table.values.sum(); result|={"statistic":float(stat),"p_value":float(p),"degrees_of_freedom":int(dof),"sample_sizes":[int(n)],"effect_size":float(np.sqrt(stat/(n*max(1,min(table.shape)-1))))}
    elif test=="t_test":
        values=[x.dropna().to_numpy() for _,x in df.groupby(columns[0])[columns[1]]];
        if len(values)!=2: raise ValueError("t-test requires exactly two groups")
        stat,p=stats.ttest_ind(*values,equal_var=False); pooled=np.sqrt(((values[0].std()**2)+(values[1].std()**2))/2); result|={"statistic":float(stat),"p_value":float(p),"sample_sizes":[len(x) for x in values],"effect_size":float((values[0].mean()-values[1].mean())/pooled) if pooled else None}
    elif test=="anova":
        values=[x.dropna().to_numpy() for _,x in df.groupby(columns[0])[columns[1]]]; stat,p=stats.f_oneway(*values); result|={"statistic":float(stat),"p_value":float(p),"sample_sizes":[len(x) for x in values]}
    else: raise ValueError("Unsupported statistical test")
    result["interpretation_hint"]="Evidence of association" if result["p_value"]<.05 else "No statistically significant association at alpha=0.05"
    return result
