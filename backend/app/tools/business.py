"""Deterministic business investigations over registered tabular datasets.

The planner chooses an objective; this module alone calculates the evidence.
It never executes generated code and does not infer causation from association.
"""
import calendar
import numpy as np
import pandas as pd
from app.tools.database import dataframe
from app.tools.profiling import profile_dataset, quality_issues

INTENTS={"quality","kpis","trend","ranking","root_cause","anomalies","drivers","discounts","customers","churn_risk","forecast","actions","investigation","patterns"}
ALIASES={
    "revenue":["revenue","sales_amount","net_sales","sales","order_value","total_amount","amount"],
    "profit":["profit","gross_profit","net_profit","profit_amount"],
    "quantity":["quantity","units","units_sold","qty"],
    "orders":["order_id","transaction_id","invoice_id","purchase_id"],
    "customers":["customer_id","client_id","buyer_id","user_id"],
    "date":["order_date","transaction_date","purchase_date","sale_date","date","datetime","timestamp"],
    "product":["product","product_name","item","item_name","sku"],
    "category":["category","product_category","segment","sub_category"],
    "region":["region","country","state","city","territory","store"],
    "discount":["discount","discount_pct","discount_rate","discount_amount"],
}
FRIENDLY={"medinc":"median income","averooms":"average rooms per home","avebedrms":"average bedrooms per home","aveoccup":"average household occupancy","houseage":"median house age"}

def _label(column): return FRIENDLY.get(column.lower(),column.replace("_"," "))

def _n(value,digits=1):
    if not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value): return "—"
    return f"{float(value):,.{digits}f}"

def _metric(label,value): return {"label":label,"value":str(value)}
def _finding(title,detail): return {"title":title,"detail":detail}
def _role(df,kind):
    lower={str(c).lower():c for c in df.columns}
    for alias in ALIASES[kind]:
        if alias in lower:
            if kind in {"revenue","profit","quantity","discount"} and not pd.api.types.is_numeric_dtype(df[lower[alias]]): continue
            return lower[alias]
    tokens={name:set(name.split("_")) for name in lower}
    hints={"revenue":{"revenue","sales","salesamount"},"profit":{"profit","margin"},"quantity":{"quantity","units","qty"},"orders":{"order","transaction","invoice","purchase"},"customers":{"customer","client","buyer","user"},"date":{"date","datetime","timestamp"},"product":{"product","item","sku"},"category":{"category","subcategory","segment"},"region":{"region","country","state","city","territory","store"},"discount":{"discount"}}
    for name,parts in tokens.items():
        if parts & hints[kind] and (kind not in {"orders","customers"} or "id" in parts):
            if kind in {"revenue","profit","quantity","discount"} and not pd.api.types.is_numeric_dtype(df[lower[name]]): continue
            return lower[name]
    return None

def _roles(df): return {kind:_role(df,kind) for kind in ALIASES}

def _date_series(df,column):
    if not column: return None
    series=pd.to_datetime(df[column],errors="coerce")
    return series if series.notna().sum()>=max(3,int(len(df)*.7)) else None

def _periods(df,dates,measure):
    if dates is None or measure is None: return None
    values=pd.to_numeric(df[measure],errors="coerce")
    work=pd.DataFrame({"period":dates.dt.to_period("M"),"value":values}).dropna()
    if work.empty: return None
    grouped=work.groupby("period")["value"].agg(["sum","count"])
    days=dates.groupby(dates.dt.to_period("M")).apply(lambda part:part.dt.day.nunique())
    if len(grouped)>2 and days.median()>10:
        first,last=dates.min(),dates.max()
        if first.day>3: grouped=grouped.iloc[1:]
        if last.day<calendar.monthrange(last.year,last.month)[1]-2: grouped=grouped.iloc[:-1]
    return grouped if len(grouped) else None

def _base(dataset,intent):
    return {"dataset":dataset.display_name,"intent":intent,"status":"supported","headline":"","metrics":[],"findings":[],"recommendations":[],"limitations":[],"chart":None}

def _unsupported(out,reason):
    out["status"]="unsupported"; out["headline"]="This dataset cannot support that calculation yet."
    out["limitations"].append(reason)
    return out

def _coverage(df,roles,out):
    date=_date_series(df,roles["date"])
    if date is not None:
        out["metrics"].append(_metric("Time covered",f"{date.min().date()} to {date.max().date()}"))
    return date

def _quality(db,df,dataset,roles,out):
    profile=profile_dataset(db,dataset)
    issues=quality_issues(profile)
    out["metrics"] += [_metric("Records",f"{len(df):,}"),_metric("Missing cells",f"{profile['missing_pct']}%"),_metric("Duplicate rows",f"{profile['duplicate_rows']:,}")]
    date=_coverage(df,roles,out)
    for issue in sorted(issues,key=lambda x:{"high":0,"medium":1,"low":2}.get(x["severity"],3))[:5]:
        out["findings"].append(_finding(issue.get("column") or issue["type"],issue.get("message") or issue["type"]))
    if not issues: out["findings"].append(_finding("Basic checks","No missing cells or exact duplicate rows were found."))
    if date is None: out["limitations"].append("No reliable date field was found, so a time-coverage check was not possible.")
    out["headline"]="The data is usable for initial analysis." if not issues else f"The data has {len(issues)} quality signals to review before important decisions."
    return out

def _kpis(df,roles,out):
    revenue,profit,orders,customers,quantity=(roles[k] for k in ["revenue","profit","orders","customers","quantity"])
    if not any([revenue,profit,orders,customers,quantity]): return _unsupported(out,"No revenue, profit, order, customer, or quantity field was found.")
    if revenue:
        total=float(pd.to_numeric(df[revenue],errors="coerce").sum()); out["metrics"].append(_metric("Total revenue",_n(total,2)))
    if profit:
        total_profit=float(pd.to_numeric(df[profit],errors="coerce").sum()); out["metrics"].append(_metric("Total profit",_n(total_profit,2)))
        if revenue and total: out["metrics"].append(_metric("Profit margin",f"{total_profit/total*100:.1f}%"))
    if orders:
        count=int(df[orders].nunique()); out["metrics"].append(_metric("Orders",f"{count:,}"))
        if revenue and count: out["metrics"].append(_metric("Average order value",_n(total/count,2)))
    if customers: out["metrics"].append(_metric("Customers",f"{df[customers].nunique():,}"))
    if quantity: out["metrics"].append(_metric("Units sold",_n(pd.to_numeric(df[quantity],errors="coerce").sum(),0)))
    out["headline"]="These are the main measured business KPIs in this dataset."
    if not orders and revenue: out["limitations"].append("Average order value needs an order identifier and was not calculated.")
    return out

def _trend(df,roles,out,measure=None):
    measure=measure or roles["revenue"] or roles["profit"] or roles["quantity"]
    date=_date_series(df,roles["date"])
    if date is None: return _unsupported(out,"A usable date field is required for a trend or forecast.")
    if measure is None:
        measure="_record_count"; frame=df.assign(_record_count=1)
    else: frame=df
    monthly=_periods(frame,date,measure)
    if monthly is None or len(monthly)<2: return _unsupported(out,"At least two comparable months are needed for a trend.")
    first,last=monthly.iloc[0],monthly.iloc[-1]
    change=float((last["sum"]-first["sum"])/abs(first["sum"])*100) if first["sum"] else None
    previous=monthly.iloc[-2]
    recent=float((last["sum"]-previous["sum"])/abs(previous["sum"])*100) if previous["sum"] else None
    label=measure.replace("_"," ") if measure!="_record_count" else "record count"
    out["metrics"] += [_metric("Months compared",str(len(monthly))),_metric("Latest month",_n(last["sum"],2)),_metric("Change vs prior month",f"{recent:+.1f}%" if recent is not None else "—")]
    high,low=monthly["sum"].idxmax(),monthly["sum"].idxmin()
    out["findings"] += [_finding("Highest month",f"{high}: {_n(monthly.loc[high,'sum'],2)} {label}."),_finding("Lowest month",f"{low}: {_n(monthly.loc[low,'sum'],2)} {label}.")]
    if change is not None: out["findings"].append(_finding("Overall direction",f"{label.capitalize()} changed {change:+.1f}% from the first to the latest comparable month."))
    out["headline"]=f"{label.capitalize()} {'rose' if recent is not None and recent>0 else 'fell' if recent is not None and recent<0 else 'was unchanged'} {abs(recent):.1f}% in the latest comparable month." if recent is not None else f"Monthly {label} was calculated."
    out["chart"]={"title":f"Monthly {label}","chart_type":"line","x":"period","y":"value","rows":[{"period":str(period),"value":round(float(row["sum"]),2)} for period,row in monthly.tail(36).iterrows()]}
    return out

def _ranking(df,roles,out,question):
    q=question.lower(); measure=roles["profit"] if "profit" in q and roles["profit"] else roles["revenue"] or roles["profit"] or roles["quantity"]
    group=next((roles[k] for k in ["product","category"] if roles[k] and any(word in q for word in ["product","item","categor","sku"])),None)
    if group is None and any(word in q for word in ["region","location","city","country","territory","store"]): group=roles["region"]
    if group is None and "customer" in q: group=roles["customers"]
    group=group or roles["category"] or roles["region"] or roles["product"]
    if not group or not measure: return _unsupported(out,"A relevant group and numeric business measure are required for a best/worst comparison.")
    work=df[[group,measure]].copy(); work[measure]=pd.to_numeric(work[measure],errors="coerce"); work=work.dropna(subset=[group,measure])
    table=work.groupby(group)[measure].agg(["sum","size"]).reset_index(); table=table[table["size"]>=max(2,int(len(df)*.001))]
    if len(table)<2: return _unsupported(out,"There are not enough comparable groups with reliable sample sizes.")
    table=table.sort_values("sum",ascending=False); best,worst=table.iloc[0],table.iloc[-1]
    out["metrics"] += [_metric("Groups compared",str(len(table))),_metric("Top group",str(best[group]))]
    out["findings"] += [_finding("Best",f"{best[group]}: {_n(best['sum'],2)} total {measure.replace('_',' ')} across {int(best['size']):,} records."),_finding("Worst",f"{worst[group]}: {_n(worst['sum'],2)} total {measure.replace('_',' ')} across {int(worst['size']):,} records.")]
    out["headline"]=f"{best[group]} leads on total {measure.replace('_',' ')} among the compared {group.replace('_',' ')} groups."
    out["chart"]={"title":f"{measure.replace('_',' ').title()} by {group.replace('_',' ')}","chart_type":"bar","x":"group","y":"value","rows":[{"group":str(row[group]),"value":round(float(row["sum"]),2)} for _,row in table.head(12).iterrows()]}
    out["limitations"].append("Totals reflect group size as well as performance; compare per-order or per-customer rates before changing strategy.")
    return out

def _numeric_drivers(df,dataset,out):
    target=next((column for column in dataset.target_suggestions if column in df and pd.api.types.is_numeric_dtype(df[column])),None)
    if target is None:
        target=next((column for column in ["revenue","profit","count","sales"] if column in df and pd.api.types.is_numeric_dtype(df[column])),None)
    if target is None: return _unsupported(out,"A numeric outcome such as revenue, profit, demand, or house value is required.")
    numeric=df.select_dtypes(include=np.number)
    candidates=[]
    for column in numeric.columns:
        if column==target or column.lower().endswith("_id") or numeric[column].nunique(dropna=True)<3: continue
        clean=numeric[[column,target]].dropna()
        if len(clean)<10: continue
        r=clean[column].corr(clean[target],method="spearman")
        if pd.notna(r): candidates.append((column,float(r),len(clean)))
    candidates.sort(key=lambda item:abs(item[1]),reverse=True)
    if not candidates: return _unsupported(out,"No numeric field had enough comparable records to calculate a relationship.")
    for column,r,n in candidates[:4]:
        direction="higher" if r>0 else "lower"
        out["findings"].append(_finding(_label(column).capitalize(),f"Measured Spearman correlation with {_label(target)}: {r:+.2f} across {n:,} records; higher values tend to align with {direction} outcomes."))
    strongest=candidates[0]
    field=strongest[0]
    clean=numeric[[field,target]].dropna()
    lower,upper=clean[field].quantile([.25,.75])
    low_group=clean[clean[field]<=lower][target]
    high_group=clean[clean[field]>=upper][target]
    if len(low_group)>=5 and len(high_group)>=5:
        out["findings"].append(_finding("High versus low group",f"Among districts in the lower quarter of {_label(field)}, the median for {_label(target)} is {_n(low_group.median(),2)}; in the upper quarter it is {_n(high_group.median(),2)} ({len(low_group):,} and {len(high_group):,} records)."))
        out["chart"]={"title":f"{_label(target).title()} by {_label(field)} group","chart_type":"bar","x":"group","y":"value","rows":[{"group":"Lower quarter","value":round(float(low_group.median()),3)},{"group":"Upper quarter","value":round(float(high_group.median()),3)}]}
    out["metrics"] += [_metric("Records",f"{len(df):,}"),_metric("Strongest measured link",_label(strongest[0])),_metric("Correlation",f"{strongest[1]:+.2f}")]
    out["headline"]=f"{_label(strongest[0]).capitalize()} has the strongest measured relationship with {_label(target)} in this dataset."
    out["recommendations"].append(f"Investigate {_label(strongest[0])} alongside other relevant factors before using this pattern to make decisions.")
    out["limitations"].append("These are associations, not proof of causes. Related fields can reflect the same underlying pattern.")
    return out

def _root_cause(df,roles,out,question):
    q=question.lower(); measure=roles["profit"] if "profit" in q and roles["profit"] else roles["revenue"] or roles["profit"]
    dates=_date_series(df,roles["date"])
    if not measure or dates is None: return _unsupported(out,"A dated revenue or profit field is required to investigate a change.")
    periods=_periods(df,dates,measure)
    if periods is None or len(periods)<2: return _unsupported(out,"At least two comparable months are required to investigate a change.")
    before,after=periods.index[-2:]; before_value=float(periods.loc[before,"sum"]); after_value=float(periods.loc[after,"sum"]); delta=after_value-before_value
    out["metrics"] += [_metric(str(before),_n(before_value,2)),_metric(str(after),_n(after_value,2)),_metric("Change",f"{delta:+,.2f}")]
    out["headline"]=f"{measure.replace('_',' ').title()} {'fell' if delta<0 else 'rose' if delta>0 else 'was unchanged'} by {_n(abs(delta),2)} from {before} to {after}."
    period=dates.dt.to_period("M"); selected=df[period.isin([before,after])].copy(); selected["_period"]=period[period.isin([before,after])].astype(str).to_numpy()
    selected[measure]=pd.to_numeric(selected[measure],errors="coerce")
    orders=roles["orders"]
    if orders and roles["revenue"]==measure:
        counts=selected.groupby("_period")[orders].nunique(); a,b=(int(counts.get(str(p),0)) for p in [before,after])
        if a and b:
            out["findings"].append(_finding("Order volume",f"Orders changed from {a:,} to {b:,} ({(b-a)/a*100:+.1f}%)."))
            if before_value:
                out["findings"].append(_finding("Average order value",f"Revenue per order changed from {_n(before_value/a,2)} to {_n(after_value/b,2)} ({((after_value/b)/(before_value/a)-1)*100:+.1f}%)."))
    group=roles["category"] or roles["product"] or roles["region"]
    if group:
        pivot=selected.pivot_table(index=group,columns="_period",values=measure,aggfunc="sum",fill_value=0)
        if str(before) in pivot and str(after) in pivot:
            contributions=(pivot[str(after)]-pivot[str(before)]).sort_values()
            for name,value in (contributions.head(3).items() if delta<0 else contributions.tail(3).sort_values(ascending=False).items()):
                out["findings"].append(_finding(str(name),f"{measure.replace('_',' ').title()} changed {float(value):+,.2f} in this {group.replace('_',' ')} group."))
            if len(contributions): out["recommendations"].append(f"Review the {group.replace('_',' ')} groups with the largest negative contribution before choosing a response.")
    out["limitations"].append("This decomposes the recorded change; it does not establish why customers or markets behaved differently.")
    return out

def _discounts(df,roles,out):
    discount,profit,revenue=roles["discount"],roles["profit"],roles["revenue"]
    if not discount or not profit or not revenue: return _unsupported(out,"Discount, revenue, and profit fields are all required to assess discount trade-offs.")
    flag=pd.to_numeric(df[discount],errors="coerce").fillna(0)>0
    if flag.nunique()<2: return _unsupported(out,"The data does not contain both discounted and non-discounted records.")
    work=pd.DataFrame({"discounted":flag,"revenue":pd.to_numeric(df[revenue],errors="coerce"),"profit":pd.to_numeric(df[profit],errors="coerce")})
    grouped=work.groupby("discounted").agg(revenue=("revenue","sum"),profit=("profit","sum"),rows=("revenue","size"))
    for key,label in [(False,"No discount"),(True,"Discounted")]:
        row=grouped.loc[key]; margin=row["profit"]/row["revenue"]*100 if row["revenue"] else None
        out["findings"].append(_finding(label,f"{int(row['rows']):,} records; revenue {_n(row['revenue'],2)}; profit {_n(row['profit'],2)}; margin {f'{margin:.1f}%' if margin is not None else 'unavailable'}."))
    out["headline"]="Discounted and non-discounted transactions show different measured margins."
    out["limitations"].append("This is an observational comparison. Discounts may be used for different products or customers, so this does not prove they caused the margin difference.")
    return out

def _customers(df,roles,out):
    customer,revenue,orders=roles["customers"],roles["revenue"],roles["orders"]
    if not customer or not revenue: return _unsupported(out,"Customer identifiers and a numeric revenue field are required for customer value analysis.")
    work=df[[customer,revenue]+([orders] if orders else [])].copy(); work[revenue]=pd.to_numeric(work[revenue],errors="coerce")
    aggregate=work.groupby(customer)[revenue].agg(["sum","size"]).sort_values("sum",ascending=False)
    if aggregate.empty: return _unsupported(out,"No valid customer revenue records were found.")
    top_count=max(1,int(np.ceil(len(aggregate)*.2))); top_share=float(aggregate.head(top_count)["sum"].sum()/aggregate["sum"].sum()*100) if aggregate["sum"].sum() else None
    out["metrics"] += [_metric("Customers",f"{len(aggregate):,}"),_metric("Top 20% revenue share",f"{top_share:.1f}%" if top_share is not None else "—")]
    for name,row in aggregate.head(3).iterrows(): out["findings"].append(_finding(str(name),f"Recorded revenue {_n(row['sum'],2)} across {int(row['size']):,} rows."))
    out["headline"]=f"The highest-value customer group accounts for {top_share:.1f}% of recorded revenue." if top_share is not None else "Customer value was measured from recorded revenue."
    out["limitations"].append("Rows are not necessarily distinct orders; use an order identifier before interpreting purchase frequency.")
    return out

def _anomalies(df,roles,out):
    measure=roles["revenue"] or roles["profit"] or roles["quantity"]
    if not measure:
        numeric=df.select_dtypes(include=np.number).columns.tolist(); measure=numeric[0] if numeric else None
    if not measure: return _unsupported(out,"No numeric measure is available to check for unusual values.")
    series=pd.to_numeric(df[measure],errors="coerce").dropna()
    if len(series)<20: return _unsupported(out,"At least 20 numeric observations are needed for a useful anomaly check.")
    q1,q3=series.quantile([.25,.75]); iqr=q3-q1; low,high=q1-1.5*iqr,q3+1.5*iqr
    count=int(((series<low)|(series>high)).sum())
    out["metrics"] += [_metric("Unusual values",f"{count:,}"),_metric("Share of records",f"{count/len(series)*100:.1f}%")]
    out["findings"].append(_finding("Range checked",f"Values below {_n(low,2)} or above {_n(high,2)} for {measure.replace('_',' ')} were flagged using the interquartile-range rule."))
    out["headline"]=f"{count:,} {measure.replace('_',' ')} values are unusual relative to the rest of this dataset."
    out["limitations"].append("An unusual value is not automatically an error or a business problem; inspect the underlying records before acting.")
    return out

def _forecast(df,roles,out):
    measure=roles["revenue"] or roles["profit"] or roles["quantity"]
    dates=_date_series(df,roles["date"])
    if not measure or dates is None: return _unsupported(out,"A dated numeric business measure is required to forecast the next month.")
    monthly=_periods(df,dates,measure)
    if monthly is None or len(monthly)<12: return _unsupported(out,"At least 12 comparable months are required for a seasonal next-month forecast.")
    series=monthly["sum"].astype(float)
    if len(series)>=18:
        errors=[abs(series.iloc[index]-series.iloc[index-12]) for index in range(12,len(series))]
        estimate=float(series.iloc[-11]); method="same month last year"
    else:
        errors=[abs(series.iloc[index]-series.iloc[index-1]) for index in range(1,len(series))]
        estimate=float(series.iloc[-1]); method="latest month"
    mae=float(np.mean(errors)); period=series.index[-1]+1
    out["metrics"] += [_metric("Forecast month",str(period)),_metric("Estimated total",_n(estimate,2)),_metric("Backtest average error",_n(mae,2))]
    out["headline"]=f"A simple baseline estimates {_n(estimate,2)} {measure.replace('_',' ')} for {period}."
    out["findings"].append(_finding("Method",f"Used the {method} as a baseline and measured its historical absolute error over {len(errors)} comparisons."))
    out["limitations"].append("This is a baseline projection, not a guarantee. Promotions, prices, supply, and market changes are not modeled.")
    return out

def analyze_business_question(db,dataset,question,intent="investigation"):
    if intent not in INTENTS: raise ValueError(f"Unsupported business objective: {intent}")
    df=dataframe(db,dataset)
    if df.empty: return _unsupported(_base(dataset,intent),"The selected dataset contains no records.")
    roles=_roles(df); out=_base(dataset,intent)
    if intent=="quality":
        return _quality(db,df,dataset,roles,out)
    if intent=="kpis": return _kpis(df,roles,out)
    if intent=="trend": return _trend(df,roles,out)
    if intent=="ranking": return _ranking(df,roles,out,question)
    if intent=="root_cause": return _root_cause(df,roles,out,question)
    if intent=="discounts": return _discounts(df,roles,out)
    if intent=="customers": return _customers(df,roles,out)
    if intent=="churn_risk": return _unsupported(out,"A churn outcome or dated customer activity history is required to identify individual customers at risk. This dataset does not provide a validated risk definition.")
    if intent=="anomalies": return _anomalies(df,roles,out)
    if intent=="forecast": return _forecast(df,roles,out)
    if intent in {"drivers","patterns"}:
        if roles["revenue"] and roles["profit"] and intent=="patterns":
            intent="investigation"
        else:
            return _numeric_drivers(df,dataset,out)
    if intent in {"actions","investigation"}:
        if roles["revenue"] and roles["profit"]:
            out=_kpis(df,roles,out)
            trend=_trend(df,roles,_base(dataset,"trend"),roles["revenue"])
            if trend["status"]=="supported":
                out["findings"] += trend["findings"][:2]; out["chart"]=trend["chart"]
            profit_trend=_trend(df,roles,_base(dataset,"trend"),roles["profit"])
            if profit_trend["status"]=="supported":
                out["findings"].append(_finding("Profit trend",profit_trend["headline"]))
            dates=_date_series(df,roles["date"])
            if dates is not None:
                monthly=pd.DataFrame({"period":dates.dt.to_period("M"),"revenue":pd.to_numeric(df[roles["revenue"]],errors="coerce"),"profit":pd.to_numeric(df[roles["profit"]],errors="coerce")}).groupby("period")[["revenue","profit"]].sum()
                if len(monthly)>=24:
                    current,previous=monthly.iloc[-12:],monthly.iloc[-24:-12]
                    revenue_change=(current.revenue.sum()/previous.revenue.sum()-1)*100 if previous.revenue.sum() else None
                    profit_change=(current.profit.sum()/previous.profit.sum()-1)*100 if previous.profit.sum() else None
                    old_margin=previous.profit.sum()/previous.revenue.sum()*100 if previous.revenue.sum() else None
                    new_margin=current.profit.sum()/current.revenue.sum()*100 if current.revenue.sum() else None
                    if None not in (revenue_change,profit_change,old_margin,new_margin):
                        out["findings"].insert(0,_finding("Last 12 months vs prior 12",f"Revenue changed {revenue_change:+.1f}%, profit changed {profit_change:+.1f}%, and profit margin moved from {old_margin:.1f}% to {new_margin:.1f}%."))
                        out["headline"]=f"Revenue changed {revenue_change:+.1f}% while profit changed {profit_change:+.1f}% over the last 12 months."
                        if new_margin<old_margin:
                            out["recommendations"].append("Investigate why the profit margin narrowed before increasing sales spend.")
            if roles["category"] or roles["region"]:
                ranked=_ranking(df,roles,_base(dataset,"ranking"),"profit by category" if roles["category"] else "profit by region")
                if ranked["status"]=="supported":
                    out["findings"] += ranked["findings"][:2]
                    out["recommendations"].append(f"Review low-profit {roles['category'] or roles['region']} groups before changing pricing or investment.")
            if not out["headline"]: out["headline"]="Revenue and profit have been compared across time and key business groups."
            out["limitations"].append("The observed patterns identify where to investigate; they do not prove a causal explanation.")
            return out
        return _numeric_drivers(df,dataset,out)
    return _unsupported(out,"The requested analysis could not be matched to available fields.")
