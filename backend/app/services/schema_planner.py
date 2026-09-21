"""Small, schema-aware parser for explicit questions when no LLM is configured.

It only plans calculations whose fields are present in the selected dataset.
Questions outside this bounded grammar fall through to the other planner rules.
"""
import re

AGGREGATIONS={
    "mean":("average","mean","avg"),
    "sum":("total","sum","combined"),
    "median":("median",),
    "count":("count","number of","how many"),
}


def _mentions(query,columns):
    normalized=re.sub(r"[^a-z0-9]+"," ",query.lower()).strip()
    found=[]
    for column in columns:
        label=column["name"].replace("_"," ").lower()
        match=re.search(r"(?<!\w)"+re.escape(label)+r"(?!\w)",normalized)
        if match: found.append((match.start(),match.end(),column))
    return found


def plan_group_question(query,context):
    schema=(context.get("schema") or {}).get("columns",[])
    if not schema or not re.search(r"\bby\b",query,re.I): return None
    normalized=re.sub(r"[^a-z0-9]+"," ",query.lower()).strip()
    mentioned=_mentions(query,schema)
    if not mentioned: return None
    by=re.search(r"\bby\b",normalized)
    groups=[item for item in mentioned if item[0]>by.start()]
    if not groups: return None
    group=max(groups,key=lambda item:len(item[2]["name"]))[2]["name"]
    measures=[item[2] for item in mentioned if item[1]<=by.start() and item[2]["name"]!=group]
    operation=next((key for key,words in AGGREGATIONS.items() if any(re.search(r"\b"+re.escape(word)+r"\b",query,re.I) for word in words)),None)
    if not operation: return None
    numeric=lambda column:any(part in column["type"].lower() for part in ("int","float","numeric","double","real","decimal"))
    if operation=="count":
        metric=None
    else:
        candidates=[column for column in measures if numeric(column)]
        if len(candidates)!=1:
            return [{"tool":"clarify_question","label":"Find a numeric measure in the selected dataset"}]
        metric=candidates[0]["name"]
    args={"group_by":group,"aggregation":operation}
    if metric: args["metric"]=metric
    steps=[{"tool":"grouped_analysis","label":f"Calculate {operation} {metric or 'records'} by {group}","args":args}]
    if re.search(r"\b(plot|chart|graph|visualize)\b",query,re.I):
        steps.append({"tool":"visualize_last_result","label":"Chart the measured comparison","args":{"title":f"{(metric or 'Records').replace('_',' ').title()} by {group.replace('_',' ')}"}})
    return steps
