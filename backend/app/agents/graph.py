from datetime import datetime,timezone
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END,START,StateGraph
from langgraph.types import interrupt
from app.agents.planner import create_plan
from app.agents.presentation import present,as_text
from app.agents.state import AnalystState
from app.agents.supervisor import execute_step
from app.database.base import SessionLocal
from app.database.models import AnalysisRun, AnalysisSession
from app.memory.store import retrieve_memory,save_memory
from app.services.llm import get_llm
from app.config import get_settings
from app.tools.database import get_dataset_schema,resolve_dataset

def load_memory(state):
    with SessionLocal() as db:
        session=db.get(AnalysisSession,state["conversation_id"])
        if not session:
            session=AnalysisSession(id=state["conversation_id"],title=state["user_query"][:80]); db.add(session); db.commit()
        experiments=[{"experiment_id":x} for x in session.state.get("model_experiments",[])]
        return {"memory_context":retrieve_memory(db,state["user_query"],8),"active_dataset":state.get("active_dataset") or session.active_dataset_id,"completed_steps":[],"analysis_results":[],"generated_artifacts":[],"model_experiments":experiments,"errors":[],"trace":[{"node":"load_memory","status":"completed"}],"current_step":0,"retry_count":0,"status":"running"}

def understand_request(state):
    q=state["user_query"].lower(); task="analysis"
    for name,words in {"dataset_discovery":["what datasets","list datasets"],"schema":["what columns","schema"],"cleaning":["clean","quality"],"machine_learning":["predict","train","model","classification","regression"],"visualization":["plot","chart","visual"],"memory":["previously","earlier"]}.items():
        if any(w in q for w in words): task=name; break
    objective="regression" if "regression" in q else "multiclass_classification" if "classify" in q and "low" in q else None
    return {"task_type":task,"objective":objective,"trace":state["trace"]+[{"node":"understand_request","status":"completed","summary":task}]}

def resolve_context(state):
    with SessionLocal() as db:
        ds=resolve_dataset(db,state.get("active_dataset"))
        if not ds:
            for candidate in db.query(__import__('app.database.models',fromlist=['Dataset']).Dataset).all():
                if candidate.name.replace("_"," ") in state["user_query"].lower() or candidate.display_name.lower() in state["user_query"].lower(): ds=candidate; break
        updates={"trace":state["trace"]+[{"node":"resolve_dataset","status":"completed","summary":ds.display_name if ds else "No dataset required"}]}
        if ds:
            updates["active_dataset"]=ds.id; updates["target"]=state.get("target") or (ds.target_suggestions[0] if ds.target_suggestions else None)
        return updates

def inspect_context(state):
    if not state.get("active_dataset"): return {"trace":state["trace"]+[{"node":"inspect_dataset","status":"skipped"}]}
    with SessionLocal() as db:
        ds=resolve_dataset(db,state["active_dataset"]); schema=get_dataset_schema(db,ds)
    return {"dataset_schema":schema,"trace":state["trace"]+[{"node":"inspect_dataset","status":"completed","summary":f"{len(schema['columns'])} columns"}]}

def plan(state):
    context={"dataset":state.get("active_dataset"),"dataset_name":(state.get("dataset_schema") or {}).get("dataset",""),"schema":state.get("dataset_schema"),"target":state.get("target"),"task_type":state.get("task_type"),"memory":state.get("memory_context"),"experiments":state.get("model_experiments",[])}
    steps=create_plan(state["user_query"],context)[:get_settings().max_agent_steps]
    return {"analysis_plan":steps,"trace":state["trace"]+[{"node":"create_plan","status":"completed","summary":f"{len(steps)} steps"}]}

def route(state):
    if state["current_step"]>=len(state.get("analysis_plan",[])): return "synthesize"
    step=state["analysis_plan"][state["current_step"]]
    return "approval" if step.get("approval") and not state.get("approval_decision") else "execute"

def approval(state):
    step=state["analysis_plan"][state["current_step"]]; preview=state.get("analysis_results",[])[-1]["result"] if state.get("analysis_results") else {}
    decision=interrupt({"type":"transformation_approval","step":step,"preview":preview,"message":"Review the deterministic transformation preview before a new dataset version is created."})
    if isinstance(decision,str): decision={"decision":decision}
    if decision.get("decision") not in {"approve","modify"}:
        skipped={"tool":step["tool"],"label":step["label"],"status":"rejected","reason":decision.get("reason","")}
        return {"approval_decision":None,"pending_approval":None,"current_step":state["current_step"]+1,"completed_steps":state["completed_steps"]+[skipped],"trace":state["trace"]+[{"node":"human_approval","status":"rejected"}]}
    if decision.get("operations") is not None:
        state["analysis_plan"][state["current_step"]]["args"]={"operations":decision["operations"],"transformations":decision["operations"]}
    return {"approval_decision":decision,"pending_approval":None,"trace":state["trace"]+[{"node":"human_approval","status":"approved"}]}

def execute(state):
    with SessionLocal() as db:
        ds=resolve_dataset(db,state.get("active_dataset")); completed,result,error=execute_step(db,state,ds)
    updates={"completed_steps":state["completed_steps"]+[completed],"current_step":state["current_step"]+1,"approval_decision":None,"trace":state["trace"]+[{"node":"execute_tool","tool":completed["tool"],"status":completed["status"]}]}
    if result:
        updates["analysis_results"]=state["analysis_results"]+[{"tool":completed["tool"],"result":result}]
        if completed["tool"]=="identify_problem": updates|={"target":result["target"],"task_type":result["task_type"]}
        if completed["tool"].startswith("apply_"): updates["active_dataset"]=result["dataset_id"]
        if completed["tool"]=="train_recommended": updates["model_experiments"]=result["models"]
        if completed["tool"] in {"auto_visualization","visualize_last_result"}: updates["generated_artifacts"]=[result]
        if completed["tool"]=="business_insights" and result.get("visualization"): updates["generated_artifacts"]=[result["visualization"]]
    if error: updates["errors"]=state["errors"]+[error]
    return updates

def validate(state): return {"trace":state["trace"]+[{"node":"validate_result","status":"completed"}]}

def synthesize(state):
    structure=present(state)
    return {"final_answer":as_text(structure),"answer_structure":structure,"status":"completed" if not state.get("errors") else "completed_with_errors","trace":state["trace"]+[{"node":"synthesize","status":"completed"}]}

def _operation_text(operation):
    kind=operation.get("operation","")
    column=operation.get("column")
    if kind=="impute_numeric": return f"Filled missing {column.replace('_',' ')} values using the {operation.get('strategy','median')}"
    if kind=="impute_categorical": return f"Filled missing {column.replace('_',' ')} values using the most common category"
    if kind=="drop_duplicates": return "Removed exact duplicate rows"
    if kind=="drop_columns": return "Removed unusable columns: "+", ".join(operation.get("columns",[]))
    if kind=="standardize_categories": return f"Standardized spelling and capitalization in {column.replace('_',' ')}"
    if kind=="convert_type": return f"Converted {column.replace('_',' ')} to {operation.get('dtype','the correct type')}"
    if kind=="cap_outliers": return f"Limited extreme values in {column.replace('_',' ')}"
    return kind.replace("_"," ").capitalize()

def _deterministic_summary(state):
    if not state.get("analysis_results"): return "I could not produce a computed result. Select a dataset or clarify the analytical objective."
    computed={item["tool"]:item["result"] for item in state["analysis_results"]}
    grouped=[item for item in state["analysis_results"] if item["tool"]=="grouped_analysis" and item["result"].get("columns",[])[1:2]==[f"mean_{state.get('target')}"]]
    if len(grouped)>1:
        lines=["Here are the groups that stand out in the recorded data. These are average rentals per hour, so groups with more recorded hours do not automatically rank higher."]
        for item in grouped:
            result=item["result"]; group_column,metric_column=result["columns"][:2]
            rows=[row for row in result["rows"] if isinstance(row.get(metric_column),(int,float))]
            if not rows: continue
            high=max(rows,key=lambda row:row[metric_column]); low=min(rows,key=lambda row:row[metric_column])
            label={"season":"Season","weather":"Weather","hour":"Time of day"}.get(group_column,group_column.replace("_"," ").capitalize())
            lines.append(f"{label}: {high[group_column]} had the highest average ({high[metric_column]:,.1f} rentals per hour); {low[group_column]} had the lowest ({low[metric_column]:,.1f}).")
        lines.append("These are historical differences, not proof that a season, weather condition, or hour caused demand to change.")
        return "\n\n".join(lines)
    if "apply_cleaning" in computed:
        applied=computed["apply_cleaning"]; preview=computed.get("preview_cleaning",{}); recommended=computed.get("recommend_cleaning",{})
        lines=[f"Cleaning complete. I created “{applied.get('display_name',applied['name'])}” as a separate dataset and kept the original unchanged."]
        operations=recommended.get("operations",[])
        if operations: lines.append("Changes applied:\n"+"\n".join(f"• {_operation_text(operation)}" for operation in operations))
        if preview:
            before,after=preview.get("before",{}),preview.get("after",{})
            lines.append(f"Result: {after.get('rows',0):,} rows and {after.get('columns',0)} columns. Missing cells changed from {before.get('missing',0):,} to {after.get('missing',0):,}.")
        if "apply_features" in computed:
            featured=computed["apply_features"]; added=featured.get("added_columns",[])
            lines.append(f"Preprocessing complete. I also created “{featured.get('display_name',featured['name'])}” with {len(added)} new analysis fields"+(f": {', '.join(x.replace('_',' ') for x in added)}." if added else "."))
        lines.append("You can find the new version in the Data sources list on the left. It is now the active dataset for your next question.")
        return "\n\n".join(lines)
    lines=[]
    for item in state["analysis_results"]:
        name,r=item["tool"],item["result"]
        if name=="list_datasets": lines.append(f"{len(r)} datasets are registered: "+", ".join(x["display_name"] for x in r)+".")
        elif name=="get_schema": lines.append(f"{r['dataset']} has {len(r['columns'])} columns: "+", ".join(x["name"] for x in r["columns"])+".")
        elif name=="get_dataset_lineage":
            active=r["active"]
            lines.append(f"Your active dataset is “{active['display_name']}” ({active['version']}) with {active['rows']:,} rows and {active['columns']} columns.")
            lines.append(f"You can find it in the {r['location']}.")
            if r.get("parent"): lines.append(f"It was created from “{r['parent']['display_name']}”. The original dataset remains unchanged.")
        elif name=="dataset_overview":
            lines.append(f"{r['display_name']} contains {r['rows']:,} real records and {r['columns']} pieces of information for each record. {r['description']}")
            if r.get("target"):
                lines.append(f"The main result column is '{r['target']}': {r.get('target_description') or 'the outcome we may want to understand or predict.'}")
            if r.get("target_distribution"):
                parts=[f"{x['count']:,} are {x['value']} ({x['percentage']}%)" for x in r['target_distribution']]
                lines.append("For that result, "+" and ".join(parts)+".")
            if r.get("target_statistics"):
                stats=r["target_statistics"]
                lines.append(f"The measured {r['target'].replace('_',' ')} averages {stats['mean']:,.1f} per record, with a median of {stats['median']:,.1f} and a range from {stats['minimum']:,.0f} to {stats['maximum']:,.0f}.")
            friendly=[x for x in r.get("numeric_highlights",[]) if x["column"] in {"age","balance","duration","campaign","median_house_value","count","quality"}]
            if friendly:
                lines.append("A few measured facts: "+"; ".join(f"average {x['column'].replace('_',' ')} is {x['mean']:,}" for x in friendly[:4])+".")
            lines.append("The available information includes "+", ".join(x.replace("_"," ") for x in r["column_names"][:12])+(" and more." if len(r["column_names"])>12 else "."))
            lines.append(f"Data check: {r['missing_pct']}% of cells are missing and there are {r['duplicate_rows']:,} exact duplicate rows. The source is {r['source']}.")
        elif name=="profile_dataset": lines.append(f"Profiled {r['shape']['rows']:,} rows and {r['shape']['columns']} columns. Missing cells: {r['missing_pct']}%; exact duplicates: {r['duplicate_rows']}.")
        elif name=="recommend_cleaning": lines.append(f"Found {len(r['issues'])} quality signals and proposed {len(r['operations'])} deterministic cleaning operations.")
        elif name.startswith("preview_"): lines.append("The transformation preview was computed without changing the source dataset.")
        elif name.startswith("apply_"): lines.append(f"Created immutable dataset version {r['version']} with recorded lineage.")
        elif name=="identify_problem": lines.append(f"This objective is {r['task_type'].replace('_',' ')} with target {r['target']}.")
        elif name=="recommend_models": lines.append("Recommended candidates: "+", ".join(x["algorithm"].replace("_"," ").title() for x in r["candidates"])+".")
        elif name=="train_recommended":
            c=r["comparison"]; lines.append(f"Trained {len(r['models'])} real pipelines with five-fold cross-validation. {c['best_algorithm'].replace('_',' ').title()} ranked first by {c['primary_metric'].upper()}. {c['selection_note']}")
            best=next(x for x in r["models"] if x["experiment_id"]==c["best_experiment_id"]); lines.append("Best holdout metrics: "+", ".join(f"{k}={v:.3f}" for k,v in best["metrics"].items())+".")
            if best["feature_importance"]: lines.append("Strongest permutation features: "+", ".join(x["feature"] for x in best["feature_importance"][:5])+".")
        elif name=="get_experiment_importance":
            lines.append("Strongest permutation features for the latest model: "+", ".join(x["feature"] for x in r["feature_importance"][:5])+"." if r["feature_importance"] else "The stored experiment has no feature-importance artifact.")
        elif name=="get_experiment_evaluation":
            matrix=r.get("cross_validation",{}).get("confusion_matrix")
            lines.append(f"The latest model is {r['algorithm'].replace('_',' ').title()}. Measured results: "+", ".join(f"{key.replace('_',' ')}={value:.3f}" for key,value in r["metrics"].items())+".")
            if matrix and len(matrix)==2: lines.append(f"Confusion matrix: {matrix[0][0]:,} correct negatives, {matrix[0][1]:,} false alarms, {matrix[1][0]:,} missed positives, and {matrix[1][1]:,} correct positives.")
        elif name=="compare_saved_models":
            lines.append(f"{r['best_algorithm'].replace('_',' ').title()} performed best using {r['primary_metric'].upper()}. {r['selection_note']}")
            lines.append("Comparison: "+"; ".join(f"{row['algorithm'].replace('_',' ').title()} {r['primary_metric'].upper()}={row.get(r['primary_metric'],0):.3f}" for row in r["rows"])+".")
        elif name=="correlations": lines.append(f"Calculated the {r['method']} correlation matrix across {len(r['columns'])} numeric columns; these are associations, not causal effects.")
        elif name=="grouped_analysis":
            rows=r["rows"]; value_column=r["columns"][1] if len(r["columns"])>1 else None; leader=max(rows,key=lambda x:x.get(value_column,float('-inf'))) if rows and value_column else None
            lines.append(f"Computed {len(rows)} groups"+(f". The highest {value_column.replace('_',' ')} is {leader.get(value_column)} for {leader.get(r['columns'][0])}." if leader else "."))
        elif name=="target_drivers":
            lines.append(f"Across all {r['total_customers']:,} customers, {r['baseline_rate_pct']}% had the outcome “{r['positive_label']}”.")
            if r["top_segments"]:
                lines.append("The customer groups with the highest historical rates were:")
                for segment in r["top_segments"][:6]:
                    field=segment["feature"].replace("_"," ")
                    lines.append(f"• {field} = {segment['segment']}: {segment['positive_rate_pct']}% ({segment['positive_customers']:,} of {segment['customers']:,} customers), {segment['lift']}× the overall rate")
            if r.get("strongest_fields"):
                lines.append("The largest differences between groups appeared in "+", ".join(x["feature"].replace("_"," ") for x in r["strongest_fields"][:3])+".")
            lines.append(r["interpretation"])
        elif name=="auto_visualization": lines.append(f"Generated the interactive chart “{r['title']}” from {r['data_rows']:,} plotted records.")
        elif name=="visualize_last_result": lines.append(f"Generated the interactive chart “{r['title']}” from the computed group table.")
        elif name=="retrieve_memory": lines.append(f"Retrieved {len(r)} relevant structured memory entries." if r else "No matching project memory was found.")
    if state.get("errors"): lines.append("Some steps could not complete: "+"; ".join(e["message"] for e in state["errors"]))
    return "\n\n".join(lines)

def save(state):
    with SessionLocal() as db:
        run=db.get(AnalysisRun,state["run_id"]); run.plan=state.get("analysis_plan",[]); run.findings=[{"tool":x["tool"],"summary":str(x["result"])[:1000]} for x in state.get("analysis_results",[])]+[{"tool":"final_answer","summary":state.get("final_answer","")}]; run.trace=state.get("trace",[]); run.status=state["status"]; run.completed_at=datetime.now(timezone.utc)
        session=db.get(AnalysisSession,state["conversation_id"]); session.active_dataset_id=state.get("active_dataset"); session.state={"target":state.get("target"),"task_type":state.get("task_type"),"last_run_id":state["run_id"],"model_experiments":[x["experiment_id"] for x in state.get("model_experiments",[])]};
        save_memory(db,"analysis_summary",f"session:{session.id}",{"query":state["user_query"],"answer":state["final_answer"],"dataset_id":state.get("active_dataset"),"experiments":[x["experiment_id"] for x in state.get("model_experiments",[])]},.9,session.id); db.commit()
    return {"trace":state["trace"]+[{"node":"save_memory","status":"completed"}]}

builder=StateGraph(AnalystState)
for name,node in [("load_memory",load_memory),("understand",understand_request),("resolve",resolve_context),("inspect",inspect_context),("plan",plan),("approval",approval),("execute",execute),("validate",validate),("synthesize",synthesize),("save",save)]: builder.add_node(name,node)
builder.add_edge(START,"load_memory"); builder.add_edge("load_memory","understand"); builder.add_edge("understand","resolve"); builder.add_edge("resolve","inspect"); builder.add_edge("inspect","plan")
builder.add_conditional_edges("plan",route,{"approval":"approval","execute":"execute","synthesize":"synthesize"}); builder.add_conditional_edges("approval",route,{"approval":"approval","execute":"execute","synthesize":"synthesize"}); builder.add_edge("execute","validate"); builder.add_conditional_edges("validate",route,{"approval":"approval","execute":"execute","synthesize":"synthesize"}); builder.add_edge("synthesize","save"); builder.add_edge("save",END)
analyst_graph=builder.compile(checkpointer=MemorySaver())
