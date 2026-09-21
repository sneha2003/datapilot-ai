import time
from datetime import datetime,timezone
from sqlalchemy.orm import Session
from app.database.models import AnalysisRun, GeneratedArtifact, ToolExecution
from app.mcp.gateway import MCPToolGateway
from app.agents.validator import validate_tool_result

def execute_step(db:Session,state,dataset):
    step=state["analysis_plan"][state["current_step"]]; started=time.perf_counter(); name=step["tool"]
    context=_context(state); args=dict(step.get("args") or {})
    try:
        result=MCPToolGateway(db,dataset).call(name,args,context); valid,error=validate_tool_result(name,result)
        if not valid: raise ValueError(error)
        duration=(time.perf_counter()-started)*1000
        if name in {"auto_visualization","create_visualization","visualize_last_result"}:
            db.add(GeneratedArtifact(session_id=state["conversation_id"],dataset_id=dataset.id,kind="visualization",title=result["title"],specification={**result["specification"],"_figure":result["figure"]},path=result["path"]))
        if name=="business_insights" and result.get("visualization"):
            chart=result["visualization"]
            db.add(GeneratedArtifact(session_id=state["conversation_id"],dataset_id=dataset.id,kind="visualization",title=chart["title"],specification={**chart["specification"],"_figure":chart["figure"]},path=chart["path"]))
        db.add(ToolExecution(run_id=state["run_id"],tool_name=name,arguments=args,result_summary=_summary(result),duration_ms=duration,success=True,retry=state.get("retry_count",0))); db.commit()
        completed={"tool":name,"label":step["label"],"status":"completed","duration_ms":round(duration,1),"result":result}
        return completed,result,None
    except Exception as exc:
        duration=(time.perf_counter()-started)*1000; error={"tool":name,"message":str(exc),"retry":state.get("retry_count",0)}
        db.add(ToolExecution(run_id=state["run_id"],tool_name=name,arguments=args,result_summary={"error":str(exc)},duration_ms=duration,success=False,retry=state.get("retry_count",0))); db.commit()
        return {"tool":name,"label":step["label"],"status":"failed","duration_ms":round(duration,1),"error":str(exc)},None,error

def _context(state):
    ctx={"session_id":state["conversation_id"],"query":state["user_query"],"schema":state.get("dataset_schema",{}),"target":state.get("target"),"task_type":state.get("task_type"),"objective":state.get("objective"),"experiments":state.get("model_experiments",[])}
    for item in state.get("analysis_results",[]):
        name=item["tool"]; value=item["result"]
        if name=="recommend_cleaning": ctx["cleaning_operations"]=value["operations"]
        elif name=="recommend_features": ctx["feature_operations"]=value["transformations"]
        elif name=="identify_problem": ctx.update(target=value["target"],task_type=value["task_type"])
        elif name=="recommend_models": ctx["candidates"]=value["candidates"]
        elif name=="grouped_analysis": ctx["last_table"]=value
    return ctx

def _summary(result):
    if isinstance(result,list): return {"items":len(result)}
    if not isinstance(result,dict): return {"value":str(result)[:500]}
    return {k:v for k,v in result.items() if k in {"dataset","shape","row_count","missing_pct","duplicate_rows","task_type","target","algorithm","metrics","best_algorithm","version","data_rows","requires_approval","headline","status","intent"}}
