import io,json,math,os,re,threading,uuid
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from fastapi import Depends,FastAPI,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from sqlalchemy import func,text
from sqlalchemy.orm import Session
from app.agents.graph import analyst_graph
from app.config import get_settings
from app.database.base import Base,engine,get_db
from app.database.models import AnalysisRun,AnalysisSession,Dataset,GeneratedArtifact,MemoryEntry,ModelExperiment,ToolExecution
from app.memory.store import clear_memory,forget_memory,retrieve_memory
from app.schemas import AnalystContinue,AnalystQuery
from app.tools.database import get_dataset_schema,list_datasets,resolve_dataset,sample_rows
from app.tools.profiling import profile_dataset,quality_issues

settings=get_settings(); app=FastAPI(title="DataPilot AI API",version="1.0.0",description="Autonomous data analyst with deterministic computation")
app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origin_list,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine); settings.artifact_root.mkdir(parents=True,exist_ok=True)
    if os.getenv("AUTO_SEED_DATASETS")=="1":
        from scripts.load_datasets import main as seed_public_datasets
        threading.Thread(target=seed_public_datasets,kwargs={"only_missing":True},daemon=True,name="public-dataset-seed").start()

def _json_safe(value):
    """Replace values that JSON cannot represent without changing valid results."""
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {str(key):_json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [_json_safe(item) for item in value]
    return value

def _result(state):
    interrupts=state.get("__interrupt__",()) if isinstance(state,dict) else ()
    pending=interrupts[0].value if interrupts else None
    return _json_safe({"session_id":state.get("conversation_id"),"run_id":state.get("run_id"),"status":"awaiting_approval" if pending else state.get("status","running"),"answer":state.get("final_answer"),"answer_structure":state.get("answer_structure"),"plan":state.get("analysis_plan",[]),"steps":state.get("completed_steps",[]),"trace":state.get("trace",[]),"artifacts":state.get("generated_artifacts",[]),"experiments":state.get("model_experiments",[]),"errors":state.get("errors",[]),"pending_approval":pending,"context":{"dataset_id":state.get("active_dataset"),"target":state.get("target"),"task_type":state.get("task_type")}})

@app.post("/api/analyst/query")
def analyst_query(body:AnalystQuery,db:Session=Depends(get_db)):
    session_id=body.session_id or str(uuid.uuid4()); session=db.get(AnalysisSession,session_id)
    if not session: session=AnalysisSession(id=session_id,title=body.query[:80]); db.add(session); db.commit()
    run=AnalysisRun(session_id=session_id,query=body.query); db.add(run); db.commit(); db.refresh(run)
    initial={"conversation_id":session_id,"run_id":run.id,"user_query":body.query,"active_dataset":body.dataset or session.active_dataset_id,"target":body.target or session.state.get("target"),"errors":[]}
    try: state=analyst_graph.invoke(initial,config={"configurable":{"thread_id":session_id}}); return _result(state)
    except Exception as exc:
        run.status="failed"; run.error=str(exc); run.completed_at=datetime.now(timezone.utc); db.commit(); raise HTTPException(422,str(exc))

@app.post("/api/analyst/continue")
def analyst_continue(body:AnalystContinue):
    payload={"decision":body.decision,"reason":body.reason}
    if body.operations is not None: payload["operations"]=body.operations
    try: state=analyst_graph.invoke(Command(resume=payload),config={"configurable":{"thread_id":body.session_id}}); return _result(state)
    except Exception as exc: raise HTTPException(422,str(exc))

@app.get("/api/analyst/sessions")
def sessions(db:Session=Depends(get_db)):
    return [{"id":x.id,"title":x.title,"active_dataset_id":x.active_dataset_id,"state":x.state,"created_at":x.created_at,"updated_at":x.updated_at} for x in db.query(AnalysisSession).order_by(AnalysisSession.updated_at.desc()).all()]
@app.get("/api/analyst/sessions/{session_id}")
def session_detail(session_id:str,db:Session=Depends(get_db)):
    s=db.get(AnalysisSession,session_id)
    if not s: raise HTTPException(404,"Session not found")
    runs=db.query(AnalysisRun).filter_by(session_id=session_id).order_by(AnalysisRun.created_at.desc()).all()
    memory=db.query(MemoryEntry).filter_by(session_id=session_id,memory_type="analysis_summary").order_by(MemoryEntry.updated_at.desc()).first()
    latest_memory=memory.value if memory else {}
    items=[]
    for index,r in enumerate(runs):
        saved=next((item.get("summary") for item in (r.findings or []) if item.get("tool")=="final_answer"),None)
        answer=saved or (latest_memory.get("answer") if index==0 and latest_memory.get("query")==r.query else None)
        items.append({"id":r.id,"query":r.query,"answer":answer,"plan":r.plan,"findings":r.findings,"trace":r.trace,"status":r.status,"created_at":r.created_at,"completed_at":r.completed_at})
    return {"id":s.id,"title":s.title,"state":s.state,"runs":items}

@app.get("/api/datasets")
def datasets(db:Session=Depends(get_db)): return list_datasets(db)

@app.post("/api/datasets/upload")
async def upload_dataset(request:Request,name:str="Uploaded dataset",db:Session=Depends(get_db)):
    raw=await request.body()
    if not raw: raise HTTPException(400,"The uploaded CSV is empty")
    if len(raw)>25*1024*1024: raise HTTPException(413,"The CSV must be smaller than 25 MB")
    try: frame=pd.read_csv(io.BytesIO(raw))
    except Exception as exc: raise HTTPException(400,f"The file could not be read as CSV: {exc}")
    if frame.empty or not len(frame.columns): raise HTTPException(400,"The CSV has no usable rows or columns")
    if len(frame)>1_000_000: raise HTTPException(413,"The CSV contains more than 1,000,000 rows")
    used=set(); renamed=[]
    for original in frame.columns:
        base=re.sub(r"[^a-z0-9_]+","_",str(original).strip().lower()).strip("_") or "column"
        candidate=base; suffix=2
        while candidate in used: candidate=f"{base}_{suffix}"; suffix+=1
        used.add(candidate); renamed.append(candidate)
    frame.columns=renamed
    base_name=re.sub(r"[^a-z0-9_]+","_",name.strip().lower()).strip("_") or "uploaded_dataset"
    dataset_name=base_name; suffix=2
    while db.query(Dataset).filter_by(name=dataset_name).first(): dataset_name=f"{base_name}_{suffix}"; suffix+=1
    table_name=f"upload_{dataset_name}_raw"
    frame.to_sql(table_name,db.bind,if_exists="fail",index=False,chunksize=5000)
    item=Dataset(name=dataset_name,display_name=name.strip() or "Uploaded dataset",table_name=table_name,description=f"CSV uploaded by the user with {len(frame):,} records.",source="User upload",license="User-provided data",row_count=len(frame),column_count=len(frame.columns),task_candidates=["business_analysis"],target_suggestions=[],column_descriptions={},version="raw")
    db.add(item); db.commit(); db.refresh(item)
    return next(entry for entry in list_datasets(db) if entry["id"]==item.id)
@app.get("/api/datasets/{dataset_id}")
def dataset_detail(dataset_id:str,db:Session=Depends(get_db)):
    ds=resolve_dataset(db,dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found")
    return {"dataset":next(x for x in list_datasets(db) if x["id"]==ds.id),"schema":get_dataset_schema(db,ds),"sample":sample_rows(db,ds,10)}
@app.get("/api/datasets/{dataset_id}/profile")
def dataset_profile(dataset_id:str,db:Session=Depends(get_db)):
    ds=resolve_dataset(db,dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found")
    p=profile_dataset(db,ds); return {**p,"issues":quality_issues(p)}
@app.get("/api/datasets/{dataset_id}/versions")
def dataset_versions(dataset_id:str,db:Session=Depends(get_db)):
    ds=resolve_dataset(db,dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found")
    root=ds
    while root.parent_id: root=db.get(Dataset,root.parent_id)
    all_ds=db.query(Dataset).all(); ids={root.id}; changed=True
    while changed:
        before=len(ids); ids|={x.id for x in all_ds if x.parent_id in ids}; changed=len(ids)>before
    return [{"id":x.id,"name":x.name,"version":x.version,"parent_id":x.parent_id,"rows":x.row_count,"columns":x.column_count,"transformations":x.transformations,"created_at":x.imported_at} for x in all_ds if x.id in ids]

@app.get("/api/experiments")
def experiments(db:Session=Depends(get_db)):
    return [{"id":x.id,"dataset_id":x.dataset_id,"target":x.target,"task_type":x.task_type,"algorithm":x.algorithm,"metrics":x.metrics,"duration_ms":x.duration_ms,"created_at":x.created_at} for x in db.query(ModelExperiment).order_by(ModelExperiment.created_at.desc()).all()]
@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id:str,db:Session=Depends(get_db)):
    x=db.get(ModelExperiment,experiment_id)
    if not x: raise HTTPException(404,"Experiment not found")
    return {"id":x.id,"dataset_id":x.dataset_id,"session_id":x.session_id,"target":x.target,"task_type":x.task_type,"algorithm":x.algorithm,"parameters":x.parameters,"metrics":x.metrics,"cross_validation":x.cv_results,"artifact_path":x.artifact_path,"artifact_exists":Path(x.artifact_path).exists(),"duration_ms":x.duration_ms,"created_at":x.created_at}
@app.get("/api/visualizations")
def visualizations(db:Session=Depends(get_db)):
    return [{"id":x.id,"title":x.title,"dataset_id":x.dataset_id,"specification":{k:v for k,v in (x.specification or {}).items() if k!="_figure"},"path":x.path,"created_at":x.created_at} for x in db.query(GeneratedArtifact).filter_by(kind="visualization").order_by(GeneratedArtifact.created_at.desc()).all()]
@app.get("/api/visualizations/{visualization_id}")
def visualization_detail(visualization_id:str,db:Session=Depends(get_db)):
    item=db.get(GeneratedArtifact,visualization_id)
    if not item or item.kind!="visualization": raise HTTPException(404,"Chart not found")
    figure=(item.specification or {}).get("_figure")
    if figure is None:
        path=Path(item.path or "").resolve(); root=(settings.artifact_root/"plots").resolve()
        if root not in path.parents or not path.exists(): raise HTTPException(404,"The saved chart file is unavailable")
        try: figure=json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc: raise HTTPException(500,f"The chart could not be opened: {exc}")
    dataset=db.get(Dataset,item.dataset_id) if item.dataset_id else None
    return {"id":item.id,"title":item.title,"dataset_id":item.dataset_id,"dataset_name":dataset.display_name if dataset else None,"specification":{k:v for k,v in (item.specification or {}).items() if k!="_figure"},"figure":figure,"created_at":item.created_at}
@app.get("/api/memory")
def memory(q:str|None=None,db:Session=Depends(get_db)): return retrieve_memory(db,q,100)
@app.delete("/api/memory/{memory_id}")
def delete_memory(memory_id:str,db:Session=Depends(get_db)):
    if not forget_memory(db,memory_id): raise HTTPException(404,"Memory entry not found")
    return {"deleted":True}
@app.delete("/api/memory")
def delete_all_memory(db:Session=Depends(get_db)): return {"deleted":clear_memory(db)}
@app.get("/api/system/health")
def health(db:Session=Depends(get_db)):
    try: db.execute(text("SELECT 1")); database="healthy"
    except Exception: database="unavailable"
    return {"status":"healthy" if database=="healthy" else "degraded","database":database,"storage":"PostgreSQL" if engine.dialect.name=="postgresql" else "Local recovery database","llm_provider":settings.llm_provider,"mcp":"available","version":"1.0.0"}
@app.get("/api/system/metrics")
def metrics(db:Session=Depends(get_db)):
    total=db.query(AnalysisRun).count(); completed=db.query(AnalysisRun).filter(AnalysisRun.status.like("completed%")).count(); tools=db.query(ToolExecution).count(); failures=db.query(ToolExecution).filter_by(success=False).count()
    return {"analysis_runs":total,"task_completion_rate":completed/total if total else None,"tool_executions":tools,"invalid_tool_call_rate":failures/tools if tools else None,"experiments":db.query(ModelExperiment).count(),"datasets":db.query(Dataset).count(),"note":"Rates are null until observed runs exist; no synthetic scores are reported."}

@app.exception_handler(Exception)
async def unhandled(_,exc): return JSONResponse(status_code=500,content={"error":{"code":"internal_error","message":str(exc)}})

# The cloud image serves the built React app from the same origin as the API.
frontend_dist=Path(__file__).resolve().parents[1]/"frontend"/"dist"
if frontend_dist.is_dir():
    app.mount("/assets",StaticFiles(directory=frontend_dist/"assets"),name="frontend-assets")
    @app.get("/{page_path:path}",include_in_schema=False)
    def frontend_page(page_path:str):
        if page_path.startswith("api/"): raise HTTPException(404,"Not found")
        return FileResponse(frontend_dist/"index.html")
