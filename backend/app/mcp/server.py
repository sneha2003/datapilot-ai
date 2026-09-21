"""Run with: python -m app.mcp.server"""
from mcp.server.fastmcp import FastMCP
from app.database.base import SessionLocal
from app.tools.database import list_datasets, resolve_dataset, get_dataset_schema, get_dataset_lineage, sample_rows, execute_readonly_sql
from app.tools.profiling import profile_dataset, dataset_overview, missing_value_report, calculate_correlations
from app.tools.cleaning import recommend_cleaning_steps, preview_cleaning, apply_cleaning
from app.tools.features import recommend_features, preview_feature_engineering, apply_feature_engineering
from app.tools.model_recommendation import identify_ml_problem, recommend_models
from app.tools.statistics import grouped_analysis,trend_analysis,statistical_test,target_driver_analysis
from app.tools.visualization import create_visualization
from app.tools.training import train_model,compare_models,evaluate_model,feature_importance
from app.tools.business import analyze_business_question
from app.memory.store import retrieve_memory,save_memory

mcp=FastMCP("DataPilot Analysis Tools")
def with_dataset(name,fn,*args,**kwargs):
    with SessionLocal() as db:
        ds=resolve_dataset(db,name)
        if not ds: raise ValueError(f"Unknown dataset: {name}")
        return fn(db,ds,*args,**kwargs)

@mcp.tool()
def list_available_datasets():
    with SessionLocal() as db: return list_datasets(db)
@mcp.tool()
def get_dataset_schema(dataset_name:str): return with_dataset(dataset_name,get_dataset_schema)
@mcp.tool()
def get_dataset_version_lineage(dataset_name:str): return with_dataset(dataset_name,get_dataset_lineage)
@mcp.tool()
def sample_dataset_rows(dataset_name:str,limit:int=10): return with_dataset(dataset_name,sample_rows,limit)
@mcp.tool()
def execute_readonly_query(query:str):
    with SessionLocal() as db: return execute_readonly_sql(db,query)
@mcp.tool()
def profile_registered_dataset(dataset_name:str): return with_dataset(dataset_name,profile_dataset)
@mcp.tool()
def explain_registered_dataset(dataset_name:str): return with_dataset(dataset_name,dataset_overview)
@mcp.tool()
def investigate_business_question(dataset_name:str,question:str,intent:str="investigation"):
    return with_dataset(dataset_name,analyze_business_question,question,intent)
@mcp.tool()
def dataset_missing_values(dataset_name:str): return with_dataset(dataset_name,missing_value_report)
@mcp.tool()
def dataset_correlations(dataset_name:str,method:str="pearson"): return with_dataset(dataset_name,calculate_correlations,method)
@mcp.tool()
def recommend_dataset_cleaning(dataset_name:str): return with_dataset(dataset_name,recommend_cleaning_steps)
@mcp.tool()
def preview_dataset_cleaning(dataset_name:str,operations:list[dict]): return with_dataset(dataset_name,preview_cleaning,operations)
@mcp.tool()
def apply_approved_cleaning(dataset_name:str,operations:list[dict]): return with_dataset(dataset_name,apply_cleaning,operations)
@mcp.tool()
def recommend_dataset_features(dataset_name:str,target:str|None=None): return with_dataset(dataset_name,recommend_features,target)
@mcp.tool()
def preview_dataset_features(dataset_name:str,transformations:list[dict]): return with_dataset(dataset_name,preview_feature_engineering,transformations)
@mcp.tool()
def apply_approved_features(dataset_name:str,transformations:list[dict]): return with_dataset(dataset_name,apply_feature_engineering,transformations)
@mcp.tool()
def identify_problem(dataset_name:str,target:str,objective:str|None=None): return with_dataset(dataset_name,identify_ml_problem,target,objective)
@mcp.tool()
def recommend_candidate_models(dataset_name:str,target:str,task_type:str): return with_dataset(dataset_name,recommend_models,target,task_type)
@mcp.tool()
def analyze_groups(dataset_name:str,group_by:str,metric:str|None=None,aggregation:str="count",bins:list[float]|None=None): return with_dataset(dataset_name,grouped_analysis,group_by,metric,aggregation,None,bins)
@mcp.tool()
def analyze_outcome_drivers(dataset_name:str,target:str|None=None,limit:int=8): return with_dataset(dataset_name,target_driver_analysis,target,limit)
@mcp.tool()
def analyze_trend(dataset_name:str,date_column:str,metric:str|None=None,frequency:str="M",aggregation:str="count"): return with_dataset(dataset_name,trend_analysis,date_column,metric,frequency,aggregation)
@mcp.tool()
def run_statistical_test(dataset_name:str,test:str,columns:list[str]): return with_dataset(dataset_name,statistical_test,test,columns)
@mcp.tool()
def create_dataset_visualization(dataset_name:str,specification:dict): return with_dataset(dataset_name,create_visualization,specification)
@mcp.tool()
def train_dataset_model(dataset_name:str,target:str,task_type:str,algorithm:str,time_column:str|None=None): return with_dataset(dataset_name,train_model,target,task_type,algorithm,None,time_column)
@mcp.tool()
def compare_trained_models(results:list[dict]): return compare_models(results)
@mcp.tool()
def evaluate_trained_model(experiment_id:str):
    with SessionLocal() as db: return evaluate_model(db,experiment_id)
@mcp.tool()
def get_model_feature_importance(experiment_id:str):
    with SessionLocal() as db: return feature_importance(db,experiment_id)
@mcp.tool()
def retrieve_analysis_memory(query:str|None=None):
    with SessionLocal() as db: return retrieve_memory(db,query)
@mcp.tool()
def save_analysis_memory(memory_type:str,key:str,value:dict,confidence:float=1.0):
    with SessionLocal() as db:
        item=save_memory(db,memory_type,key,value,confidence); return {"id":item.id,"key":item.key}

if __name__=="__main__": mcp.run(transport="stdio")
