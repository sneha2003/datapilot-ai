from pathlib import Path
from app.memory.store import forget_memory,retrieve_memory,save_memory
from app.tools.cleaning import apply_cleaning,preview_cleaning,recommend_cleaning_steps
from app.tools.database import resolve_dataset
from app.tools.features import preview_feature_engineering
from app.tools.model_recommendation import identify_ml_problem,recommend_models
from app.tools.profiling import calculate_correlations,profile_dataset,dataset_overview
from app.tools.training import compare_models,train_model
from app.tools.visualization import create_visualization
from app.tools.statistics import grouped_analysis,target_driver_analysis

def test_profile_computes_missing_and_distributions(db):
    p=profile_dataset(db,resolve_dataset(db,"titanic")); age=next(x for x in p["columns"] if x["name"]=="age")
    assert p["shape"]=={"rows":12,"columns":7} and age["missing"]==1 and age["statistics"]["median"] is not None
def test_numeric_outcome_overview_is_bounded(db):
    ds=resolve_dataset(db,"california_housing")
    overview=dataset_overview(db,ds)
    assert overview["target_distribution"]==[] and overview["target_statistics"]["mean"]>0
def test_cleaning_preview_then_versioned_apply(db):
    ds=resolve_dataset(db,"titanic"); ops=[{"operation":"impute_numeric","column":"age","strategy":"median"}]
    preview=preview_cleaning(db,ds,ops); assert preview["before"]["missing"]==1 and preview["after"]["missing"]==0
    result=apply_cleaning(db,ds,ops); assert result["version"]=="clean_v1" and result["lineage"]["parent_id"]==ds.id
def test_feature_preview_is_deterministic(db):
    ds=resolve_dataset(db,"titanic"); ops=[{"operation":"family_size","sibsp":"sibsp","parch":"parch"},{"operation":"is_alone","from":"family_size"}]
    result=preview_feature_engineering(db,ds,ops); assert result["added_columns"]==["family_size","is_alone"]
def test_problem_identification_and_recommendation(db):
    ds=resolve_dataset(db,"titanic"); problem=identify_ml_problem(db,ds,"survived"); rec=recommend_models(db,ds,"survived",problem["task_type"])
    assert problem["task_type"]=="binary_classification" and 2<=len(rec["candidates"])<=5
def test_classification_training_persists_real_artifact(db):
    ds=resolve_dataset(db,"titanic"); result=train_model(db,ds,"survived","binary_classification","logistic_regression")
    assert 0<=result["metrics"]["f1"]<=1 and len(result["cross_validation"]["f1"]["folds"])==5 and Path(result["artifact_path"]).exists()
def test_regression_training_and_comparison(db):
    ds=resolve_dataset(db,"california_housing"); a=train_model(db,ds,"median_house_value","regression","ridge"); b=train_model(db,ds,"median_house_value","regression","random_forest"); comparison=compare_models([a,b])
    assert comparison["primary_metric"]=="rmse" and comparison["best_experiment_id"] in {a["experiment_id"],b["experiment_id"]}
def test_visualization_is_actual_plotly_json(db):
    ds=resolve_dataset(db,"titanic"); out=create_visualization(db,ds,{"chart_type":"histogram","x":"age","title":"Age"})
    assert out["figure"]["data"] and Path(out["path"]).exists()
def test_correlation_and_memory(db):
    ds=resolve_dataset(db,"california_housing"); corr=calculate_correlations(db,ds); assert len(corr["matrix"])==4
    item=save_memory(db,"analysis_preference","classification_metric",{"value":"f1"}); assert retrieve_memory(db,"classification metric")[0]["value"]["value"]=="f1"; assert forget_memory(db,item.id)
def test_grouped_positive_rate_is_computed(db):
    out=grouped_analysis(db,resolve_dataset(db,"titanic"),"pclass","survived","positive_rate")
    assert out["columns"]==["pclass","positive_rate","sample_size"] and len(out["rows"])==3
def test_grouped_mean_includes_sample_sizes(db):
    out=grouped_analysis(db,resolve_dataset(db,"titanic"),"pclass","age","mean")
    assert out["columns"]==["pclass","mean_age","sample_size"]
    assert sum(row["sample_size"] for row in out["rows"])==12
def test_target_driver_analysis_computes_ranked_segments(db):
    out=target_driver_analysis(db,resolve_dataset(db,"titanic"),"survived")
    assert out["baseline_rate_pct"]==50.0
    assert out["top_segments"] and out["top_segments"][0]["customers"]>0
    assert all("positive_rate_pct" in row and "lift" in row for row in out["top_segments"])
