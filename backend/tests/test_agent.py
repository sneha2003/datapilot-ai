from app.agents.planner import create_plan
from app.mcp.gateway import MCPToolGateway
from app.tools.database import resolve_dataset
from fastapi.testclient import TestClient
from app.main import app
from app.agents.presentation import present

def tools(plan): return [x["tool"] for x in plan]
def test_tool_selection_schema_is_minimal(): assert tools(create_plan("What columns are available in Titanic?",{"target":"survived"}))==["get_schema"]
def test_tool_selection_cleaning_requires_approval():
    plan=create_plan("Clean this dataset and fix missing values",{"target":"survived"}); assert tools(plan)==["profile_dataset","recommend_cleaning","preview_cleaning","apply_cleaning"] and plan[-1]["approval"]
def test_clean_and_preprocess_builds_two_reviewed_versions():
    plan=create_plan("Clean this dataset and do all preprocessing steps and make it ready for analysis",{"target":"survived"})
    assert tools(plan)==["profile_dataset","recommend_cleaning","preview_cleaning","apply_cleaning","recommend_features","preview_features","apply_features"]
    assert [step["tool"] for step in plan if step.get("approval")]==["apply_cleaning","apply_features"]
def test_tool_selection_training_workflow():
    selected=tools(create_plan("Train models to predict survival",{"target":"survived"})); assert selected==["profile_dataset","identify_problem","recommend_models","train_recommended"]
def test_mcp_gateway_invokes_real_tool(db):
    out=MCPToolGateway(db,resolve_dataset(db,"titanic")).call("profile_dataset"); assert out["shape"]["rows"]==12
def test_dataset_lineage_locates_active_version(db):
    out=MCPToolGateway(db,resolve_dataset(db,"titanic")).call("get_dataset_lineage")
    assert out["active"]["display_name"]=="Titanic" and "Data sources" in out["location"]
def test_location_followup_uses_dataset_lineage():
    assert tools(create_plan("Where can I find the newly created dataset?",{"target":"survived"}))==["get_dataset_lineage"]
def test_clean_preprocess_and_location_followup_end_to_end(db):
    client=TestClient(app); source=resolve_dataset(db,"titanic")
    first=client.post("/api/analyst/query",json={"query":"Clean this dataset and do all preprocessing steps and make it ready for analysis","dataset":source.id}).json()
    assert first["status"]=="awaiting_approval" and first["pending_approval"]["step"]["tool"]=="apply_cleaning"
    second=client.post("/api/analyst/continue",json={"session_id":first["session_id"],"run_id":first["run_id"],"decision":"approve"}).json()
    assert second["status"]=="awaiting_approval" and second["pending_approval"]["step"]["tool"]=="apply_features"
    third=client.post("/api/analyst/continue",json={"session_id":first["session_id"],"run_id":first["run_id"],"decision":"approve"}).json()
    assert third["status"]=="completed" and "Your data is ready" in third["answer"] and "New analysis fields" in third["answer"]
    assert third["context"]["dataset_id"] not in {source.id,second["context"]["dataset_id"]}
    followup=client.post("/api/analyst/query",json={"query":"Where can I find the newly created dataset?","session_id":first["session_id"],"dataset":third["context"]["dataset_id"]}).json()
    assert "Your current dataset" in followup["answer"] and "Data sources" in followup["answer"]
def test_followup_feature_question_uses_stored_experiment():
    plan=create_plan("What was the strongest feature?",{"target":"survived","experiments":[{"experiment_id":"exp-1"}]}); assert tools(plan)==["get_experiment_importance"]
def test_broad_beginner_question_uses_business_overview():
    assert tools(create_plan("Tell me about marketing",{"target":"y"}))==["dataset_overview"]
def test_business_overview_is_computed(db):
    out=MCPToolGateway(db,resolve_dataset(db,"titanic")).call("dataset_overview")
    assert out["rows"]==12 and out["target"]=="survived" and out["target_distribution"]
def test_bike_segment_question_compares_average_rentals():
    context={"dataset_name":"bike_sharing","target":"count","schema":{"columns":[{"name":name,"type":kind} for name,kind in [("season","TEXT"),("weather","TEXT"),("hour","INTEGER"),("count","INTEGER")]]}}
    plan=create_plan("What groups or segments stand out in Bike Sharing Demand?",context)
    assert tools(plan)==["grouped_analysis"]*3
    assert [step["args"]["group_by"] for step in plan]==["season","weather","hour"]
def test_business_answer_ignores_tiny_groups():
    answer=present({"analysis_results":[{"tool":"grouped_analysis","result":{"columns":["weather","mean_count","sample_size"],"rows":[{"weather":"clear","mean_count":200,"sample_size":1000},{"weather":"rain","mean_count":100,"sample_size":100},{"weather":"heavy_rain","mean_count":5,"sample_size":2}]}}]})
    assert "heavy rain" not in str(answer)
    assert "rain" in answer["sections"][0]["items"][0]
def test_customer_likelihood_question_uses_real_segment_analysis():
    plan=create_plan("Which customers are most likely to subscribe?",{"target":"y"})
    assert tools(plan)==["target_drivers"]
def test_unknown_business_question_never_returns_bare_profile():
    assert tools(create_plan("Help me understand what is happening",{"target":"survived"}))==["dataset_overview"]
    assert tools(create_plan("Is the moon made of cheese?",{"target":"survived"}))==["clarify_question"]
def test_full_analysis_uses_multiple_computed_tools():
    selected=tools(create_plan("Analyze this dataset and show the most important findings",{"target":"survived"}))
    assert selected==["business_insights"]
def test_data_quality_wording_is_understood():
    assert tools(create_plan("Does this data need cleaning?",{"target":"survived"}))==["profile_dataset","recommend_cleaning"]
def test_model_followups_reuse_saved_experiments():
    context={"target":"survived","experiments":[{"experiment_id":"one"},{"experiment_id":"two"}]}
    assert tools(create_plan("Show the confusion matrix",context))==["get_experiment_evaluation"]
    assert tools(create_plan("Which model performed best?",context))==["compare_saved_models"]
