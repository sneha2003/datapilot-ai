import pytest
from types import SimpleNamespace
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app
from app.services import llm
from app.agents.planner import create_plan,normalize_plan


SCHEMA={"columns":[{"name":"product_type","type":"TEXT"},{"name":"sale_price","type":"FLOAT"},{"name":"store_region","type":"TEXT"}]}


def test_uploaded_dataset_columns_drive_real_group_calculation():
    context={"schema":SCHEMA,"dataset_name":"uploaded_sales"}
    plan=create_plan("Show average sale price by product type as a chart",context)
    assert [step["tool"] for step in plan]==["grouped_analysis","visualize_last_result"]
    assert plan[0]["args"]=={"group_by":"product_type","metric":"sale_price","aggregation":"mean"}
    assert create_plan("Count records by store region",context)[0]["args"]=={"group_by":"store_region","aggregation":"count"}


def test_ambiguous_or_missing_field_does_not_invent_a_calculation():
    assert create_plan("Show average profit by product type",{"schema":SCHEMA})[0]["tool"]=="clarify_question"
    with pytest.raises(ValueError,match="not present"):
        normalize_plan([{"tool":"grouped_analysis","args":{"group_by":"invented"}}],{"schema":SCHEMA})


def test_generated_persistent_change_cannot_bypass_review():
    plan=normalize_plan([{"tool":"apply_cleaning","args":{},"approval":False}],{"schema":SCHEMA})
    assert [step["tool"] for step in plan]==["profile_dataset","recommend_cleaning","preview_cleaning","apply_cleaning"]
    assert plan[-1]["approval"] is True
    assert all(not step.get("approval") for step in plan[:-1])


def test_unknown_tool_cannot_be_executed():
    with pytest.raises(ValueError,match="unavailable"):
        normalize_plan([{"tool":"run_python","args":{"code":"print(1)"}}],{"schema":SCHEMA})


def test_uploaded_csv_question_returns_calculated_group_answer():
    client=TestClient(app)
    name=f"Sales {uuid4().hex[:8]}"
    csv="product_type,sale_price,store_region\n"+"Widget,15,North\n"*30+"Other,5,South\n"*30
    uploaded=client.post(f"/api/datasets/upload?name={name.replace(' ','%20')}",content=csv.encode(),headers={"content-type":"text/csv"})
    assert uploaded.status_code==200
    response=client.post("/api/analyst/query",json={"dataset":uploaded.json()["id"],"query":"What is the average sale price by product type?"})
    assert response.status_code==200
    result=response.json()
    assert result["status"]=="completed"
    assert result["steps"][0]["tool"]=="grouped_analysis"
    assert "Widget" in result["answer"] and "15.0" in result["answer"]


def test_local_model_provider_uses_structured_tool_plan_without_network(monkeypatch):
    monkeypatch.setattr(llm,"get_settings",lambda:SimpleNamespace(ollama_url="http://127.0.0.1:11434",llm_model="installed-model"))
    provider=llm.OllamaLLM()
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"message":{"content":'{"steps":[{"tool":"get_schema","label":"Inspect fields","args":{}}]}'} }
    class FakeClient:
        def post(self,url,json):
            assert url=="http://127.0.0.1:11434/api/chat" and json["format"]=="json"
            return FakeResponse()
    provider.client=FakeClient()
    assert provider.plan("What fields exist?",{"schema":SCHEMA})[0]["tool"]=="get_schema"


def test_local_model_rejects_remote_endpoint(monkeypatch):
    monkeypatch.setattr(llm,"get_settings",lambda:SimpleNamespace(ollama_url="https://example.com",llm_model="installed-model"))
    with pytest.raises(ValueError,match="local HTTP"):
        llm.OllamaLLM()
