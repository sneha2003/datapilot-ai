"""Business questions must return computed evidence, not a generic dataset summary."""
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.agents.planner import create_plan
from app.database.base import engine
from app.database.models import Dataset
from app.mcp.gateway import MCPToolGateway
from app.tools.database import resolve_dataset


def _sales_dataset(db):
    rows=[]
    for year in [2024,2025]:
        for month in range(1,13):
            for group,weight in [("Core",1.0),("Growth",.7)]:
                revenue=(1000+month*20)*weight*(1.15 if year==2025 else 1)
                margin=.25 if year==2024 else .16
                rows.append({"order_date":f"{year}-{month:02d}-15","order_id":f"{year}-{month}-{group}","customer_id":group,"category":group,"region":"West" if group=="Core" else "East","revenue":revenue,"profit":revenue*margin,"discount":0.1 if group=="Growth" else 0,"quantity":10})
    frame=pd.DataFrame(rows)
    frame.to_sql("test_sales_raw",engine,if_exists="replace",index=False)
    ds=Dataset(name="test_sales",display_name="Test Sales",table_name="test_sales_raw",row_count=len(frame),column_count=len(frame.columns),task_candidates=[],target_suggestions=[],version="raw")
    db.add(ds); db.commit(); return ds


def test_housing_patterns_are_measured_and_answered(db):
    query="give me some important patterns that i need to notice"
    assert [step["tool"] for step in create_plan(query,{"target":"median_house_value","dataset_name":"california_housing"})]==["business_insights"]
    dataset=resolve_dataset(db,"california_housing")
    result=MCPToolGateway(db,dataset).call("business_insights",{"intent":"patterns"},{"query":query})
    assert result["status"]=="supported"
    assert "relationship" in result["headline"]
    assert result["findings"] and "correlation" in result["findings"][0]["detail"]
    response=TestClient(app).post("/api/analyst/query",json={"query":query,"dataset":dataset.id})
    assert response.status_code==200
    answer=response.json()["answer_structure"]
    assert "relationship" in answer["headline"]
    assert "What the data shows" in [section["title"] for section in answer["sections"]]


def test_sales_business_investigation_uses_actual_revenue_profit_and_trends(db):
    dataset=_sales_dataset(db)
    query="Revenue is growing but profit isn't improving. Find the patterns and recommend actions."
    plan=create_plan(query,{"dataset_name":"test_sales"})
    assert plan[0]["tool"]=="business_insights"
    result=MCPToolGateway(db,dataset).call("business_insights",plan[0]["args"],{"query":query})
    assert result["status"]=="supported"
    assert any(metric["label"]=="Profit margin" for metric in result["metrics"])
    assert any(finding["title"]=="Profit trend" for finding in result["findings"])
    assert result["recommendations"]
    assert result["visualization"]["figure"]["data"]
    response=TestClient(app).post("/api/analyst/query",json={"query":query,"dataset":dataset.id})
    assert response.status_code==200
    assert "Revenue changed" in response.json()["answer_structure"]["headline"]
    assert any(section["title"]=="What to do next" for section in response.json()["answer_structure"]["sections"])


def test_sales_questions_route_to_relevant_calculation(db):
    dataset=_sales_dataset(db)
    samples={
        "What are the most important KPIs?":"kpis",
        "How are sales performing over time?":"trend",
        "Which products are performing best and worst?":"ranking",
        "Revenue dropped last month. Why?":"root_cause",
        "Are discounts helping the business?":"discounts",
        "Which customers are most valuable?":"customers",
        "Are there unusual values?":"anomalies",
        "Forecast revenue next month":"forecast",
    }
    gateway=MCPToolGateway(db,dataset)
    for query,intent in samples.items():
        plan=create_plan(query,{"dataset_name":"test_sales"})
        assert plan[0]["args"]["intent"]==intent,query
        result=gateway.call("business_insights",plan[0]["args"],{"query":query})
        assert result["status"]=="supported",(query,result["limitations"])
        assert result["headline"]


def test_business_question_without_required_fields_is_honest(db):
    dataset=resolve_dataset(db,"california_housing")
    result=MCPToolGateway(db,dataset).call("business_insights",{"intent":"discounts"},{"query":"Are discounts helping us?"})
    assert result["status"]=="unsupported"
    assert "required" in result["limitations"][0]
