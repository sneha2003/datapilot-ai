from fastapi.testclient import TestClient
from sqlalchemy import text
from app.database.base import engine
from app.main import _json_safe,app

def test_api_json_safety_replaces_non_finite_numbers():
    safe=_json_safe({"values":[1.5,float("nan"),float("inf"),float("-inf")]})
    assert safe=={"values":[1.5,None,None,None]}

def test_csv_upload_is_registered_and_stored_in_database():
    client=TestClient(app)
    response=client.post("/api/datasets/upload?name=Upload verification",content=b"customer,revenue\nA,10.5\nB,20.0\n",headers={"Content-Type":"text/csv"})
    assert response.status_code==200
    dataset=response.json()
    assert dataset["rows"]==2 and dataset["columns"]==2 and dataset["source"]=="User upload"
    detail=client.get(f"/api/datasets/{dataset['id']}")
    assert detail.status_code==200 and detail.json()["sample"][0]["customer"]=="A"
    with engine.begin() as connection:
        connection.execute(text(f'DELETE FROM datasets WHERE id = :id'),{"id":dataset["id"]})
        connection.execute(text(f'DROP TABLE "{dataset["table"]}"'))
