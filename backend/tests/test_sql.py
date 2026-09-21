import pytest
from app.tools.database import SQLValidationError,execute_readonly_sql,resolve_dataset,validate_sql,get_dataset_schema

def test_readonly_sql_executes_and_limits(db):
    result=execute_readonly_sql(db,'SELECT pclass, AVG(survived) AS survival_rate FROM titanic_raw GROUP BY pclass')
    assert result["row_count"]==3 and "LIMIT" in result["query"]
def test_sql_rejects_mutations_and_multiple_statements(db):
    for query in ["DROP TABLE titanic_raw","SELECT 1; DELETE FROM titanic_raw","SELECT pg_sleep(3)"]:
        with pytest.raises(SQLValidationError): validate_sql(db,query)
def test_sql_rejects_unknown_table(db):
    with pytest.raises(SQLValidationError,match="Unknown table"): validate_sql(db,"SELECT * FROM imaginary")
    with pytest.raises(SQLValidationError,match="Unknown column"): validate_sql(db,"SELECT imaginary_column FROM titanic_raw")
def test_registry_and_schema(db):
    ds=resolve_dataset(db,"Titanic"); schema=get_dataset_schema(db,ds)
    assert ds.name=="titanic" and "survived" in [x["name"] for x in schema["columns"]]
