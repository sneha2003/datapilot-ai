import os,sys
from pathlib import Path
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
# Tests must never inherit the live PostgreSQL connection from the project .env.
TEST_DATABASE=(ROOT/"test_datapilot.db").resolve()
os.environ["DATABASE_URL"]=f"sqlite:///{TEST_DATABASE.as_posix()}"
from sqlalchemy import inspect,text
from app.database.base import Base,SessionLocal,engine
from app.database.models import Dataset

@pytest.fixture(autouse=True)
def seeded_db():
    Base.metadata.drop_all(engine); Base.metadata.create_all(engine)
    with engine.begin() as connection:
        for table in inspect(engine).get_table_names():
            if "_clean_v" in table or "_features_v" in table:
                connection.execute(text(f'DROP TABLE "{table}"'))
    titanic=pd.DataFrame({"survived":[0,1,1,0,1,0,1,0,1,1,0,0],"pclass":[3,1,2,3,1,3,2,3,1,2,3,2],"sex":["male","female"]*6,"age":[22,38,None,35,28,18,32,40,26,30,50,21],"sibsp":[1,1,0,0,1,0,0,2,0,1,0,0],"parch":[0,0,0,0,1,0,0,0,0,1,0,0],"fare":[7.2,71.3,8.0,8.1,53.1,7.9,13.0,9.2,30.0,16.0,7.5,10.5]})
    housing=pd.DataFrame({"median_income":[1,2,3,4,5,6,7,8,9,10,4,6],"housing_age":[10,12,20,22,30,35,40,42,50,52,15,25],"latitude":[34,35,34,36,37,38,37,36,35,34,39,33],"median_house_value":[90,110,145,170,210,240,290,330,370,410,180,260]})
    titanic.to_sql("titanic_raw",engine,if_exists="replace",index=False); housing.to_sql("california_housing_raw",engine,if_exists="replace",index=False)
    with SessionLocal() as db:
        db.add_all([Dataset(name="titanic",display_name="Titanic",table_name="titanic_raw",row_count=len(titanic),column_count=len(titanic.columns),task_candidates=["binary_classification"],target_suggestions=["survived"],version="raw"),Dataset(name="california_housing",display_name="California Housing",table_name="california_housing_raw",row_count=len(housing),column_count=len(housing.columns),task_candidates=["regression"],target_suggestions=["median_house_value"],version="raw")]); db.commit()
    yield

@pytest.fixture
def db():
    with SessionLocal() as session: yield session
