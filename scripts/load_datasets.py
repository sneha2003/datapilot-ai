"""Load downloaded CSVs into PostgreSQL and populate the registry."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"backend"))
from app.database.base import Base
from app.config import get_settings
from app.database.models import Dataset
from sqlalchemy.orm import Session

def main(only_missing=False):
    engine=create_engine(get_settings().database_url)
    Base.metadata.create_all(engine)
    registry=json.loads((ROOT/"data/metadata/datasets.json").read_text())
    with Session(engine) as db:
        for item in registry:
            found=db.query(Dataset).filter_by(name=item["name"],version="raw").first()
            if only_missing and found:
                print(f"Keeping existing {item['name']}")
                continue
            path=ROOT/"data/raw"/f"{item['name']}.csv"
            if not path.exists(): print(f"Skipping {item['name']}; run download_datasets.py"); continue
            df=pd.read_csv(path)
            max_parameters=900 if engine.dialect.name=="sqlite" else 30_000
            chunk_size=max(1,min(2_000,max_parameters//max(1,len(df.columns))))
            # Give each raw table load its own committed transaction before the
            # registry session reads it. This avoids an idle transaction and
            # lock wait when reloading an existing PostgreSQL database.
            with engine.begin() as connection:
                df.to_sql(item["table"],connection,if_exists="replace",index=False,chunksize=chunk_size,method="multi")
            values=dict(name=item["name"],display_name=item["display_name"],table_name=item["table"],description=item["description"],source=item["source"],license=item["license"],row_count=len(df),column_count=len(df.columns),task_candidates=item["task_candidates"],target_suggestions=[item["default_target"]],column_descriptions=item.get("column_descriptions",{}),version="raw")
            if found:
                for k,v in values.items(): setattr(found,k,v)
            else: db.add(Dataset(**values))
            # Release registry writes before pandas opens its next loader
            # connection (required by SQLite; harmless for PostgreSQL).
            db.commit()
    print("Dataset registry loaded")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--only-missing",action="store_true",help="Keep datasets already in the database")
    main(only_missing=parser.parse_args().only_missing)
