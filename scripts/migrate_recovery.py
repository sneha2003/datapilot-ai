"""Copy the local recovery database into PostgreSQL without overwriting data.

Run after `docker compose up -d postgres`. A conflict aborts before any writes.
The migration keeps dataset IDs, version lineage, sessions, runs and experiments.
"""
import argparse
from collections import Counter
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env",override=False)
sys.path.insert(0,str(ROOT/"backend"))
from app.database.base import Base
from app.database.models import (Dataset, AnalysisSession, AnalysisRun, ToolExecution,
                                 ModelExperiment, GeneratedArtifact, MemoryEntry)

MODELS=(Dataset,AnalysisSession,AnalysisRun,ToolExecution,ModelExperiment,GeneratedArtifact,MemoryEntry)


def _records(engine, table_name):
    quoted=engine.dialect.identifier_preparer.quote(table_name)
    with engine.connect() as conn:
        return pd.read_sql_query(text(f"SELECT * FROM {quoted}"),conn)


def _same_records(source_frame, target_frame):
    """Compare unordered SQL rows, allowing SQLite booleans to become PG booleans."""
    if list(source_frame.columns)!=list(target_frame.columns) or len(source_frame)!=len(target_frame):
        return False
    for column in source_frame.columns:
        if pd.api.types.is_bool_dtype(target_frame[column]):
            source_frame[column]=source_frame[column].astype(bool)
    def rows(frame):
        normalized=frame.astype(object).where(pd.notna(frame),None)
        return Counter(map(tuple,normalized.astype(str).to_numpy()))
    return rows(source_frame)==rows(target_frame)


def _remap(value, dataset_ids):
    if isinstance(value,str): return dataset_ids.get(value,value)
    if isinstance(value,list): return [_remap(item,dataset_ids) for item in value]
    if isinstance(value,dict): return {key:_remap(item,dataset_ids) for key,item in value.items()}
    return value


def migrate(source_url,target_url,merge_equivalent=False):
    if source_url==target_url: raise ValueError("Source and destination must be different databases")
    source=create_engine(source_url)
    target=create_engine(target_url)
    try:
        with Session(source) as source_db, Session(target) as target_db:
            source_datasets=source_db.scalars(select(Dataset)).all()
            if not source_datasets: raise ValueError("The recovery database contains no registered datasets")
            by_id={item.id:item for item in source_datasets}
            def depth(item):
                count=0; seen={item.id}
                while item.parent_id in by_id:
                    if item.parent_id in seen: raise ValueError("Dataset lineage contains a cycle")
                    seen.add(item.parent_id); item=by_id[item.parent_id]; count+=1
                return count
            source_datasets.sort(key=depth)
            target_tables=set(inspect(target).get_table_names())
            dataset_ids={}
            # Check every conflict before writing any table or metadata row.
            if "datasets" in target_tables:
                for item in source_datasets:
                    conflict=target_db.scalar(select(Dataset).where((Dataset.name==item.name)|(Dataset.table_name==item.table_name)))
                    if conflict and conflict.id!=item.id:
                        if not merge_equivalent or conflict.name!=item.name or conflict.table_name!=item.table_name or conflict.version!=item.version or conflict.row_count!=item.row_count or conflict.column_count!=item.column_count:
                            raise ValueError(f"PostgreSQL already has a different dataset named '{item.name}'. Nothing was replaced; reconcile it before migrating.")
                        if not _same_records(_records(source,item.table_name),_records(target,item.table_name)):
                            raise ValueError(f"Dataset '{item.name}' differs between databases. Nothing was replaced.")
                        dataset_ids[item.id]=conflict.id
            for item in source_datasets:
                if item.table_name in target_tables:
                    existing=target_db.get(Dataset,dataset_ids.get(item.id,item.id)) if "datasets" in target_tables else None
                    if not existing or existing.table_name!=item.table_name:
                        raise ValueError(f"Table {item.table_name} exists without the matching registry entry. Nothing was replaced.")
                    with target.connect() as conn:
                        table=target.dialect.identifier_preparer.quote(item.table_name)
                        rows=conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
                    if rows!=item.row_count:
                        raise ValueError(f"Table {item.table_name} already exists with {rows} rows, expected {item.row_count}. Nothing was replaced.")
            Base.metadata.create_all(target)
            copied_tables=0
            for item in source_datasets:
                if item.table_name in target_tables: continue
                with source.connect() as conn:
                    table=source.dialect.identifier_preparer.quote(item.table_name)
                    frame=pd.read_sql_query(text(f"SELECT * FROM {table}"),conn)
                if len(frame)!=item.row_count: raise ValueError(f"Source table {item.table_name} does not match its registry row count")
                frame.to_sql(item.table_name,target,if_exists="fail",index=False,chunksize=max(1,20000//len(frame.columns)))
                copied_tables+=1
            counts={}
            for model in MODELS:
                copied=0
                records=source_datasets if model is Dataset else source_db.scalars(select(model)).all()
                for item in records:
                    if model is Dataset and item.id in dataset_ids: continue
                    if target_db.get(model,item.id): continue
                    values={column.name:_remap(getattr(item,column.name),dataset_ids) for column in model.__table__.columns}
                    target_db.add(model(**values)); copied+=1
                target_db.commit()
                counts[model.__tablename__]=copied
            return {"copied_data_tables":copied_tables,"copied_records":counts}
    finally:
        source.dispose(); target.dispose()


def main():
    parser=argparse.ArgumentParser(description="Preserve local recovery data while moving to PostgreSQL")
    parser.add_argument("--source",default=f"sqlite:///{(ROOT/'backend/datapilot_fallback.db').as_posix()}")
    parser.add_argument("--target",default=os.getenv("DATABASE_URL"))
    parser.add_argument("--merge-equivalent",action="store_true",help="Merge histories when both databases contain identical datasets with different IDs")
    args=parser.parse_args()
    if not args.target or not args.target.startswith("postgresql+"):
        parser.error("Set DATABASE_URL to a PostgreSQL URL or provide --target")
    outcome=migrate(args.source,args.target,merge_equivalent=args.merge_equivalent)
    print(f"Copied {outcome['copied_data_tables']} dataset tables. Metadata records copied: {outcome['copied_records']}")


if __name__=="__main__": main()
