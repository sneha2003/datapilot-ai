import sys
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.database.base import engine
from app.database.models import AnalysisRun, AnalysisSession, Dataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from migrate_recovery import migrate


def test_recovery_migration_preserves_datasets_history_and_is_repeatable(db,tmp_path):
    source=db.query(Dataset).filter_by(name="titanic").one()
    session=AnalysisSession(title="A real analysis",active_dataset_id=source.id,state={"target":"survived"})
    db.add(session); db.flush()
    db.add(AnalysisRun(session_id=session.id,query="How many passengers survived?",status="completed",findings=[{"summary":"Measured from records"}]))
    db.commit()
    target_url=f"sqlite:///{(tmp_path/'new_database.db').as_posix()}"
    first=migrate(str(engine.url),target_url)
    second=migrate(str(engine.url),target_url)
    assert first["copied_data_tables"]==2
    assert second["copied_data_tables"]==0
    with Session(create_engine(target_url)) as target:
        assert target.scalar(select(Dataset).where(Dataset.id==source.id)).row_count==12
        assert target.scalar(select(AnalysisSession).where(AnalysisSession.id==session.id)).active_dataset_id==source.id
        assert target.scalar(select(AnalysisRun).where(AnalysisRun.session_id==session.id)).query=="How many passengers survived?"


def test_recovery_migration_merges_equivalent_dataset_ids(db,tmp_path):
    source=db.query(Dataset).filter_by(name="titanic").one()
    session=AnalysisSession(title="Recovered analysis",active_dataset_id=source.id,state={"dataset_id":source.id})
    db.add(session); db.commit()
    target_url=f"sqlite:///{(tmp_path/'existing_database.db').as_posix()}"
    migrate(str(engine.url),target_url)
    with Session(create_engine(target_url)) as target:
        other=target.get(Dataset,source.id)
        other.id="different-dataset-id"
        target.commit()
        old=target.get(AnalysisSession,session.id)
        target.delete(old)
        target.commit()
    outcome=migrate(str(engine.url),target_url,merge_equivalent=True)
    assert outcome["copied_data_tables"]==0
    with Session(create_engine(target_url)) as target:
        assert target.scalar(select(Dataset).where(Dataset.name=="titanic")).id=="different-dataset-id"
        recovered=target.get(AnalysisSession,session.id)
        assert recovered.active_dataset_id=="different-dataset-id"
        assert recovered.state["dataset_id"]=="different-dataset-id"
