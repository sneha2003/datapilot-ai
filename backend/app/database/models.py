import uuid
from datetime import datetime, timezone
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database.base import Base

def uid(): return str(uuid.uuid4())
def now(): return datetime.now(timezone.utc)
class Dataset(Base):
    __tablename__="datasets"
    id: Mapped[str]=mapped_column(String(36),primary_key=True,default=uid); name: Mapped[str]=mapped_column(String(100),unique=True,index=True)
    display_name: Mapped[str]=mapped_column(String(160)); table_name: Mapped[str]=mapped_column(String(100),unique=True)
    description: Mapped[str]=mapped_column(Text,default=""); source: Mapped[str]=mapped_column(Text,default=""); license: Mapped[str]=mapped_column(String(200),default="")
    row_count: Mapped[int]=mapped_column(Integer,default=0); column_count: Mapped[int]=mapped_column(Integer,default=0)
    task_candidates: Mapped[list]=mapped_column(JSON,default=list); target_suggestions: Mapped[list]=mapped_column(JSON,default=list); column_descriptions: Mapped[dict]=mapped_column(JSON,default=dict)
    imported_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now); parent_id: Mapped[str|None]=mapped_column(ForeignKey("datasets.id"),nullable=True)
    version: Mapped[str]=mapped_column(String(40),default="raw"); transformations: Mapped[list]=mapped_column(JSON,default=list)
class AnalysisSession(Base):
    __tablename__="analysis_sessions"; id: Mapped[str]=mapped_column(String(36),primary_key=True,default=uid); title: Mapped[str]=mapped_column(String(200),default="New analysis")
    active_dataset_id: Mapped[str|None]=mapped_column(ForeignKey("datasets.id"),nullable=True); state: Mapped[dict]=mapped_column(JSON,default=dict)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now); updated_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now,onupdate=now)
class AnalysisRun(Base):
    __tablename__="analysis_runs"; id: Mapped[str]=mapped_column(String(36),primary_key=True,default=uid); session_id: Mapped[str]=mapped_column(ForeignKey("analysis_sessions.id"),index=True)
    query: Mapped[str]=mapped_column(Text); plan: Mapped[list]=mapped_column(JSON,default=list); findings: Mapped[list]=mapped_column(JSON,default=list); status: Mapped[str]=mapped_column(String(30),default="running")
    trace: Mapped[list]=mapped_column(JSON,default=list); error: Mapped[str|None]=mapped_column(Text,nullable=True); created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now); completed_at: Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
class ToolExecution(Base):
    __tablename__="tool_executions"; id: Mapped[str]=mapped_column(String(36),primary_key=True,default=uid); run_id: Mapped[str]=mapped_column(ForeignKey("analysis_runs.id"),index=True)
    tool_name: Mapped[str]=mapped_column(String(100)); arguments: Mapped[dict]=mapped_column(JSON,default=dict); result_summary: Mapped[dict]=mapped_column(JSON,default=dict); duration_ms: Mapped[float]=mapped_column(Float,default=0); success: Mapped[bool]=mapped_column(Boolean,default=True); retry: Mapped[int]=mapped_column(Integer,default=0); created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
class ModelExperiment(Base):
    __tablename__="model_experiments"; id: Mapped[str]=mapped_column(String(36),primary_key=True,default=uid); session_id: Mapped[str|None]=mapped_column(ForeignKey("analysis_sessions.id"),nullable=True); dataset_id: Mapped[str]=mapped_column(ForeignKey("datasets.id")); target: Mapped[str]=mapped_column(String(120)); task_type: Mapped[str]=mapped_column(String(60)); algorithm: Mapped[str]=mapped_column(String(100)); parameters: Mapped[dict]=mapped_column(JSON,default=dict); metrics: Mapped[dict]=mapped_column(JSON,default=dict); cv_results: Mapped[dict]=mapped_column(JSON,default=dict); artifact_path: Mapped[str]=mapped_column(Text); duration_ms: Mapped[float]=mapped_column(Float,default=0); created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
class GeneratedArtifact(Base):
    __tablename__="generated_artifacts"; id: Mapped[str]=mapped_column(String(36),primary_key=True,default=uid); session_id: Mapped[str|None]=mapped_column(ForeignKey("analysis_sessions.id"),nullable=True); dataset_id: Mapped[str|None]=mapped_column(ForeignKey("datasets.id"),nullable=True); kind: Mapped[str]=mapped_column(String(50)); title: Mapped[str]=mapped_column(String(200)); specification: Mapped[dict]=mapped_column(JSON,default=dict); path: Mapped[str|None]=mapped_column(Text,nullable=True); created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
class MemoryEntry(Base):
    __tablename__="memory_entries"; id: Mapped[str]=mapped_column(String(36),primary_key=True,default=uid); memory_type: Mapped[str]=mapped_column(String(60),index=True); key: Mapped[str]=mapped_column(String(160),index=True); value: Mapped[dict]=mapped_column(JSON); confidence: Mapped[float]=mapped_column(Float,default=1.0); session_id: Mapped[str|None]=mapped_column(ForeignKey("analysis_sessions.id"),nullable=True); created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now); updated_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now,onupdate=now)
