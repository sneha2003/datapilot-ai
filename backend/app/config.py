from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")
    app_name: str = "DataPilot AI"
    database_url: str = "sqlite:///./datapilot.db"
    llm_provider: str = "mock"
    llm_model: str = ""
    groq_api_key: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    max_agent_steps: int = 12
    sql_timeout_ms: int = 10_000
    sql_max_rows: int = 1_000
    cors_origins: str = "http://localhost:5173"
    artifact_root: Path = Path("../artifacts")
    @field_validator("database_url")
    @classmethod
    def use_psycopg_driver(cls,value:str)->str:
        if value.startswith("postgres://"):
            return "postgresql+psycopg://"+value[len("postgres://"):]
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://"+value[len("postgresql://"):]
        return value
    @property
    def cors_origin_list(self): return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

@lru_cache
def get_settings(): return Settings()
