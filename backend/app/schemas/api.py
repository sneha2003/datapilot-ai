from typing import Any,Literal
from pydantic import BaseModel,Field
class AnalystQuery(BaseModel):
    query:str=Field(min_length=2,max_length=4000); session_id:str|None=None; dataset:str|None=None; target:str|None=None
class AnalystContinue(BaseModel):
    session_id:str; run_id:str; decision:Literal["approve","reject","modify"]; operations:list[dict[str,Any]]|None=None; reason:str|None=None
