from typing import Any, TypedDict

class AnalystState(TypedDict,total=False):
    conversation_id:str; run_id:str; user_query:str; active_dataset:str|None; dataset_schema:dict
    task_type:str; target:str|None; analysis_plan:list[dict]; current_step:int; completed_steps:list[dict]
    analysis_results:list[dict]; generated_artifacts:list[dict]; model_experiments:list[dict]; memory_context:list[dict]
    pending_approval:dict|None; approval_decision:dict|None; errors:list[dict]; trace:list[dict]; final_answer:str; answer_structure:dict; status:str; retry_count:int
