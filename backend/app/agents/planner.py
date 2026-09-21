"""Validate LLM plans before they reach the tool gateway or approval node."""
from app.config import get_settings
from app.services.llm import get_llm

ALLOWED={
    "list_datasets","retrieve_memory","get_schema","get_dataset_lineage","dataset_overview",
    "business_insights","profile_dataset","correlations","readonly_sql","recommend_cleaning",
    "preview_cleaning","apply_cleaning","recommend_features","preview_features","apply_features",
    "identify_problem","recommend_models","train_recommended","get_experiment_importance",
    "get_experiment_evaluation","compare_saved_models","grouped_analysis","target_drivers",
    "trend_analysis","statistical_test","create_visualization","visualize_last_result",
    "auto_visualization","clarify_question",
}
PREREQUISITES={
    "apply_cleaning":["profile_dataset","recommend_cleaning","preview_cleaning"],
    "apply_features":["recommend_features","preview_features"],
}


def normalize_plan(steps,context):
    if not isinstance(steps,list): raise ValueError("The planner did not return a list of steps")
    normalized=[]
    columns={column["name"] for column in (context.get("schema") or {}).get("columns",[])}
    for step in steps:
        if not isinstance(step,dict) or step.get("tool") not in ALLOWED:
            raise ValueError("The planner selected an unavailable analysis tool")
        name=step["tool"]
        args=step.get("args") or {}
        if not isinstance(args,dict): raise ValueError(f"Invalid arguments for {name}")
        for key in ("group_by","metric","date_column","target"):
            if key in args and args[key] is not None and columns and args[key] not in columns:
                raise ValueError(f"The requested {key} is not present in the selected dataset")
        if name in PREREQUISITES:
            selected={item["tool"] for item in normalized}
            for prerequisite in PREREQUISITES[name]:
                if prerequisite not in selected:
                    normalized.append({"tool":prerequisite,"label":prerequisite.replace("_"," ").capitalize()})
        normalized.append({"tool":name,"label":str(step.get("label") or name.replace("_"," ").capitalize())[:140],"args":args,**({"approval":True} if name in PREREQUISITES else {})})
        if len(normalized)>get_settings().max_agent_steps:
            raise ValueError("The proposed analysis exceeds the safe step limit")
    return normalized or [{"tool":"clarify_question","label":"Clarify the analytical question","args":{}}]


def create_plan(query,context):
    return normalize_plan(get_llm().plan(query,context),context)
