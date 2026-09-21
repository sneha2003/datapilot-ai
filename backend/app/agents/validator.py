def validate_tool_result(tool,result):
    if result is None: return False,"Tool returned no result"
    if tool=="readonly_sql" and not result.get("columns"): return False,"SQL returned no columns"
    if tool in {"auto_visualization","create_visualization"} and not result.get("figure"): return False,"Visualization has no figure"
    if tool=="train_recommended" and not result.get("models"): return False,"No model completed training"
    if tool.startswith("apply_") and not result.get("dataset_id"): return False,"Transformation did not create a version"
    return True,None
