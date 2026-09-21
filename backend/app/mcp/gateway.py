"""In-process MCP-compatible tool gateway used by the graph and MCP server."""
from app.memory.store import retrieve_memory
from app.tools import business, cleaning, features, profiling, statistics, training, visualization
from app.tools.database import execute_readonly_sql, get_dataset_lineage, get_dataset_schema, list_datasets
from app.tools.model_recommendation import identify_ml_problem, recommend_models

class MCPToolGateway:
    def __init__(self,db,dataset=None): self.db=db; self.dataset=dataset
    def call(self,name,args=None,context=None):
        args=args or {}; ctx=context or {}; ds=self.dataset
        if name=="list_datasets": return list_datasets(self.db)
        if name=="retrieve_memory": return retrieve_memory(self.db,args.get("query"))
        if not ds: raise ValueError("A dataset is required for this tool")
        if name=="clarify_question":
            schema=get_dataset_schema(self.db,ds)
            return {"dataset":ds.display_name,"columns":[column["name"] for column in schema["columns"]],"target":ds.target_suggestions[0] if ds.target_suggestions else None}
        if name=="get_schema": return get_dataset_schema(self.db,ds)
        if name=="get_dataset_lineage": return get_dataset_lineage(self.db,ds)
        if name=="dataset_overview": return profiling.dataset_overview(self.db,ds)
        if name=="business_insights":
            result=business.analyze_business_question(self.db,ds,ctx.get("query",""),args.get("intent","investigation"))
            if result.get("chart"):
                chart=result["chart"]
                result["visualization"]=visualization.create_result_visualization(chart["rows"],{key:chart[key] for key in ("title","chart_type","x","y")})
                result.pop("chart")
            return result
        if name=="profile_dataset": return profiling.profile_dataset(self.db,ds)
        if name=="correlations": return profiling.calculate_correlations(self.db,ds,args.get("method","pearson"))
        if name=="readonly_sql": return execute_readonly_sql(self.db,args["query"])
        if name=="recommend_cleaning": return cleaning.recommend_cleaning_steps(self.db,ds)
        if name=="preview_cleaning": return cleaning.preview_cleaning(self.db,ds,args.get("operations",ctx.get("cleaning_operations",[])))
        if name=="apply_cleaning": return cleaning.apply_cleaning(self.db,ds,args.get("operations",ctx.get("cleaning_operations",[])),ctx.get("session_id"))
        if name=="recommend_features": return features.recommend_features(self.db,ds,args.get("target",ctx.get("target")))
        if name=="preview_features": return features.preview_feature_engineering(self.db,ds,args.get("transformations",ctx.get("feature_operations",[])))
        if name=="apply_features": return features.apply_feature_engineering(self.db,ds,args.get("transformations",ctx.get("feature_operations",[])),ctx.get("session_id"))
        if name=="identify_problem": return identify_ml_problem(self.db,ds,args.get("target") or ctx.get("target"),args.get("objective") or ctx.get("objective"))
        if name=="recommend_models": return recommend_models(self.db,ds,ctx["target"],ctx["task_type"],ctx.get("interpretability",False))
        if name=="train_recommended":
            results=[]
            for candidate in ctx.get("candidates",[])[:4]: results.append(training.train_model(self.db,ds,ctx["target"],ctx["task_type"],candidate["algorithm"],ctx.get("session_id"),ctx.get("time_column")))
            return {"models":results,"comparison":training.compare_models(results)}
        if name=="get_experiment_importance": return training.feature_importance(self.db,args["experiment_id"])
        if name=="get_experiment_evaluation": return training.evaluate_model(self.db,args["experiment_id"])
        if name=="compare_saved_models":
            ids=args.get("experiment_ids") or [item["experiment_id"] for item in ctx.get("experiments",[]) if item.get("experiment_id")]
            return training.compare_saved_models(self.db,ids)
        if name=="grouped_analysis": return statistics.grouped_analysis(self.db,ds,**args)
        if name=="target_drivers": return statistics.target_driver_analysis(self.db,ds,args.get("target") or ctx.get("target"),args.get("limit",8))
        if name=="trend_analysis": return statistics.trend_analysis(self.db,ds,**args)
        if name=="statistical_test": return statistics.statistical_test(self.db,ds,**args)
        if name=="create_visualization": return visualization.create_visualization(self.db,ds,args)
        if name=="visualize_last_result":
            table=ctx.get("last_table") or {}; rows=table.get("rows",[]); columns=table.get("columns",[])
            if len(columns)<2: raise ValueError("No grouped result is available to visualize")
            return visualization.create_result_visualization(rows,{"chart_type":"bar","x":columns[0],"y":columns[1],"title":args.get("title",f"Analysis result — {ds.display_name}")})
        if name=="auto_visualization": return visualization.create_visualization(self.db,ds,_auto_spec(ds,ctx.get("query",""),ctx.get("schema",{})))
        raise ValueError(f"Unknown MCP tool: {name}")

def _auto_spec(dataset,query,schema):
    q=query.lower(); columns=[c["name"] for c in schema.get("columns",[])]; lower={c.lower():c for c in columns}
    if "age" in q and "age" in lower:
        return {"chart_type":"histogram","x":lower["age"],"title":f"Age distribution — {dataset.display_name}"}
    if "correlation" in q or "associated" in q: return {"chart_type":"heatmap","title":f"Correlation heatmap — {dataset.display_name}"}
    numeric=[c for c in columns if any(x in next((z["type"].lower() for z in schema["columns"] if z["name"]==c),"") for x in ["int","float","numeric","double"])]
    categorical=[c for c in columns if c not in numeric]
    if categorical: return {"chart_type":"bar","x":categorical[0],"aggregation":"count","title":f"Records by {categorical[0]}"}
    if numeric: return {"chart_type":"histogram","x":numeric[0],"title":f"Distribution of {numeric[0]}"}
    raise ValueError("No plottable columns found")
