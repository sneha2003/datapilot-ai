"""Turn measured tool results into concise, reviewable business answers."""
from math import isfinite


def _number(value, decimals=1):
    if not isinstance(value, (int, float)) or not isfinite(value):
        return "—"
    return f"{value:,.{decimals}f}" if isinstance(value, float) else f"{value:,}"


def _name(value):
    return str(value).replace("_", " ")


def _section(title, items):
    return {"title": title, "items": [str(item) for item in items if item][:6]}


def _build(headline, metrics=None, sections=None, note=None):
    return {"headline": headline, "metrics": metrics or [], "sections": sections or [], "note": note}


def _group_summary(result):
    columns = result.get("columns", [])
    if len(columns) < 2:
        return None
    group, metric = columns[:2]
    rows = [row for row in result.get("rows", []) if isinstance(row.get(metric), (int, float)) and isfinite(row[metric])]
    if rows and all(isinstance(row.get("sample_size"), (int, float)) for row in rows):
        minimum=max(10, int(sum(row["sample_size"] for row in rows)*0.005))
        rows=[row for row in rows if row["sample_size"]>=minimum]
    if not rows:
        return None
    high = max(rows, key=lambda row: row[metric])
    low = min(rows, key=lambda row: row[metric])
    return group, metric, high, low


def present(state):
    """Prefer the actual requested outcome over intermediate technical steps."""
    results = state.get("analysis_results", [])
    by_tool = {item["tool"]: item["result"] for item in results}
    errors = state.get("errors", [])
    if not results:
        return _build("I could not verify an answer from the data.", sections=[_section("What happened", [error.get("message") for error in errors])], note="Try a more specific question or choose a dataset.")

    if "apply_cleaning" in by_tool:
        applied = by_tool["apply_cleaning"]
        featured = by_tool.get("apply_features")
        preview = by_tool.get("preview_cleaning", {})
        before, after = preview.get("before", {}), preview.get("after", {})
        operations = by_tool.get("recommend_cleaning", {}).get("operations", [])
        changes = []
        for operation in operations:
            kind = operation.get("operation", "")
            column = _name(operation.get("column", ""))
            changes.append({"impute_numeric": f"Filled missing {column} with the median", "impute_categorical": f"Filled missing {column} with the most common value", "drop_duplicates": "Removed exact duplicate rows", "drop_columns": f"Removed unusable columns: {', '.join(operation.get('columns', []))}"}.get(kind, f"{_name(kind).capitalize()} {column}".strip()))
        name = (featured or applied).get("display_name") or (featured or applied).get("name")
        sections = [_section("Changes made", changes or ["No cleaning changes were needed."])]
        if featured:
            sections.append(_section("New analysis fields", [_name(column) for column in featured.get("added_columns", [])] or ["No extra fields were needed."]))
        sections.append(_section("Where to find it", [f"Select “{name}” in Data sources on the left. The original dataset is still available."]))
        return _build(f"Your data is ready in a new version: {name}.", [{"label": "Rows", "value": _number(after.get("rows"))}, {"label": "Missing cells", "value": f"{_number(before.get('missing'))} → {_number(after.get('missing'))}"}], sections)

    if "apply_features" in by_tool:
        data = by_tool["apply_features"]
        name = data.get("display_name") or data.get("name")
        return _build(f"Your prepared dataset is ready: {name}.", [{"label": "New fields", "value": str(len(data.get("added_columns", [])))}], [_section("Fields added", [_name(column) for column in data.get("added_columns", [])] or ["No extra fields were needed."]), _section("Where to find it", [f"Select “{name}” in Data sources on the left. The original data remains unchanged."])])

    if "get_dataset_lineage" in by_tool:
        data = by_tool["get_dataset_lineage"]
        active = data["active"]
        items = [f"Choose “{active['display_name']}” in Data sources on the left."]
        if data.get("parent"):
            items.append(f"Created from “{data['parent']['display_name']}”; the original remains available.")
        return _build(f"Your current dataset is “{active['display_name']}”.", [{"label": "Version", "value": active["version"]}, {"label": "Rows", "value": _number(active["rows"])}], [_section("Where it is", items)])

    if "business_insights" in by_tool:
        data=by_tool["business_insights"]
        sections=[]
        if data.get("findings"):
            sections.append(_section("What the data shows",[f"{item['title']}: {item['detail']}" for item in data["findings"]]))
        if data.get("recommendations"):
            sections.append(_section("What to do next",data["recommendations"]))
        if data.get("visualization"):
            sections.append(_section("Chart",[f"Open “{data['visualization']['title']}” in the Charts tab."]))
        return _build(data["headline"],data.get("metrics",[]),sections," ".join(data.get("limitations",[])) or None)

    if "train_recommended" in by_tool:
        data = by_tool["train_recommended"]
        comparison = data["comparison"]
        best = next((model for model in data["models"] if model["experiment_id"] == comparison["best_experiment_id"]), None)
        score = best["metrics"].get(comparison["primary_metric"]) if best else None
        ranking = [f"{_name(model['algorithm']).title()}: {_name(comparison['primary_metric'])} {_number(model['metrics'].get(comparison['primary_metric']), 3)}" for model in data["models"]]
        sections = [_section("Models compared", ranking), _section("How to use this", [comparison.get("selection_note", "Compare the primary score alongside other business requirements.")])]
        if best and best.get("feature_importance"):
            sections.append(_section("Most influential fields", [_name(item["feature"]) for item in best["feature_importance"][:4]]))
        return _build(f"{_name(comparison['best_algorithm']).title()} performed best in this test.", [{"label": _name(comparison["primary_metric"]).title(), "value": _number(score, 3)}, {"label": "Models tested", "value": str(len(data["models"]))}], sections, "These are measured test results, not a guarantee of future performance.")

    if "target_drivers" in by_tool:
        data = by_tool["target_drivers"]
        segments = data.get("top_segments", [])[:4]
        items = [f"{_name(segment['feature'])}: {segment['segment']} — {segment['positive_rate_pct']}% ({segment['positive_customers']:,} of {segment['customers']:,})" for segment in segments]
        fields = [_name(field["feature"]) for field in data.get("strongest_fields", [])[:3]]
        return _build(f"These groups had the highest observed rate of “{data['positive_label']}”.", [{"label": "Overall rate", "value": f"{data['baseline_rate_pct']}%"}, {"label": "Records compared", "value": _number(data["total_customers"])}], [_section("Groups that stand out", items), _section("Fields with the biggest differences", fields)], data.get("interpretation"))

    groups = [_group_summary(item["result"]) for item in results if item["tool"] == "grouped_analysis"]
    groups = [group for group in groups if group]
    if groups:
        facts = []
        for group, metric, high, low in groups:
            label = {"season": "Season", "weather": "Weather", "hour": "Time of day"}.get(group, _name(group).capitalize())
            unit = "rentals per hour" if metric == "mean_count" else _name(metric)
            high_label=f"{int(high[group]):02d}:00" if group=="hour" else _name(high[group])
            low_label=f"{int(low[group]):02d}:00" if group=="hour" else _name(low[group])
            high_size=f" across {_number(high['sample_size'])} records" if high.get("sample_size") is not None else ""
            low_size=f" across {_number(low['sample_size'])} records" if low.get("sample_size") is not None else ""
            facts.append(f"{label}: highest for {high_label} ({_number(high[metric])} {unit}{high_size}); lowest for {low_label} ({_number(low[metric])} {unit}{low_size}).")
        main_group, _, high, _ = groups[0]
        main_label=f"{int(high[main_group]):02d}:00" if main_group=="hour" else _name(high[main_group])
        headline=f"{_name(main_group).capitalize()} stands out: {main_label} has the highest measured average."
        if [group for group,_,_,_ in groups]==["season","weather","hour"] and all(metric=="mean_count" for _,metric,_,_ in groups):
            headline=f"Separate comparisons found the highest average bike rentals in {_name(groups[0][2]['season'])}, under {_name(groups[1][2]['weather'])} weather, and at {int(groups[2][2]['hour']):02d}:00."
        return _build(headline, [{"label": "Comparisons", "value": str(len(groups))}], [_section("Measured differences", facts)], "Small groups were excluded from the comparison. These historical patterns do not establish cause and effect.")

    if "get_schema" in by_tool:
        data = by_tool["get_schema"]
        columns = [_name(column["name"]) for column in data["columns"]]
        return _build(f"{_name(data['dataset']).title()} has {len(columns)} fields.", [{"label": "Fields", "value": str(len(columns))}], [_section("Available information", [", ".join(columns[index:index + 7]) for index in range(0, len(columns), 7)])])

    if "clarify_question" in by_tool:
        data=by_tool["clarify_question"]
        target=data.get("target")
        examples=[f"Compare average {_name(target)} by a group" if target else "Count records by a group", "Show a trend over time", "Check whether any fields need cleaning"]
        return _build("I couldn't match that wording to a reliable calculation yet.", sections=[_section(f"Try a measurable question about {data['dataset']}", examples)], note="I won't substitute a general overview for an answer to a different question.")

    if "list_datasets" in by_tool:
        data = by_tool["list_datasets"]
        return _build(f"You have {len(data)} datasets available.", [{"label": "Total datasets", "value": str(len(data))}], [_section("Choose a dataset", [f"{item['display_name']} — {_number(item['rows'])} records" for item in data])])

    if "dataset_overview" in by_tool:
        data = by_tool["dataset_overview"]
        facts = [data.get("description")]
        if data.get("target_statistics"):
            stats = data["target_statistics"]
            facts.append(f"{_name(data['target']).capitalize()} averages {_number(stats['mean'])} per record; the median is {_number(stats['median'])}.")
        elif data.get("target_distribution"):
            facts.extend(f"{item['value']}: {item['percentage']}% ({_number(item['count'])} records)" for item in data["target_distribution"][:5])
        sections = [_section("What this data shows", facts), _section("Data quality", [f"{data['missing_pct']}% of cells are missing; {_number(data['duplicate_rows'])} exact duplicate rows."])]
        return _build(f"{data['display_name']} contains {_number(data['rows'])} records to analyze.", [{"label": "Records", "value": _number(data["rows"])}, {"label": "Fields", "value": str(data["columns"])}], sections)

    if "recommend_cleaning" in by_tool or "profile_dataset" in by_tool:
        profile = by_tool.get("profile_dataset", {})
        recommendation = by_tool.get("recommend_cleaning", {})
        issues = recommendation.get("issues", [])
        issue_text = [issue.get("message") or f"{_name(issue.get('column', ''))}: {_name(issue.get('type', 'issue'))}" for issue in issues[:5]]
        operations = recommendation.get("operations", [])
        sections = []
        if issue_text:
            sections.append(_section("Needs attention", issue_text))
        if recommendation:
            sections.append(_section("Proposed changes", [f"{_name(item['operation']).capitalize()} {_name(item.get('column', ''))}".strip() for item in operations] or ["No changes are recommended."]))
        return _build("This dataset needs a review before important analysis." if issues else "No major cleaning problem was found.", [{"label": "Missing cells", "value": f"{profile.get('missing_pct', 0)}%"}, {"label": "Duplicate rows", "value": _number(profile.get("duplicate_rows", 0))}], sections)

    if "recommend_features" in by_tool:
        operations = by_tool["recommend_features"].get("transformations", [])
        return _build("These fields could make the dataset more useful for analysis.", [{"label": "Proposed fields", "value": str(len(operations))}], [_section("Recommended preparation", [_name(item["operation"]).capitalize() for item in operations] or ["No extra fields are needed."])])

    if "recommend_models" in by_tool:
        data = by_tool["recommend_models"]
        return _build("These models are suitable candidates to test.", sections=[_section("Recommended models", [_name(item["algorithm"]).title() for item in data["candidates"]])])

    if "identify_problem" in by_tool:
        data = by_tool["identify_problem"]
        return _build(f"This is a {_name(data['task_type'])} question.", sections=[_section("Prediction goal", [f"Predict {_name(data['target'])} from the other available fields."])])

    if "correlations" in by_tool:
        data = by_tool["correlations"]
        target = state.get("target")
        columns = data.get("columns", [])
        strongest = []
        if target in columns:
            index = columns.index(target)
            strongest = sorted(((column, data["matrix"][row][index]) for row, column in enumerate(columns) if column != target and isinstance(data["matrix"][row][index], (int, float))), key=lambda pair: abs(pair[1]), reverse=True)[:4]
        return _build(f"I compared the measured relationships with {_name(target) if target else 'the numeric fields'}.", [{"label": "Fields compared", "value": str(len(columns))}], [_section("Strongest associations", [f"{_name(column)}: correlation {_number(value, 2)}" for column, value in strongest] or ["Open the data details for the full comparison."])], "Correlation shows association, not causation.")

    if "get_experiment_importance" in by_tool:
        data = by_tool["get_experiment_importance"]
        importance = data.get("feature_importance", [])[:5]
        return _build(f"{_name(importance[0]['feature']).capitalize()} was the strongest measured feature." if importance else "No feature-importance result was saved.", sections=[_section("Top features", [_name(item["feature"]) for item in importance])])

    if "get_experiment_evaluation" in by_tool:
        data = by_tool["get_experiment_evaluation"]
        return _build(f"Here is how {_name(data['algorithm']).title()} performed.", sections=[_section("Measured test results", [f"{_name(key).capitalize()}: {_number(value, 3)}" for key, value in data.get("metrics", {}).items()])])

    if "compare_saved_models" in by_tool:
        data = by_tool["compare_saved_models"]
        return _build(f"{_name(data['best_algorithm']).title()} ranked first among the saved models.", sections=[_section("Comparison", [f"{_name(row['algorithm']).title()}: {_name(data['primary_metric'])} {_number(row.get(data['primary_metric']), 3)}" for row in data.get("rows", [])]), _section("Why", [data.get("selection_note")])])

    if "retrieve_memory" in by_tool:
        memories = by_tool["retrieve_memory"]
        findings = [item.get("value", {}).get("answer") or item.get("value", {}).get("query") or item.get("key") for item in memories[:3]]
        return _build("Here is the relevant earlier work I found." if memories else "I found no matching earlier analysis.", [{"label": "Saved findings", "value": str(len(memories))}], [_section("Previous findings", [str(finding).split("\n", 1)[0][:250] for finding in findings])])

    for tool in ["auto_visualization", "visualize_last_result", "create_visualization"]:
        if tool in by_tool:
            data = by_tool[tool]
            return _build(f"I created “{data['title']}” from the selected data.", sections=[_section("Where to view it", ["Open the Charts tab here, or choose Charts from the main menu to revisit it later."])])

    if "readonly_sql" in by_tool:
        data = by_tool["readonly_sql"]
        return _build(f"The query returned {_number(data['row_count'])} rows.", [{"label": "Rows", "value": _number(data["row_count"])}], [_section("Result preview", [str(row) for row in data.get("rows", [])[:5]])])

    return _build("The requested calculation is complete.", sections=[_section("What was checked", [item["tool"].replace("_", " ") for item in results])])


def as_text(presentation):
    parts = [presentation["headline"]]
    if presentation.get("metrics"):
        parts.append("At a glance:\n" + "\n".join(f"• {item['label']}: {item['value']}" for item in presentation["metrics"]))
    for section in presentation["sections"]:
        if section["items"]:
            parts.append(section["title"] + ":\n" + "\n".join(f"• {item}" for item in section["items"]))
    if presentation.get("note"):
        parts.append(presentation["note"])
    return "\n\n".join(parts)
