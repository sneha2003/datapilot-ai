import json
from pathlib import Path
from app.agents.planner import create_plan

def run():
    path=Path(__file__).resolve().parents[3]/"evaluation"/"tool_selection.json"; cases=json.loads(path.read_text()); rows=[]
    for case in cases:
        plan=create_plan(case["query"],case.get("context",{"target":"target"}))
        actual=[x["tool"] for x in plan]; expected=case["expected_tools"]
        rows.append({"query":case["query"],"expected":expected,"actual":actual,"passed":actual==expected,"approval_compliant":not case.get("requires_approval") or any(x.get("approval") for x in plan)})
    passed=sum(x["passed"] and x["approval_compliant"] for x in rows); print(json.dumps({"cases":rows,"passed":passed,"total":len(rows),"task_completion_rate":passed/len(rows)},indent=2))
    raise SystemExit(0 if passed==len(rows) else 1)
if __name__=="__main__": run()
