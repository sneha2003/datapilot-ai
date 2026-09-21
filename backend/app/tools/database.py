import re
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from sqlglot import exp, parse_one
from app.config import get_settings
from app.database.models import Dataset

BLOCKED={"insert","update","delete","drop","alter","truncate","create","grant","revoke","copy","call","execute"}
DANGEROUS={"pg_read_file","pg_ls_dir","dblink","lo_import","lo_export","pg_sleep"}

class SQLValidationError(ValueError): pass

def list_datasets(db: Session):
    return [{"id":d.id,"name":d.name,"display_name":d.display_name,"table":d.table_name,"description":d.description,"source":d.source,"license":d.license,"rows":d.row_count,"columns":d.column_count,"tasks":d.task_candidates,"targets":d.target_suggestions,"version":d.version,"parent_id":d.parent_id} for d in db.query(Dataset).order_by(Dataset.display_name).all()]

def get_dataset_lineage(db:Session,dataset:Dataset):
    parent=db.get(Dataset,dataset.parent_id) if dataset.parent_id else None
    children=db.query(Dataset).filter_by(parent_id=dataset.id).order_by(Dataset.imported_at.desc()).all()
    return {"active":{"id":dataset.id,"name":dataset.name,"display_name":dataset.display_name,"version":dataset.version,"rows":dataset.row_count,"columns":dataset.column_count},"parent":{"id":parent.id,"display_name":parent.display_name,"version":parent.version} if parent else None,"derived_versions":[{"id":item.id,"display_name":item.display_name,"version":item.version} for item in children],"location":"Data sources panel on the left side of the analytics workspace"}

def resolve_dataset(db: Session, value: str|None):
    if value:
        key=value.lower().replace(" ","_")
        ds=db.query(Dataset).filter((Dataset.id==value)|(Dataset.name==key)|(Dataset.table_name==key)).first()
        if ds: return ds
        for item in db.query(Dataset).all():
            if item.name in key or key in item.name or item.display_name.lower() in value.lower(): return item
    return None

def get_dataset_schema(db: Session, dataset: Dataset):
    cols=inspect(db.bind).get_columns(dataset.table_name)
    return {"dataset":dataset.name,"table":dataset.table_name,"version":dataset.version,"task_candidates":dataset.task_candidates,"target_suggestions":dataset.target_suggestions,"columns":[{"name":c["name"],"type":str(c["type"]),"nullable":c.get("nullable",True),"description":dataset.column_descriptions.get(c["name"],"")} for c in cols]}

def sample_rows(db: Session,dataset: Dataset,limit:int=10):
    limit=max(1,min(limit,100)); rows=db.execute(text(f'SELECT * FROM "{dataset.table_name}" LIMIT :limit'),{"limit":limit}).mappings().all()
    return [dict(r) for r in rows]

def validate_sql(db: Session, query: str, max_rows: int|None=None):
    if not query.strip(): raise SQLValidationError("Query is empty")
    if ";" in query.rstrip().rstrip(";"): raise SQLValidationError("Multiple SQL statements are not allowed")
    lowered=re.sub(r"--.*?$|/\*.*?\*/","",query.lower(),flags=re.M|re.S)
    if any(re.search(rf"\b{x}\b",lowered) for x in BLOCKED): raise SQLValidationError("Only read-only SELECT queries are allowed")
    if any(x in lowered for x in DANGEROUS): raise SQLValidationError("Dangerous database functions are blocked")
    try: tree=parse_one(query,read="postgres")
    except Exception as exc: raise SQLValidationError(f"Invalid SQL syntax: {exc}") from exc
    if not isinstance(tree,(exp.Select,exp.Union,exp.Subquery)): raise SQLValidationError("Only SELECT queries are allowed")
    inspector=inspect(db.bind); available=set(inspector.get_table_names()); cte_names={cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    referenced={t.name for t in tree.find_all(exp.Table)}-cte_names
    unknown=referenced-available
    if unknown: raise SQLValidationError(f"Unknown table(s): {', '.join(sorted(unknown))}")
    known_columns={c["name"] for table in referenced for c in inspector.get_columns(table)}
    select_aliases={alias.alias for alias in tree.find_all(exp.Alias)}
    invalid={column.name for column in tree.find_all(exp.Column) if column.name!="*" and column.name not in known_columns and column.name not in select_aliases and column.table not in cte_names}
    if invalid: raise SQLValidationError(f"Unknown column(s): {', '.join(sorted(invalid))}")
    max_rows=max_rows or get_settings().sql_max_rows
    if tree.args.get("limit") is None: query=query.rstrip().rstrip(";")+f" LIMIT {max_rows}"
    return query

def execute_readonly_sql(db: Session,query:str):
    safe=validate_sql(db,query)
    if db.bind.dialect.name=="postgresql": db.execute(text(f"SET LOCAL statement_timeout = {get_settings().sql_timeout_ms}"))
    result=db.execute(text(safe)); columns=list(result.keys()); rows=[dict(r) for r in result.mappings().all()]
    return {"query":safe,"columns":columns,"rows":rows,"row_count":len(rows),"truncated":len(rows)>=get_settings().sql_max_rows}

def dataframe(db: Session,dataset: Dataset):
    import pandas as pd
    return pd.read_sql(text(f'SELECT * FROM "{dataset.table_name}"'),db.bind)
