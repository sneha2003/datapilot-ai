import json, uuid
from pathlib import Path
import pandas as pd
import plotly.express as px
from app.config import get_settings
from app.tools.database import dataframe

ALLOWED={"bar","line","scatter","histogram","box","heatmap","donut"}
def validate_spec(df,spec):
    if spec.get("chart_type") not in ALLOWED: raise ValueError("Unsupported chart type")
    for key in ("x","y","color"):
        if spec.get(key) and spec[key] not in df.columns: raise ValueError(f"Unknown column: {spec[key]}")

def create_visualization(db,dataset,spec):
    df=dataframe(db,dataset); validate_spec(df,spec); chart=spec["chart_type"]; x=spec.get("x"); y=spec.get("y"); color=spec.get("color"); agg=spec.get("aggregation"); title=spec.get("title",f"{dataset.display_name} — {chart}")
    plot=df
    if agg and x:
        if agg=="count": plot=df.groupby(x,dropna=False).size().reset_index(name="count"); y="count"
        elif agg in {"mean","sum","median"}: plot=getattr(df.groupby(x,dropna=False)[y],agg)().reset_index()
    if len(plot)>5000: plot=plot.sample(5000,random_state=42)
    if chart=="bar": fig=px.bar(plot,x=x,y=y,color=color,title=title)
    elif chart=="line": fig=px.line(plot,x=x,y=y,color=color,title=title)
    elif chart=="scatter": fig=px.scatter(plot,x=x,y=y,color=color,title=title)
    elif chart=="histogram": fig=px.histogram(plot,x=x,color=color,title=title)
    elif chart=="box": fig=px.box(plot,x=x,y=y,color=color,title=title)
    elif chart=="donut": fig=px.pie(plot,names=x,values=y,hole=.55,title=title)
    else:
        corr=df.select_dtypes("number").corr(); fig=px.imshow(corr,text_auto=".2f",aspect="auto",title=title,color_continuous_scale="RdBu_r",zmin=-1,zmax=1)
    fig.update_layout(template="plotly_white",font_family="Inter, sans-serif",margin=dict(l=40,r=20,t=60,b=40))
    root=Path(get_settings().artifact_root)/"plots"; root.mkdir(parents=True,exist_ok=True); path=root/f"{uuid.uuid4()}.json"; path.write_text(fig.to_json(),encoding="utf-8")
    return {"title":title,"specification":spec,"figure":json.loads(fig.to_json()),"path":str(path),"data_rows":len(plot)}

def create_result_visualization(rows,spec):
    df=pd.DataFrame(rows)
    if df.empty: raise ValueError("Analysis result is empty")
    validate_spec(df,spec); title=spec.get("title","Analysis result"); chart=spec["chart_type"]
    if chart=="bar": fig=px.bar(df,x=spec.get("x"),y=spec.get("y"),color=spec.get("color"),title=title)
    elif chart=="line": fig=px.line(df,x=spec.get("x"),y=spec.get("y"),color=spec.get("color"),title=title)
    else: raise ValueError("Result tables support bar and line charts")
    fig.update_layout(template="plotly_white",font_family="Inter, sans-serif",margin=dict(l=40,r=20,t=60,b=40))
    root=Path(get_settings().artifact_root)/"plots"; root.mkdir(parents=True,exist_ok=True); path=root/f"{uuid.uuid4()}.json"; path.write_text(fig.to_json(),encoding="utf-8")
    return {"title":title,"specification":spec,"figure":json.loads(fig.to_json()),"path":str(path),"data_rows":len(df)}
