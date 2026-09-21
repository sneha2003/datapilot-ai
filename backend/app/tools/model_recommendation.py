import pandas as pd
from app.tools.database import dataframe

def identify_ml_problem(db,dataset,target,objective=None):
    df=dataframe(db,dataset)
    if target not in df: raise ValueError(f"Target '{target}' does not exist")
    y=df[target].dropna(); unique=y.nunique()
    if objective:
        task=objective
    elif not pd.api.types.is_numeric_dtype(y) or unique<=2: task="binary_classification" if unique==2 else "multiclass_classification"
    elif unique<=20 and unique/max(len(y),1)<.05: task="multiclass_classification"
    else: task="regression"
    if unique<2: raise ValueError("Target must contain at least two values")
    return {"target":target,"task_type":task,"rows":len(y),"classes":int(unique),"class_distribution":{str(k):int(v) for k,v in y.value_counts().items()} if "classification" in task else None,"reason":"User objective" if objective else "Target dtype and cardinality"}

def recommend_models(db,dataset,target,task_type,interpretability=False):
    df=dataframe(db,dataset); n=len(df); p=len(df.columns)-1; imbalance=None
    if "classification" in task_type:
        counts=df[target].value_counts(); imbalance=float(counts.min()/counts.max()) if len(counts) else 1
        candidates=[{"algorithm":"logistic_regression","why":"Interpretable calibrated baseline","primary_metric":"f1" if imbalance<.5 else "roc_auc"},{"algorithm":"random_forest","why":"Nonlinear interactions and mixed tabular data","primary_metric":"f1"},{"algorithm":"gradient_boosting","why":"Strong tabular predictive performance","primary_metric":"roc_auc"}]
        if n<5000 and p<50: candidates.append({"algorithm":"svm","why":"Suitable for a moderate-sized feature space","primary_metric":"f1"})
    else:
        candidates=[{"algorithm":"ridge","why":"Stable interpretable linear baseline","primary_metric":"rmse"},{"algorithm":"random_forest","why":"Nonlinear relationships with little tuning","primary_metric":"rmse"},{"algorithm":"gradient_boosting","why":"Strong tabular regression performance","primary_metric":"rmse"}]
    return {"task_type":task_type,"dataset_shape":[n,p+1],"imbalance_ratio":imbalance,"candidates":candidates[:2] if interpretability else candidates[:4]}
