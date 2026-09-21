import time, uuid
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from app.config import get_settings
from app.database.models import ModelExperiment
from app.tools.database import dataframe

def _model(name,classification,random_state=42):
    models={
      "logistic_regression":LogisticRegression(max_iter=1000,class_weight="balanced",random_state=random_state),
      "random_forest":RandomForestClassifier(n_estimators=120,class_weight="balanced",n_jobs=1,random_state=random_state) if classification else RandomForestRegressor(n_estimators=120,n_jobs=1,random_state=random_state),
      "gradient_boosting":GradientBoostingClassifier(random_state=random_state) if classification else GradientBoostingRegressor(random_state=random_state),
      "svm":SVC(probability=True,class_weight="balanced",random_state=random_state),
      "ridge":Ridge(alpha=1.0)
    }
    if name not in models: raise ValueError(f"Unsupported algorithm: {name}")
    return models[name]

def _pipeline(X,model):
    numeric=X.select_dtypes(include=np.number).columns.tolist(); categorical=[c for c in X if c not in numeric]
    prep=ColumnTransformer([("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",min_frequency=2))]),categorical)],remainder="drop")
    return Pipeline([("preprocess",prep),("model",model)]),numeric,categorical

def train_model(db,dataset,target,task_type,algorithm,session_id=None,time_column=None):
    started=time.perf_counter(); df=dataframe(db,dataset).dropna(subset=[target]); excluded=[c for c,description in dataset.column_descriptions.items() if c in df and "leakage" in str(description).lower()]; X=df.drop(columns=[target,*excluded]); y=df[target]; classification="classification" in task_type; encoder=None
    if classification:
        encoder=LabelEncoder(); y=pd.Series(encoder.fit_transform(y.astype(str)),index=y.index,name=target)
    pipe,numeric,categorical=_pipeline(X,_model(algorithm,classification))
    temporal=bool(time_column and time_column in X)
    if temporal:
        order=pd.to_datetime(X[time_column],errors="coerce").sort_values().index; X=X.loc[order]; y=y.loc[order]; cut=int(len(X)*.8); X_train,X_test=X.iloc[:cut],X.iloc[cut:]; y_train,y_test=y.iloc[:cut],y.iloc[cut:]
    else:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y if classification else None)
    pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
    if classification:
        average="binary" if y.nunique()==2 else "weighted"; metrics={"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,average=average,zero_division=0),"recall":recall_score(y_test,pred,average=average,zero_division=0),"f1":f1_score(y_test,pred,average=average,zero_division=0)}
        score=pipe.predict_proba(X_test) if hasattr(pipe,"predict_proba") else None
        if score is not None and y.nunique()==2:
            metrics["roc_auc"]=roc_auc_score(y_test,score[:,1]); metrics["pr_auc"]=average_precision_score(y_test,score[:,1]); fpr,tpr,_=roc_curve(y_test,score[:,1]); roc={"fpr":fpr.tolist(),"tpr":tpr.tolist()}
        else: roc=None
        matrix=confusion_matrix(y_test,pred).tolist(); scoring={"accuracy":"accuracy","f1":"f1_weighted" if y.nunique()>2 else "f1","roc_auc":"roc_auc_ovr_weighted" if y.nunique()>2 else "roc_auc"}; cv=StratifiedKFold(5,shuffle=True,random_state=42)
    else:
        metrics={"mae":mean_absolute_error(y_test,pred),"mse":mean_squared_error(y_test,pred),"rmse":mean_squared_error(y_test,pred)**.5,"r2":r2_score(y_test,pred)}; matrix=None; roc=None; scoring={"mae":"neg_mean_absolute_error","rmse":"neg_root_mean_squared_error","r2":"r2"}; cv=KFold(5,shuffle=True,random_state=42)
    # Single-process CV is slower on large machines but reliable in local Windows,
    # containers, and worker processes where nested process pools can fail.
    raw_cv=cross_validate(pipe,X,y,cv=cv,scoring=scoring,n_jobs=1,error_score="raise")
    cv_results={k.replace("test_",""):{"mean":float(np.mean(-v if k in {"test_mae","test_rmse"} else v)),"std":float(np.std(v)),"folds":[float(abs(x)) if k in {"test_mae","test_rmse"} else float(x) for x in v]} for k,v in raw_cv.items() if k.startswith("test_")}
    perm=permutation_importance(pipe,X_test,y_test,n_repeats=5,random_state=42,scoring="f1_weighted" if classification else "neg_root_mean_squared_error",n_jobs=1)
    importance=sorted([{"feature":c,"importance":float(v),"std":float(s)} for c,v,s in zip(X.columns,perm.importances_mean,perm.importances_std)],key=lambda z:z["importance"],reverse=True)
    cv_results["feature_importance"]=importance[:20]
    cv_results["confusion_matrix"]=matrix
    cv_results["roc_curve"]=roc
    cv_results["split"]={"train":len(X_train),"test":len(X_test),"strategy":"chronological" if temporal else "stratified_random" if classification else "random"}
    root=Path(get_settings().artifact_root)/"models"; root.mkdir(parents=True,exist_ok=True); experiment_id=str(uuid.uuid4()); path=root/f"{experiment_id}.joblib"; joblib.dump({"pipeline":pipe,"target_encoder":encoder,"dataset":dataset.name,"target":target,"task_type":task_type},path)
    metrics={k:round(float(v),6) for k,v in metrics.items()}; duration=(time.perf_counter()-started)*1000
    exp=ModelExperiment(id=experiment_id,session_id=session_id,dataset_id=dataset.id,target=target,task_type=task_type,algorithm=algorithm,parameters=pipe.named_steps["model"].get_params(),metrics=metrics,cv_results=cv_results,artifact_path=str(path),duration_ms=duration); db.add(exp); db.commit()
    return {"experiment_id":experiment_id,"algorithm":algorithm,"task_type":task_type,"metrics":metrics,"cross_validation":cv_results,"confusion_matrix":matrix,"roc_curve":roc,"feature_importance":importance[:20],"split":{"train":len(X_train),"test":len(X_test),"strategy":"chronological" if temporal else "stratified_random" if classification else "random"},"features":{"numeric":numeric,"categorical":categorical,"excluded_as_leakage":excluded},"duration_ms":round(duration,1),"artifact_path":str(path)}

def compare_models(results):
    if not results: raise ValueError("No model results")
    classification="classification" in results[0]["task_type"]
    metric="f1" if classification else "rmse"; best=max(results,key=lambda r:r["metrics"][metric]) if classification else min(results,key=lambda r:r["metrics"][metric])
    return {"primary_metric":metric,"best_experiment_id":best["experiment_id"],"best_algorithm":best["algorithm"],"rows":[{"experiment_id":r["experiment_id"],"algorithm":r["algorithm"],**r["metrics"]} for r in results],"selection_note":"Selected by F1; review precision/recall trade-offs." if classification else "Selected by lowest RMSE; review MAE and R² as well."}

def evaluate_model(db,experiment_id):
    exp=db.get(ModelExperiment,experiment_id)
    if not exp: raise ValueError("Experiment not found")
    return {"id":exp.id,"algorithm":exp.algorithm,"target":exp.target,"task_type":exp.task_type,"metrics":exp.metrics,"cross_validation":exp.cv_results,"artifact_exists":Path(exp.artifact_path).exists()}

def feature_importance(db,experiment_id):
    exp=db.get(ModelExperiment,experiment_id)
    if not exp: raise ValueError("Experiment not found")
    return {"experiment_id":experiment_id,"algorithm":exp.algorithm,"target":exp.target,"feature_importance":exp.cv_results.get("feature_importance",[]),"method":"permutation importance"}

def compare_saved_models(db,experiment_ids):
    experiments=[db.get(ModelExperiment,item) for item in experiment_ids]
    experiments=[item for item in experiments if item]
    if not experiments: raise ValueError("No trained models were found in this analysis")
    classification="classification" in experiments[0].task_type
    metric="f1" if classification else "rmse"
    best=max(experiments,key=lambda item:item.metrics.get(metric,float("-inf"))) if classification else min(experiments,key=lambda item:item.metrics.get(metric,float("inf")))
    return {
        "primary_metric":metric,
        "best_experiment_id":best.id,
        "best_algorithm":best.algorithm,
        "rows":[{"experiment_id":item.id,"algorithm":item.algorithm,**item.metrics} for item in experiments],
        "selection_note":"Selected by F1 while also showing precision, recall, and ROC-AUC." if classification else "Selected by lowest RMSE while also showing MAE and R².",
    }
