"""Download documented public datasets. Run from the repository root."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)

def save_openml(name: str, data_id: int, target_name: str | None = None):
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    # OpenML's combined frame can contain the target twice for older datasets.
    # Build from the feature frame and attach one explicitly named target.
    frame = bunch.data.copy()
    if target_name and bunch.target is not None:
        frame[target_name] = bunch.target.to_numpy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate columns returned for {name}: {frame.columns[frame.columns.duplicated()].tolist()}")
    frame.to_csv(RAW / f"{name}.csv", index=False)
    print(f"Downloaded {name}: {frame.shape}")

def save_bank_marketing():
    bunch=fetch_openml(data_id=1461,as_frame=True,parser="auto")
    frame=bunch.data.copy()
    official=["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]
    if len(frame.columns)!=len(official): raise ValueError("Unexpected Bank Marketing schema from OpenML")
    frame.columns=official
    encoded=bunch.target.astype(str); frame["y"]=encoded.map({"1":"no","2":"yes"}).fillna(encoded).to_numpy()
    frame.to_csv(RAW/"bank_marketing.csv",index=False)
    print(f"Downloaded bank_marketing: {frame.shape}")

def main():
    save_openml("titanic", 40945, "survived")
    save_bank_marketing()
    save_openml("bike_sharing", 42712, "count")
    save_openml("wine_quality", 287, "quality")
    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.rename(columns={"MedHouseVal":"median_house_value"})
    frame.columns = [c.lower() for c in frame.columns]
    frame.to_csv(RAW / "california_housing.csv", index=False)
    print(f"Downloaded california_housing: {frame.shape}")

if __name__ == "__main__": main()
