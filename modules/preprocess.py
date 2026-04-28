
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder




def full_preprocess(df, target):
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)

    df = df.dropna()

    # Drop ID-like columns
    for col in df.columns:
        if "id" in col.lower():
            df.drop(columns=[col],inplace=True)

    X = df.drop(columns=[target])
    y = df[target]

    feature_names = list(X.columns)   # ⭐ IMPORTANT

    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoders = {}
    for col in X.select_dtypes(include="object"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    meta = {
        "features": feature_names,
        "encoders": encoders,
        "scaler": scaler
    }

    return X, y, meta
