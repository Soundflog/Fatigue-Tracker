import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def fit_lr(X, y):
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(Xn, y)
    return (scaler, clf)

def predict_lr(model, X):
    scaler, clf = model
    Xn = scaler.transform(X)
    return clf.predict_proba(Xn)[:,1]

def fit_rf(X, y):
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf

def predict_rf(model, X):
    return model.predict_proba(X)[:,1]
