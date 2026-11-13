"""Grouped classification training; run python backend/train_model.py from the project root."""
from pathlib import Path
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
FEATURES = ['Occupation', 'self_employed', 'family_history', 'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness']
TARGET = 'treatment'

def train():
    raw = pd.read_csv(ROOT / 'data' / 'Mental Health Dataset.csv')
    data = raw.copy()
    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].str.strip().replace('', np.nan)
    data = data.drop_duplicates()
    data = data[data[TARGET].isin(['Yes', 'No'])].reset_index(drop=True)
    X, y = data[FEATURES], data[TARGET]
    groups = data['Timestamp'].fillna('unknown').astype(str)
    train_idx, test_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=43).split(X_train, y_train, groups.iloc[train_idx]))
    candidates = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic_regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        'naive_bayes': BernoulliNB(),
        'random_forest': RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=10, bootstrap=True, max_samples=0.1, n_jobs=2, random_state=42),
        'extra_trees': ExtraTreesClassifier(n_estimators=60, max_depth=10, min_samples_leaf=10, bootstrap=True, max_samples=0.1, n_jobs=2, random_state=42),
    }
    comparison, pipelines = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)), ('classifier', estimator)])
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring={'f1_macro':'f1_macro', 'balanced_accuracy':'balanced_accuracy', 'roc_auc':'roc_auc'}, error_score='raise')
        comparison[name] = {k[5:]:float(v.mean()) for k,v in scores.items() if k.startswith('test_')}
        comparison[name]['f1_macro_std'] = float(scores['test_f1_macro'].std())
        pipelines[name] = pipeline
        print(name, comparison[name], flush=True)
    winner = max(comparison, key=lambda k: comparison[k]['f1_macro'])
    model = pipelines[winner].fit(X_train, y_train)
    truth = y.iloc[test_idx]
    predicted = model.predict(X.iloc[test_idx])
    classes = model.classes_.tolist()
    probs = model.predict_proba(X.iloc[test_idx])[:,classes.index('Yes')]
    metrics = {
        'accuracy':float(accuracy_score(truth,predicted)),
        'precision_weighted':float(precision_score(truth,predicted,average='weighted',zero_division=0)),
        'recall_weighted':float(recall_score(truth,predicted,average='weighted',zero_division=0)),
        'f1_weighted':float(f1_score(truth,predicted,average='weighted',zero_division=0)),
        'f1_macro':float(f1_score(truth,predicted,average='macro')),
        'roc_auc':float(roc_auc_score(truth=='Yes',probs)),
        'brier_score':float(brier_score_loss(truth=='Yes',probs)),
        'classification_report':classification_report(truth,predicted,output_dict=True,zero_division=0),
        'classes':classes, 'confusion_matrix':confusion_matrix(truth,predicted,labels=classes).tolist(),
        'training_rows':len(train_idx), 'test_rows':len(test_idx), 'source_rows':len(raw),
        'cleaned_rows':len(data), 'removed_rows':len(raw)-len(data), 'selected_model':winner,
        'cross_validation':comparison,
        'split_method':'Timestamp-grouped stratified holdout; 3-fold grouped CV on training only',
        'group_overlap':len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])),
        'train_groups':groups.iloc[train_idx].nunique(), 'test_groups':groups.iloc[test_idx].nunique(),
        'target_distribution':{str(k):int(v) for k,v in y.value_counts().items()},
        'missing_values_by_feature':{str(k):int(v) for k,v in X.isna().sum().items()},
        'trained_at':datetime.now(timezone.utc).isoformat(),
        'limitations':'Timestamp is a proxy, not a verified respondent ID. No external or clinical validation. Probabilities are uncalibrated model scores.'}
    versions = {'scikit-learn':sklearn.__version__, 'pandas':pd.__version__, 'numpy':np.__version__, 'joblib':joblib.__version__}
    options = {c:sorted(X[c].dropna().unique().tolist()) for c in FEATURES}
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    joblib.dump({'pipeline':model,'features':FEATURES,'target':TARGET,'classes':classes,'metrics':metrics,'options':options,'versions':versions}, folder/'mental_health_model.joblib')
    (folder/'model_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (folder/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    print('Selected:',winner,'Holdout accuracy:',metrics['accuracy'],flush=True)

if __name__ == '__main__':
    train()


