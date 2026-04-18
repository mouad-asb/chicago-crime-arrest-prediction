import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import shap
import matplotlib.pyplot as plt

DB_URL = "postgresql+psycopg2://admin:admin@127.0.0.1:5433/chicago_crimes"
engine = create_engine(DB_URL)

# Load transformed data
print("Loading data from dbt feature table...")
df = pd.read_sql("SELECT * FROM analytics.feat_crimes", engine)
print(f"Shape: {df.shape}")

# Incorporate statistical analysis to engineer features
print("Engineering features...")

# Interaction feature: district x crime type (confounding)
df['district_crime_type'] = df['district'].astype(str) + '_' + df['crime_type']

# High-leakage crime types (class imbalance)
HIGH_LEAKAGE_TYPES = [
    'NARCOTICS', 'PROSTITUTION', 'GAMBLING',
    'LIQUOR LAW VIOLATION', 'PUBLIC INDECENCY',
    'CONCEALED CARRY LICENSE VIOLATION'
]
df['is_high_leakage_type'] = df['crime_type'].isin(HIGH_LEAKAGE_TYPES).astype(int)

# Encode categoricals
cat_cols = ['crime_type', 'time_of_day', 'location_description',
            'fbi_code', 'district_crime_type']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Train/test split (pre-2020 train/post-2020 test)
print("Splitting data...")
train = df[df['crime_year'] < 2020]
test  = df[df['crime_year'] >= 2020]

FEATURES = [
    'crime_year', 'crime_month', 'crime_dow', 'crime_hour',
    'is_weekend', 'time_of_day', 'crime_type', 'is_domestic',
    'location_description', 'fbi_code', 'beat', 'district',
    'ward', 'community_area', 'crime_type_arrest_rate',
    'crime_type_total', 'district_arrest_rate', 'district_total',
    'hour_arrest_rate', 'dow_arrest_rate', 'is_high_leakage_type',
    'district_crime_type'
]

X_train = train[FEATURES]
y_train = train['arrest']
X_test  = test[FEATURES]
y_test  = test['arrest']

print(f"Train: {len(X_train):,} rows | arrest rate: {y_train.mean():.3f}")
print(f"Test:  {len(X_test):,} rows  | arrest rate: {y_test.mean():.3f}")

# Model training (using LightGBM, can also use other GB-based algos)
print("\nTraining LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(50, first_metric_only=True),
        lgb.log_evaluation(100)
    ]
)

# General evaluation
print("\n" + "="*50)
print("EVALUATION")
print("="*50)

train_preds = model.predict_proba(X_train)[:, 1]
test_preds  = model.predict_proba(X_test)[:, 1]

print(f"Train AUC: {roc_auc_score(y_train, train_preds):.4f}")
print(f"Test AUC:  {roc_auc_score(y_test, test_preds):.4f}")

# Evaluate further the model on more difficult cases
test_hard = test[~test['crime_type'].isin(HIGH_LEAKAGE_TYPES)]
hard_preds = model.predict_proba(test_hard[FEATURES])[:, 1]
print(f"AUC on hard cases: {roc_auc_score(test_hard['arrest'], hard_preds):.4f}")

# Classification report at 0.5 threshold
print("\nClassification Report:")
print(classification_report(y_test, (test_preds > 0.5).astype(int),
                            target_names=['No Arrest', 'Arrest']))

# SHAP analysis
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(model)
X_sample = X_test.sample(2000, random_state=42)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, max_display=20)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to shap_summary.png")


joblib.dump(model, 'model.pkl')
print("\nModel saved to model.pkl")