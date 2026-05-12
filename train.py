import numpy as np
import pandas as pd
import pickle
import gzip
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception):
    XGB_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = 'telco_churn.csv'
MODEL_PATH = 'model.pkl'

print("=" * 60)
print("TELCO CUSTOMER CHURN - MODEL TRAINING")
print("=" * 60)

try:
    data = pd.read_csv(CSV_PATH)
    print(f"\nLoaded existing dataset: {CSV_PATH}")
except FileNotFoundError:
    print(f"\n{CSV_PATH} not found. Generating synthetic data...")
    n = 7043
    data = pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52]),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
        'tenure': np.random.randint(1, 73, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.35, 0.43, 0.22]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.25, 0.53, 0.22]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.48, 0.22]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.48, 0.22]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.60, 0.40]),
        'PaymentMethod': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            n, p=[0.34, 0.22, 0.22, 0.22]
        ),
        'MonthlyCharges': np.round(np.random.uniform(18, 120, n), 2),
    })
    data['TotalCharges'] = np.round(data['tenure'] * data['MonthlyCharges'] * np.random.uniform(0.9, 1.1, n), 2)
    churn_prob = (
        0.15 + 0.25 * (data['Contract'] == 'Month-to-month')
        + 0.15 * (data['InternetService'] == 'Fiber optic')
        + 0.10 * (data['OnlineSecurity'] == 'No')
        + 0.10 * (data['TechSupport'] == 'No')
        + 0.05 * (data['PaymentMethod'] == 'Electronic check')
        - 0.15 * (data['tenure'] > 12).astype(int)
        - 0.10 * (data['Contract'] == 'Two year')
        + np.random.uniform(-0.1, 0.1, n)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    data['Churn'] = np.random.binomial(1, churn_prob)
    data['Churn'] = data['Churn'].map({1: 'Yes', 0: 'No'})
    data.to_csv(CSV_PATH, index=False)
    print(f"Generated synthetic dataset: {CSV_PATH}")

print(f"\nDataset shape: {data.shape}")
print(f"Churn distribution:\n{data['Churn'].value_counts()}")
churn_rate = data['Churn'].value_counts(normalize=True)['Yes']
print(f"Churn rate: {churn_rate:.2%}")

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

cat_cols = data.select_dtypes(include='object').columns.drop('Churn')
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

X = data.drop('Churn', axis=1)
y = LabelEncoder().fit_transform(data['Churn'])

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train size: {X_train.shape[0]} rows")
print(f"Test size: {X_test.shape[0]} rows")

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" + "=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                            n_jobs=-1, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                     random_state=RANDOM_STATE),
}

if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
model_report = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n  Training {name}...")

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = {
        'model': model,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_proba),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }

    model_report[name] = report

    print(f"    CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"    Test Accuracy: {report['test_accuracy']:.4f}")
    print(f"    Test AUC: {report['test_auc']:.4f}")
    print(f"    Precision: {report['test_precision']:.4f}")
    print(f"    Recall: {report['test_recall']:.4f}")
    print(f"    F1 Score: {report['test_f1']:.4f}")

    if report['test_auc'] > best_score:
        best_score = report['test_auc']
        best_model = model
        best_model_name = name

print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_model_name} (Test AUC: {best_score:.4f})")
print(f"{'=' * 60}")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
elif best_model_name == 'XGBoost' and XGB_AVAILABLE:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
    }
    base_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
    }
    base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
else:
    param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
    base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

print(f"\nHyperparameter tuning {best_model_name}...")
grid = GridSearchCV(base_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV AUC: {grid.best_score_:.4f}")

final_model = grid.best_estimator_
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]
final_auc = roc_auc_score(y_test, y_proba)
final_acc = accuracy_score(y_test, y_pred)

print(f"Tuned Test AUC: {final_auc:.4f}")
print(f"Tuned Test Accuracy: {final_acc:.4f}")

if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
else:
    importances = np.abs(final_model.coef_[0])

feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

model_report['feature_importances'] = feat_imp_df

artifacts = {
    'model': final_model,
    'scaler': scaler,
    'label_encoders': le_dict,
    'feature_names': feature_names,
    'model_report': model_report,
    'best_model_name': best_model_name,
    'best_params': grid.best_params_,
    'final_test_auc': final_auc,
    'final_test_accuracy': final_acc,
}

with gzip.open(MODEL_PATH + '.gz', 'wb', compresslevel=9) as f:
    pickle.dump(artifacts, f)

print(f"\nModel saved as {MODEL_PATH}.gz (compressed)")
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
