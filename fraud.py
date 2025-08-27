import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------- 1. Load dataset ----------------
print("🔹 Loading dataset...")
df = pd.read_csv(r"C:\Users\Lenvo\OneDrive\Desktop\Programming\python_codes\creditcard.csv")
print("Shape:", df.shape)
print(df['Class'].value_counts())

# ---------------- 2. Preprocessing ----------------
features = [c for c in df.columns if c != 'Class']
X = df[features].copy()
y = df['Class'].copy()

# Create time-based features
if 'Time' in X.columns:
    X['hour'] = (X['Time'] // 3600) % 24
    X.drop(columns=['Time'], inplace=True)

# Scale Amount and hour
scaler = StandardScaler()
if 'Amount' in X.columns:
    X['Amount_scaled'] = scaler.fit_transform(X[['Amount']])
    X.drop(columns=['Amount'], inplace=True)
if 'hour' in X.columns:
    X['hour_sin'] = np.sin(2*np.pi*X['hour']/24)
    X['hour_cos'] = np.cos(2*np.pi*X['hour']/24)
    X.drop(columns=['hour'], inplace=True)

print("✅ Features ready:", X.shape)

# ---------------- 3. Train/test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train fraud ratio:", y_train.mean(), "Test fraud ratio:", y_test.mean())

# ---------------- 4. Logistic Regression + SMOTE ----------------
pipe_lr = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_pred_lr)
ap_lr = average_precision_score(y_test, y_pred_lr)
print(f"🔹 Logistic Regression: AUC={auc_lr:.4f}, AP={ap_lr:.4f}")

# ---------------- 5. Random Forest ----------------
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:,1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
ap_rf = average_precision_score(y_test, y_pred_rf)
print(f"🔹 Random Forest: AUC={auc_rf:.4f}, AP={ap_rf:.4f}")

# ---------------- 6. XGBoost ----------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'test')], early_stopping_rounds=20, verbose_eval=20)
y_pred_xgb = bst.predict(dtest)
auc_xgb = roc_auc_score(y_test, y_pred_xgb)
ap_xgb = average_precision_score(y_test, y_pred_xgb)
print(f"🔹 XGBoost: AUC={auc_xgb:.4f}, AP={ap_xgb:.4f}")

# ---------------- 7. Isolation Forest ----------------
iso = IsolationForest(n_estimators=200, contamination=y_train.mean(), random_state=42)
iso.fit(X_train)
iso_scores = -iso.decision_function(X_test)  # higher = more anomalous
auc_iso = roc_auc_score(y_test, iso_scores)
ap_iso = average_precision_score(y_test, iso_scores)
print(f"🔹 Isolation Forest: AUC={auc_iso:.4f}, AP={ap_iso:.4f}")

# ---------------- 8. Autoencoder ----------------
input_dim = X_train.shape[1]
encoding_dim = input_dim // 2
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(encoding_dim*2, activation='relu'),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(encoding_dim*2, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')

X_train_legit = X_train[y_train==0]
ae_scaler = StandardScaler()
X_train_legit_scaled = ae_scaler.fit_transform(X_train_legit)
X_test_scaled = ae_scaler.transform(X_test)

es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
autoencoder.fit(X_train_legit_scaled, X_train_legit_scaled,
                epochs=20, batch_size=2048, validation_split=0.1,
                callbacks=[es], verbose=1)

reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.square(reconstructions - X_test_scaled), axis=1)
auc_ae = roc_auc_score(y_test, mse)
ap_ae = average_precision_score(y_test, mse)
print(f"🔹 Autoencoder: AUC={auc_ae:.4f}, AP={ap_ae:.4f}")

# ---------------- 9. Combined Ensemble ----------------
scores_df = pd.DataFrame({
    'lr': y_pred_lr,
    'rf': y_pred_rf,
    'xgb': y_pred_xgb,
    'iso': iso_scores,
    'ae': mse
})
scaler_scores = MinMaxScaler()
scores_scaled = pd.DataFrame(
    scaler_scores.fit_transform(scores_df),
    columns=scores_df.columns
)
weights = {'lr':0.1, 'rf':0.1, 'xgb':0.5, 'iso':0.15, 'ae':0.15}
combined_score = sum(weights[c] * scores_scaled[c] for c in weights)
auc_combined = roc_auc_score(y_test, combined_score)
ap_combined = average_precision_score(y_test, combined_score)
print(f"✅ Combined Model: AUC={auc_combined:.4f}, AP={ap_combined:.4f}")

# ---------------- 10. Confusion Matrix ----------------
threshold = np.percentile(combined_score, 99)
y_pred_comb = (combined_score >= threshold).astype(int)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_comb))
print("Classification Report:\n", classification_report(y_test, y_pred_comb, digits=4))

# ---------------- 11. Save Models ----------------
joblib.dump(pipe_lr, 'model_logistic_smote.joblib')
joblib.dump(rf, 'model_rf.joblib')
bst.save_model('model_xgb.json')
autoencoder.save('autoencoder_model.h5')
joblib.dump(ae_scaler, 'ae_scaler.joblib')
joblib.dump(scaler, 'amount_scaler.joblib')
print("Models saved.")
