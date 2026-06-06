import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv('leads_data.csv')
print("✅ Data loaded:", df.shape)
print("\nConversion Rate:", round(df['converted'].mean()*100, 2), "%")

# ── 2. PREPROCESSING ──────────────────────────────────────
le = LabelEncoder()
df['company_size_enc'] = le.fit_transform(df['company_size'])
df['industry_enc'] = le.fit_transform(df['industry'])

features = [
    'age', 'annual_income', 'website_visits', 'emails_opened',
    'demo_requested', 'response_time_hours', 'previous_purchase',
    'company_size_enc', 'industry_enc'
]

X = df[features]
y = df['converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 3. TRAIN MODELS ───────────────────────────────────────
print("\n🤖 Training Models...")

# Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# Model 2: Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)

print(f"\n📊 Random Forest Accuracy:     {round(rf_acc*100, 2)}%")
print(f"📊 Logistic Regression Accuracy: {round(lr_acc*100, 2)}%")

# ── 4. DETAILED REPORT ────────────────────────────────────
print("\n📋 Random Forest — Full Report:")
print(classification_report(y_test, rf_preds))

# ── 5. FEATURE IMPORTANCE ─────────────────────────────────
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔑 Feature Importance (What drives conversion):")
print(importance_df.to_string(index=False))

# ── 6. LEAD SCORING FUNCTION ──────────────────────────────
def score_lead(age, annual_income, website_visits, emails_opened,
               demo_requested, response_time_hours, previous_purchase,
               company_size, industry):

    cs_map = {'Large': 0, 'Medium': 2, 'Small': 1}
    ind_map = {'Finance': 0, 'Healthcare': 1, 'Manufacturing': 2, 'Retail': 3, 'Tech': 4}

    input_data = pd.DataFrame([{
        'age': age,
        'annual_income': annual_income,
        'website_visits': website_visits,
        'emails_opened': emails_opened,
        'demo_requested': demo_requested,
        'response_time_hours': response_time_hours,
        'previous_purchase': previous_purchase,
        'company_size_enc': cs_map.get(company_size, 1),
        'industry_enc': ind_map.get(industry, 4)
    }])

    prob = rf.predict_proba(input_data)[0][1]
    score = round(prob * 100, 1)

    if score >= 70:
        priority = "🔥 HIGH PRIORITY"
    elif score >= 40:
        priority = "⚡ MEDIUM PRIORITY"
    else:
        priority = "❄️ LOW PRIORITY"

    print(f"\n{'='*40}")
    print(f"  LEAD SCORE: {score}/100")
    print(f"  STATUS: {priority}")
    print(f"{'='*40}")
    return score

# ── 7. TEST THE SCORER ────────────────────────────────────
print("\n🧪 Testing Lead Scorer on Sample Leads:")

print("\nLead A — Hot prospect (large tech company, demo requested):")
score_lead(35, 1500000, 15, 8, 1, 5, 1, 'Large', 'Tech')

print("\nLead B — Cold prospect (small retail, no engagement):")
score_lead(45, 400000, 1, 0, 0, 60, 0, 'Small', 'Retail')

print("\nLead C — Medium prospect (medium finance, some engagement):")
score_lead(40, 900000, 7, 4, 0, 24, 1, 'Medium', 'Finance')

# ── 8. SAVE VISUALIZATIONS ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
axes[0].set_title('Feature Importance — What Drives Conversion', fontweight='bold')
axes[0].set_xlabel('Importance Score')

cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix — Random Forest', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('lead_scorer_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Chart saved as lead_scorer_results.png")

# ── 9. SAVE MODEL & SCALER ✅ (this is the new part, at the END) ──
joblib.dump(rf, 'lead_scorer_model.pkl')
joblib.dump(scaler, 'lead_scorer_scaler.pkl')
print("💾 Model saved! lead_scorer_model.pkl is ready.")

print("\n🎉 Model Complete!")