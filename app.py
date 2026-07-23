from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.')
CORS(app)

# Load the saved model & scaler
rf = joblib.load('lead_scorer_model.pkl')
scaler = joblib.load('lead_scorer_scaler.pkl')
print("✅ Model loaded successfully!")

# ── Serve the UI ──────────────────────────────────────────
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# ── Score a lead ──────────────────────────────────────────
@app.route('/score', methods=['POST'])
def score_lead():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'A JSON request body is required.'}), 400

        cs_map = {'Large': 0, 'Medium': 2, 'Small': 1}
        ind_map = {'Finance': 0, 'Healthcare': 1, 'Manufacturing': 2, 'Retail': 3, 'Tech': 4}

        input_data = pd.DataFrame([{
            'age':                  float(data.get('age', 35)),
            'annual_income':        float(data.get('annual_income', 60000)),
            'website_visits':       float(data.get('website_visits', 0)),
            'emails_opened':        float(data.get('emails_opened', 0)),
            'demo_requested':       float(data.get('demo_requested', 0)),
            'response_time_hours':  float(data.get('response_time_hours', 24)),
            'previous_purchase':    float(data.get('previous_purchase', 0)),
            'company_size_enc':     float(cs_map.get(data.get('company_size', 'Medium'), 2)),
            'industry_enc':         float(ind_map.get(data.get('industry', 'Tech'), 4)),
        }])

        if input_data.isna().any().any():
            return jsonify({'error': 'All numeric fields must be valid numbers.'}), 400

        prob = rf.predict_proba(input_data)[0][1]
        score = round(prob * 100, 1)

        if score >= 70:
            status = "🔥 HIGH PRIORITY"
        elif score >= 40:
            status = "⚡ MEDIUM PRIORITY"
        else:
            status = "❄️ LOW PRIORITY"

        return jsonify({
            'score': score,
            'status': status,
            'conversion_probability': f"{score}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
