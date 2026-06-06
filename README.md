Lead AI Scorer — Live ML Web App

**Live Demo:** https://lead-ai-scorer.onrender.com/  
**Tech:** Python · Scikit-learn · Random Forest · HTML/CSS/JS · Render

## 🚀 What it does
Scores any sales lead 0–100 based on conversion probability using Machine Learning.

## ✅ Features
- Single lead scoring with visual gauge
- Bulk CSV upload — score 1000 leads at once  
- Google Sheets integration — paste sheet URL, get scores instantly
- Download results as CSV
- 🔥 Hot / ⚡ Warm / ❄️ Cold priority classification

## 🧠 Model
- Algorithm: Random Forest Classifier
- Training data: 1,000 synthetic lead records
- Accuracy: 85.5%
- Features: website visits, emails opened, demo requested, 
  annual income, response time, previous purchase, 
  age, company size, industry

## 📁 Files
| File | Description |
|------|-------------|
| `index.html` | Full frontend — scoring UI |
| `generate_data.py` | Generates training dataset |
| `model.py` | Trains Random Forest model |
| `leads_data.csv` | Generated training data |

## 🚀 Run Locally
```bash
pip install pandas scikit-learn matplotlib seaborn
python generate_data.py
python model.py
```
Open `index.html` in browser or use Live Server in VS Code.
