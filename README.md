# Lead AI Scorer

A deployed machine-learning application that converts sales-lead attributes
into a 0-100 conversion score. The project demonstrates a complete workflow
from synthetic data generation and model evaluation to a Flask API and browser
interface.

[View live application](https://lead-ai-scorer.onrender.com/)

![Lead AI Scorer interface](Screenshot%202026-06-06%20152827.png)

## Business Use Case

Sales teams often have more leads than they can contact immediately. A
consistent score can help prioritise follow-up, while the final decision remains
with the sales team.

## Features

- Scores an individual lead from 0 to 100.
- Groups results into high, medium, and low priority.
- Exposes a Flask `/score` endpoint for browser-based predictions.
- Includes a responsive HTML, CSS, and JavaScript interface.
- Deploys with Gunicorn on Render.

## Model

| Item | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Development data | 1,000 reproducible synthetic records |
| Reported test accuracy | 85.5% |
| Target | Lead conversion |
| Inputs | Engagement, income, response time, purchase history, company size, industry, and age |

The synthetic target is generated from predefined business rules plus noise.
The reported accuracy therefore measures performance on this demonstration
dataset and should not be interpreted as validated real-world performance.

## Repository Structure

| File | Description |
|---|---|
| `app.py` | Flask application and scoring API |
| `index.html` | Browser-based scoring interface |
| `generate_data.py` | Generates training dataset |
| `model.py` | Trains and evaluates the models |
| `leads_data.csv` | Reproducible synthetic training data |
| `lead_scorer_results.png` | Feature importance and confusion matrix |
| `render.yaml` | Render deployment configuration |

## Run Locally

```bash
pip install -r requirements.txt
python generate_data.py
python model.py
python app.py
```

Open `http://localhost:5000`.

## Skills Demonstrated

`Python` `Flask` `scikit-learn` `Pandas` `Random Forest` `REST API`
`Model Deployment`

## Next Validation Step

Retrain the pipeline on permissioned CRM outcomes, compare precision and recall
against a baseline, calibrate probabilities, and monitor drift before business
use.

## Author

Adarsh Dubey - Data Analyst
