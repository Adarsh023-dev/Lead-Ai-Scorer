import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'lead_id': range(1, n+1),
    'age': np.random.randint(22, 60, n),
    'annual_income': np.random.randint(300000, 2000000, n),
    'company_size': np.random.choice(['Small', 'Medium', 'Large'], n),
    'industry': np.random.choice(['Tech', 'Finance', 'Retail', 'Healthcare', 'Manufacturing'], n),
    'website_visits': np.random.randint(0, 20, n),
    'emails_opened': np.random.randint(0, 10, n),
    'demo_requested': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'response_time_hours': np.random.randint(1, 72, n),
    'previous_purchase': np.random.choice([0, 1], n, p=[0.6, 0.4]),
})

# Create realistic conversion logic
score = (
    df['website_visits'] * 2 +
    df['emails_opened'] * 3 +
    df['demo_requested'] * 20 +
    df['previous_purchase'] * 15 +
    (df['annual_income'] / 100000) +
    np.where(df['company_size'] == 'Large', 10, 
    np.where(df['company_size'] == 'Medium', 5, 0)) -
    df['response_time_hours'] * 0.2 +
    np.random.normal(0, 5, n)
)

df['converted'] = (score > score.median()).astype(int)

df.to_csv('leads_data.csv', index=False)
print("✅ Dataset created! Shape:", df.shape)
print(df.head())