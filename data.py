from faker import Faker
import numpy as np
import pandas as pd
import random

fake = Faker()
n = 1000

def generate_data(n):
    data = []
    for _ in range(n):
        age = random.randint(18, 55)
        monthly_income = random.randint(8000, 60000)
        on_time_utility_payments = np.round(np.random.uniform(0.5, 1.0), 2)  # %
        job_stability = random.randint(0, 10)  # years in current job
        social_score = np.round(np.random.uniform(0, 1), 2)  # 0 = inactive, 1 = very active
        ecommerce_monthly_spend = random.randint(0, 20000)
        phone_usage_score = np.round(np.random.uniform(0.2, 1.0), 2)
        default = 1 if (monthly_income < 12000 and on_time_utility_payments < 0.7) else 0

        data.append([
            age, monthly_income, on_time_utility_payments, job_stability,
            social_score, ecommerce_monthly_spend, phone_usage_score, default
        ])

    df = pd.DataFrame(data, columns=[
        'age', 'monthly_income', 'on_time_utility_payments', 'job_stability',
        'social_score', 'ecommerce_monthly_spend', 'phone_usage_score', 'default'
    ])
    return df

df = generate_data(n)
df.to_csv("simulated_credit_risk_data.csv", index=False)
