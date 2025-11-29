import pandas as pd
import numpy as np

df = pd.read_csv('worker_c_hybrid.csv')

def burnout_risk(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate Burnout Risk Score
    df['burnout_risk_score'] = (df['Hours_Worked'] / df['Income_Total']) * 1000
    print(df[['Hours_Worked', 'Income_Total', 'burnout_risk_score']].head(10))
    return df

def extract_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Get New Features Based on Existing Columns
    dataframe['income_rolling_avg_3m'] = round(dataframe['Income_Total'].rolling(window=90).mean(), -1)
    dataframe['expense_to_income_ratio'] = round(dataframe['Expenses_Total'] / dataframe['Income_Total'], 2)
    dataframe['net_Income'] = round(dataframe['Income_Total'] - dataframe['Expenses_Total'], -1)
    dataframe['savings'] = dataframe['Monthly_Income'] - dataframe['Monthly_Expenses']
    
    dataframe.fillna(0, inplace=True)
    mu = dataframe['Income_Total'].mean()
    sigma = dataframe['Income_Total'].std(ddof=0)  # population std; use ddof=1 for sample
    dataframe['y'] = (dataframe['Income_Total'] - mu) / sigma

    return dataframe


new_df = extract_features(df)
mu = new_df['Income_Total'].mean()
sigma = new_df['Income_Total'].std(ddof=0)

del df
new_df = new_df.rename(columns= {
    'Date': 'ds'
})


new_df.to_csv('features_c.csv', index=False)