import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Configuration & Indian Context Data ---

# Date Range: Past 365 days ending Dec 31, 2024
end_date = datetime(2024, 12, 31)
start_date = end_date - timedelta(days=365)
date_range = [start_date + timedelta(days=x) for x in range(365)]
print(start_date, end_date)
# Indian Public Holidays (Approximate for 2023-2024 window)
indian_holidays = [
    "2024-01-01", # New Year
    "2024-01-26", # Republic Day
    "2024-03-25", # Holi
    "2024-04-10", # Eid al-Fitr (Approx)
    "2024-04-14", # Ambedkar Jayanti
    "2024-08-15", # Independence Day
    "2024-10-02", # Gandhi Jayanti
    "2024-10-28", # Diwali (Approx)
    "2024-11-15", # Guru Nanak Jayanti
    "2024-12-25", # Christmas
]

# Helper function to check holiday
def get_holiday_flag(date_obj):
    return 1 if date_obj.strftime("%Y-%m-%d") in indian_holidays else 0

# Helper for gas price with temporal variation throughout 2024
def get_gas_price(date_obj):
    """
    Gas price evolution:
    - Jan 2024: 75.00
    - Feb end: 76.1
    - March: drops to 73.5
    - Dec end: 77.84
    Returns price with small random noise.
    """
    # Convert date to month number (1-12)
    month = date_obj.month
    day = date_obj.day
    
    # Define key prices at specific points
    if month == 1:
        base = 75.00
    elif month == 2:
        # Linear increase from 75.00 to 76.1
        base = 75.00 + (76.1 - 75.00) * (month - 1 + day / 28.0) / 2.0
    elif month == 3:
        # Drop to 73.5 by end of March
        base = 76.1 - (76.1 - 73.5) * (day / 31.0)
    else:
        # Steady increase from 73.5 (April) to 77.84 (Dec end)
        # Months 4-12 = 9 months
        progress = (month - 3 - 1 + day / 31.0) / 9.0
        base = 73.5 + (77.84 - 73.5) * progress
    
    # Add small random noise
    noise = np.random.normal(0, 0.5)
    return round(max(70.0, base + noise), 2)

# Helper for Unemployment Rate (CMIE trends ~7-9%)
def get_unemployment_rate(month):
    # Slight seasonal variation
    base_rate = 7.8
    if month in [4, 5, 6]: # Summer/Job entry season
        base_rate += 0.5
    return round(base_rate + np.random.normal(0, 0.2), 1)

# Synthetic inflation: expenses grow slightly faster than income over time
def get_inflation_multipliers(date_obj):
    """
    Returns (income_multiplier, expense_multiplier) based on time progression.
    Design:
    - Expenses: ~7% annualized -> daily factor ~ (1 + 0.07/365)
    - Income:   ~3% annualized -> daily factor ~ (1 + 0.03/365)
    Multipliers are computed relative to `start_date`.
    """
    days_since_start = (date_obj - start_date).days
    # Clamp non-negative
    days_since_start = max(days_since_start, 0)
    expense_daily = 1 + (0.07 / 365.0)
    income_daily = 1 + (0.03 / 365.0)
    expense_mult = expense_daily ** days_since_start
    income_mult = income_daily ** days_since_start
    return income_mult, expense_mult

def generate_expenses(base, spike_prob=0.03, spike_multiplier=(3, 8), spike_add_range=(1000, 7000)):
    """
    Given a base expense value, return a mostly-normal expense with occasional spikes.
    - spike_prob: probability of a large spike on any given row.
    - spike_multiplier: multiplicative spike applied to base (if base > 0).
    - spike_add_range: additive spike range used when base is very small or choose additive.
    """
    base = float(base or 0.0)
    # Normal jitter +/-20%
    normal = base * random.uniform(0.8, 1.2)
    if random.random() < spike_prob:
        # 60% chance to use multiplicative spike (if base > 0), else additive
        if base > 0 and random.random() < 0.7:
            spike = base * random.uniform(spike_multiplier[0], spike_multiplier[1])
        else:
            spike = base + random.uniform(spike_add_range[0], spike_add_range[1])
        return round(max(spike, 0.0), 0)
    return round(max(normal, 0.0), 0)

# --- Generator Functions ---

def generate_worker_a(dates):
    """
    Worker A: Driver (Uber/Ola/Swiggy).
    High frequency, low variance, peaks on Fri/Sat.
    """
    data = []
    current_target = 35000 # Initial monthly target

    for d in dates:
        row = {}
        row['Date'] = d.strftime("%Y-%m-%d")
        row['Job_Type'] = "Ride-Sharing/Delivery"
        # row['Education_Level'] = "12th Pass"
        row['Job_Categories'] = "Transport"
        
        is_weekend = d.weekday() in [4, 5] # Fri(4), Sat(5)
        holiday = get_holiday_flag(d)
        
        # Hours: Consistent, longer on weekends
        base_hours = 9
        row['Hours_Worked'] = round(base_hours + (2 if is_weekend else 0) + np.random.normal(0, 1), 0)
        
        # Platforms: usually multi-apping
        row['Platform_Count'] = random.choice([2, 3])
        
        # Jobs Completed (High volume)
        # row['Jobs_Completed'] = int(row['Hours_Worked'] * random.uniform(1.5, 2.5))
        
        # Income: Base + Weekend Bonus
        # Approx ₹100-150 per hour equivalent revenue
        daily_income = (row['Hours_Worked'] * 100) + (500 if is_weekend else 0) + (300 if holiday else 0)
        inc_mult, exp_mult = get_inflation_multipliers(d)
        row['Income_Total'] = round(daily_income * inc_mult, 0)
        
        # Expenses: High fuel/maintenance (approx 35% of income)
        gas_price = get_gas_price(d)
        row['Local_Gas_Price'] = gas_price

        if d.day == 1:
            base_exp = row['Income_Total'] * random.uniform(0.2, 0.4) + gas_price + 3000    # Rent
            row['Expenses_Total'] = generate_expenses(base_exp, spike_prob=0.001)
        else:
            base_exp = row['Income_Total'] * random.uniform(0.3, 0.5) + gas_price
            row['Expenses_Total'] = generate_expenses(base_exp, spike_prob=0.04)

        # Apply inflation to expenses
        row['Expenses_Total'] = round(row['Expenses_Total'] * exp_mult, 0)
        
        
        current_target = row['Income_Total'] + (row['Income_Total'] - row['Expenses_Total']) * 0.25 # Update target based on adjusted values

        row['Public_Holiday_Flag'] = holiday
        row['Monthly_Unemployment_Rate'] = get_unemployment_rate(d.month)
        row['Job_Duration_Days'] = 0 # Daily gigs
        
        # Update target monthly
        
        current_target = round(current_target * random.uniform(0.98, 1.1), -2)
        row['Target_Income_Next_Day'] = current_target
        
        data.append(row)
    return pd.DataFrame(data)

def generate_worker_b(dates):
    """
    Worker B: Freelancer (Tech/Design).
    Low frequency, high variance, monthly spikes (invoices), zeroes.
    """
    data = []
    current_target = 80000 

    for d in dates:
        row = {}
        row['Date'] = d.strftime("%Y-%m-%d")
        row['Job_Type'] = "Freelance"
        # row['Education_Level'] = "B.Tech/B.Des"
        row['Job_Categories'] = "Tech Services"
        
        holiday = get_holiday_flag(d)
        row['Public_Holiday_Flag'] = holiday
        row['Local_Gas_Price'] = get_gas_price(d) # Still affects COL
        row['Monthly_Unemployment_Rate'] = get_unemployment_rate(d.month)
        
        # Trigger for payment day (End of month or random project completion)
        is_payment_day = (d.day in [1, 15, 30]) or (random.random() < 0.05)
        
        inc_mult, exp_mult = get_inflation_multipliers(d)
        if is_payment_day:
            row['Income_Total'] = round(random.uniform(15000, 50000) * inc_mult, 2)
            row['Jobs_Completed'] = 1
            row['Job_Duration_Days'] = random.randint(5, 20)
            row['Hours_Worked'] = random.randint(2, 8) # Admin work on payment day
        else:
            row['Income_Total'] = 0
            row['Jobs_Completed'] = 0
            row['Job_Duration_Days'] = 0
            # Working on projects but not getting paid today
            is_working_day = d.weekday() < 6
            row['Hours_Worked'] = random.randint(4, 10) if is_working_day else 0

        # Expenses: Low variable, fixed subscriptions (amortized or spike)
        # Adding a random software subscription cost occasionally
        if d.day == 5:
            base_exp = 5000  # Software subs / invoice-like expense
        else:
            base_exp = random.uniform(500, 2500)  # Coffee/Internet
        row['Expenses_Total'] = generate_expenses(base_exp, spike_prob=0.09)
        row['Expenses_Total'] = round(row['Expenses_Total'] * exp_mult, 2)
            
        row['Platform_Count'] = 1 # Usually Upwork or LinkedIn
        current_target = (row['Income_Total'] - row['Expenses_Total']) * 30 * 0.9 # Update target based on daily income trend
        
        current_target = round(current_target * random.uniform(0.98, 1.1), -2)
        row['Target_Income_Next_Day'] = current_target
        
        data.append(row)
    return pd.DataFrame(data)

def generate_worker_c(dates):
    """
    Worker C: Hybrid (Student/Part-time).
    Random mix of delivery and micro-tasks.
    """
    data = []
    current_target = 15000

    for d in dates:
        row = {}
        row['Date'] = d.strftime("%Y-%m-%d")
        row['Job_Type'] = "Hybrid/Gig-Task"
        row['Education_Level'] = "Undergraduate"
        row['Job_Categories'] = random.choice(["Delivery", "Data Entry", "Event Staff", "None"])
        
        holiday = get_holiday_flag(d)
        row['Public_Holiday_Flag'] = holiday
        row['Local_Gas_Price'] = get_gas_price(d)
        row['Monthly_Unemployment_Rate'] = get_unemployment_rate(d.month)
        
        # Random work pattern
        inc_mult, exp_mult = get_inflation_multipliers(d)
        if row['Job_Categories'] == "None":
            row['Income_Total'] = 0
            row['Expenses_Total'] = generate_expenses(500, spike_prob=0.05) # Food / small spend
            row['Expenses_Total'] = round(row['Expenses_Total'] * exp_mult, 2)
            row['Hours_Worked'] = 0
            row['Jobs_Completed'] = 0
            row['Platform_Count'] = 0
            row['Job_Duration_Days'] = 0
        else:
            row['Hours_Worked'] = round(random.uniform(2, 6), 1)
            row['Platform_Count'] = random.randint(1, 4)
            row['Jobs_Completed'] = random.randint(1, 5)
            row['Job_Duration_Days'] = 1
            
            # Income varies wildly based on task
            if row['Job_Categories'] == "Delivery":
                row['Income_Total'] = row['Hours_Worked'] * 90 * inc_mult
                base_exp = row['Income_Total'] * 0.2 + row['Local_Gas_Price'] # Bike fuel
                row['Expenses_Total'] = generate_expenses(base_exp, spike_prob=0.05)
            elif row['Job_Categories'] == "Data Entry":
                row['Income_Total'] = row['Hours_Worked'] * 150 * inc_mult
                row['Expenses_Total'] = generate_expenses(500, spike_prob=0.05) # Internet
            else: # Event Staff
                row['Income_Total'] = 800 * inc_mult # Fixed shift pay
                row['Expenses_Total'] = generate_expenses(500, spike_prob=0.05) # Transport
            row['Expenses_Total'] = round(row['Expenses_Total'] * exp_mult, 2)
            
        row['Income_Total'] = round(row['Income_Total'], 2)
        row['Expenses_Total'] = round(row['Expenses_Total'], 2)

        current_target = (row['Income_Total'] - row['Expenses_Total']) * 30 * 0.9 # Update target based on daily income trend
        
        current_target = round(current_target * random.uniform(0.98, 1.1), -2)
        row['Target_Income_Next_Day'] = current_target
        
        data.append(row)
    return pd.DataFrame(data)


def compute_income_volatility(df, date_col='Date', income_col='Income_Total', window=7):
    """
    Compute income volatility index as the rolling standard deviation of daily income
    over a specified window (default 7 days for weekly volatility).
    
    Returns a new DataFrame with an added `income_volatility_index` column.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Calculate rolling standard deviation with minimum 1 period to avoid NaN at start
    df['income_volatility_index'] = df[income_col].rolling(window=window, min_periods=1).std().round(2)
    
    return df


def compute_monthly_target(df, date_col='Date'):
    """
    Given a DataFrame with a `Date`, `Income_Total` and `Expenses_Total` columns,
    compute monthly totals and a pragmatic `Target_Income_Next_Month` for each month.

    Formula (heuristic):
    - net = monthly_income - monthly_expenses
    - if net >= 0: target = expenses + 0.5*net + 0.10*expenses  (cover expenses + save half surplus + 10% buffer)
    - if net < 0:  target = expenses + abs(net) + 0.15*expenses (cover expense + make up deficit + 15% buffer)

    The computed values are merged back into the original DataFrame, and the
    function returns a new DataFrame with added columns:
    `Month`, `Monthly_Income`, `Monthly_Expenses`, `Target_Income_Next_Month`.
    """
    # Ensure Date is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['Month'] = df[date_col].dt.to_period('M').astype(str)

    agg = (
        df.groupby('Month')
        .agg(Monthly_Income=('Income_Total', 'sum'), Monthly_Expenses=('Expenses_Total', 'sum'))
        .reset_index()
    )

    def _calc_target(row):
        inc = float(row['Monthly_Income'])
        exp = float(row['Monthly_Expenses'])
        net = inc - exp
        if net >= 0:
            target = exp + net + 0.10 * exp
        else:
            target = exp + abs(net) + 0.15 * exp
        return round(max(target, 0.0), -2)

    agg['Target_Income_Next_Month'] = agg.apply(_calc_target, axis=1)

    # Merge back into original df; every row in the same month gets the same monthly values
    out = df.merge(agg, on='Month', how='left')
    # Keep consistent column order where possible
    return out

# --- Execution ---

df_a = generate_worker_a(date_range)
df_b = generate_worker_b(date_range)
df_c = generate_worker_c(date_range)

# Compute income volatility index (7-day rolling std of daily income)
df_a = compute_income_volatility(df_a)
df_b = compute_income_volatility(df_b)
df_c = compute_income_volatility(df_c)

# Compute monthly totals and Target_Income_Next_Month for each dataset
df_a = compute_monthly_target(df_a)
df_b = compute_monthly_target(df_b)
df_c = compute_monthly_target(df_c)

# Outputting to CSV (will include the new monthly and target columns)
df_a.to_csv('worker_a_driver.csv', index=False)
df_b.to_csv('worker_b_freelancer.csv', index=False)
df_c.to_csv('worker_c_hybrid.csv', index=False)

print("Datasets generated successfully:")
print("1. worker_a_driver.csv")
print("2. worker_b_freelancer.csv")
print("3. worker_c_hybrid.csv")

# Displaying a preview of Worker A (Driver) to verify logic
print("\n--- Preview: Worker A (Driver) ---")
print(df_a.head().to_string())

print("\n--- Preview TAIL: Worker A (Driver) ---")
print(df_a.tail().to_string())

print("\n--- Preview: Worker B (Freelancer) ---")
print(df_b.head(10).to_string())

print("\n--- Preview TAIL: Worker B (Freelancer) ---")
print(df_b.tail(10).to_string())