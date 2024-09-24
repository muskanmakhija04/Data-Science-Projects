import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Assuming your dataframe is named df
# df has columns: 'date', 'customer_group', 'ledger', 'vertical', 'consolidated_revenue'
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
# Group data by month, customer group, ledger, and vertical
df['month'] = df['date'].dt.to_period('M')
grouped_df = df.groupby(['month', 'customer_group', 'ledger', 'vertical']).agg({
    'consolidated_revenue': 'sum'
}).reset_index()

# Label Encoding for categorical features
label_encoders = {}
for col in ['customer_group', 'ledger', 'vertical']:
    le = LabelEncoder()
    grouped_df[col] = le.fit_transform(grouped_df[col])
    label_encoders[col] = le

# Create lag features for past 12 months to capture full 1-year trend
for lag in range(1, 13):
    grouped_df[f'revenue_lag_{lag}'] = grouped_df.groupby(['customer_group', 'ledger', 'vertical'])['consolidated_revenue'].shift(lag)

# Drop NA values caused by lagging
grouped_df.dropna(inplace=True)

# Split into train and test sets (train on data before 2024)
train_df = grouped_df[grouped_df['month'] < '2024-01']
test_df = grouped_df[grouped_df['month'] >= '2024-01']

X_train = train_df.drop(['consolidated_revenue', 'month'], axis=1)
y_train = train_df['consolidated_revenue']
X_test = test_df.drop(['consolidated_revenue', 'month'], axis=1)
y_test = test_df['consolidated_revenue']

# Train LightGBM model
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=['customer_group', 'ledger', 'vertical'])
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=['customer_group', 'ledger', 'vertical'])

params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=10)

# Predict for the next quarter
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Forecast next quarter (3 months) step by step
last_known = grouped_df[grouped_df['month'] == '2023-12'].copy()

for i in range(1, 4):  # 3 months forecast
    # Prepare next month’s input
    next_month = last_known.copy()
    next_month['month'] = last_known['month'] + pd.offsets.MonthEnd(i)
    
    # Shift the lag features for the next forecast
    for lag in range(1, 12):  # Shift previous lag values
        next_month[f'revenue_lag_{lag}'] = last_known[f'revenue_lag_{lag+1}']
    
    next_month['revenue_lag_12'] = last_known['consolidated_revenue']  # Last month is now 12-month lag
    
    next_month.drop(['consolidated_revenue'], axis=1, inplace=True)
    
    # Predict the revenue for the next month
    next_revenue = model.predict(next_month.drop(['month'], axis=1))
    print(f'Forecasted revenue for {next_month["month"].iloc[0]}: {next_revenue[0]}')
    
    # Update last_known with the predicted revenue
    last_known['consolidated_revenue'] = next_revenue
