import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace with actual file path or dataframe)
# Assume df contains 'Month', 'customer_group', 'ledger', 'vertical', 'consolidated_revenue'
df = pd.read_csv('consolidated_revenue_data.csv')

# Ensure 'Month' column is datetime
df['Month'] = pd.to_datetime(df['Month'])

# Sort by time to ensure temporal order
df = df.sort_values('Month')

# Step 1: Target encoding for categorical features (Customer Group, Ledger, Vertical)
cat_features = ['customer_group', 'ledger', 'vertical']
for col in cat_features:
    # Compute mean target (consolidated revenue) for each category
    target_mean = df.groupby(col)['consolidated_revenue'].mean()
    df[col + '_encoded'] = df[col].map(target_mean)

# Step 2: Feature engineering - extract time-based features
df['month'] = df['Month'].dt.month
df['quarter'] = df['Month'].dt.quarter
df['year'] = df['Month'].dt.year

# Step 3: Prepare data for LightGBM model
# Define features and target
features = ['month', 'quarter', 'year'] + [col + '_encoded' for col in cat_features]
X = df[features]
y = df['consolidated_revenue']

# Step 4: Train-test split using time-based cross-validation (TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=3)

# Step 5: Train LightGBM model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_data_in_leaf': 20
}

models = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Convert data to LightGBM dataset format
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train model
    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], early_stopping_rounds=50)
    models.append(model)

# Step 6: Prediction (Example for future month prediction)
future_months = pd.date_range(start='2024-03-01', periods=3, freq='MS')
future_df = pd.DataFrame({
    'Month': future_months,
    'customer_group_encoded': 0,  # Replace with actual target-encoded values
    'ledger_encoded': 0,
    'vertical_encoded': 0,
    'month': future_months.month,
    'quarter': future_months.quarter,
    'year': future_months.year
})

# Predict using the last model trained
future_X = future_df[features]
future_pred = models[-1].predict(future_X)
print(future_pred)



 #graph
 import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already split the data and trained the model

# Make predictions on the test set (use your actual test data)
y_pred = models[-1].predict(X_test)

# Create a DataFrame to compare actual vs predicted
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Plot using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(results_df['Actual'].reset_index(drop=True), label='Actual', color='blue', marker='o', linestyle='-')
plt.plot(results_df['Predicted'].reset_index(drop=True), label='Predicted', color='orange', marker='x', linestyle='--')
plt.title('Predicted vs Actual Consolidated Revenue')
plt.xlabel('Index')
plt.ylabel('Consolidated Revenue')
plt.legend()
plt.grid()
plt.show()
                                                                                                            
                                                                                                            
#dummmy
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load your dataset (replace with actual file path or dataframe)
df = pd.read_csv('consolidated_revenue_data.csv')

# Ensure 'Month' column is datetime
df['Month'] = pd.to_datetime(df['Month'])

# Sort by time to ensure temporal order
df = df.sort_values('Month')

# Step 1: Label Encoding for Customer Group
label_encoder = LabelEncoder()
df['customer_group_encoded'] = label_encoder.fit_transform(df['customer_group'])

# Step 2: One-Hot Encoding for Ledger and Vertical
# Create dummy variables for 'ledger' and 'vertical' (One-Hot Encoding)
one_hot_features = pd.get_dummies(df[['ledger', 'vertical']], drop_first=True)

# Combine one-hot encoded features with the original dataset
df = pd.concat([df, one_hot_features], axis=1)

# Drop the original categorical columns after encoding
df.drop(['customer_group', 'ledger', 'vertical'], axis=1, inplace=True)

# Step 3: Feature engineering - extract time-based features
df['month'] = df['Month'].dt.month
df['quarter'] = df['Month'].dt.quarter
df['year'] = df['Month'].dt.year

# Step 4: Prepare data for LightGBM model
# Define features and target
features = ['customer_group_encoded', 'month', 'quarter', 'year'] + list(one_hot_features.columns)
X = df[features]
y = df['consolidated_revenue']

# Step 5: Train-test split using time-based cross-validation (TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=3)

# Step 6: Train LightGBM model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_data_in_leaf': 20
}

models = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Convert data to LightGBM dataset format
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train model
    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], early_stopping_rounds=50)
    models.append(model)

# Step 7: Prediction and evaluation
y_pred = models[-1].predict(X_test)

# Step 8: Plot Predicted vs Actual
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame to compare actual vs predicted
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Plot using seaborn or matplotlib
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, palette="tab10", linewidth=2.5)
plt.title('Predicted vs Actual Consolidated Revenue')
plt.xlabel('Index')
plt.ylabel('Consolidated Revenue')
plt.legend(['Actual', 'Predicted'])
plt.show()
                                                                                                           
