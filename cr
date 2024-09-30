import pandas as pd
import os

# Step 1: Load all Excel files from a directory into DataFrames
def load_datasets(directory):
    datasets = {}
    for file in os.listdir(directory):
        if file.endswith('.xlsx'):
            file_path = os.path.join(directory, file)
            datasets[file.replace('.xlsx', '')] = pd.read_excel(file_path)
    return datasets

# Step 2: Define Keyword to KPI Mapping
kpi_mapping = {
    'sales': ['revenue', 'sales_growth', 'average_order_value'],
    'customer': ['customer_satisfaction', 'nps', 'retention_rate', 'churn_rate'],
    'campaign': ['conversion_rate', 'click_through_rate', 'marketing_roi'],
}

# Step 3: Map KPIs to the correct dataset (this can be customized based on file names)
def map_kpis_to_datasets(datasets):
    kpi_datasets = {
        'revenue': datasets.get('transaction'),
        'sales_growth': datasets.get('transaction'),
        'average_order_value': datasets.get('transaction'),
        'customer_satisfaction': datasets.get('customer_feedback'),
        'nps': datasets.get('customer_feedback'),
        'retention_rate': datasets.get('customer_feedback'),
        'churn_rate': datasets.get('customer_feedback'),
        'conversion_rate': datasets.get('campaign_performance'),
        'click_through_rate': datasets.get('campaign_performance'),
        'marketing_roi': datasets.get('campaign_performance')
    }
    return kpi_datasets

# Step 4: Parse User Input
def parse_objective(user_input):
    user_input = user_input.lower()
    for key, kpis in kpi_mapping.items():
        if key in user_input:
            return kpis
    return []

# Step 5: Query the Relevant KPI from Datasets
def get_kpis_for_objective(user_input, kpi_datasets):
    matched_kpis = parse_objective(user_input)
    if not matched_kpis:
        return "No KPIs matched the input objective."

    result = {}
    for kpi in matched_kpis:
        if kpi in kpi_datasets and kpi_datasets[kpi] is not None:
            df = kpi_datasets[kpi]
            if kpi in df.columns:
                result[kpi] = df[kpi].mean()  # Example: Return mean value of KPI (customize as needed)
            else:
                result[kpi] = f"{kpi} column not found in the dataset."
        else:
            result[kpi] = f"{kpi} dataset not found."

    return result

# Step 6: Example Usage
directory = 'path_to_excel_files'  # Specify the path where all your Excel files are stored
datasets = load_datasets(directory)
kpi_datasets = map_kpis_to_datasets(datasets)

user_input = "I want to improve sales growth"
kpi_result = get_kpis_for_objective(user_input, kpi_datasets)

print("Matched KPIs and their values:")
for kpi, value in kpi_result.items():
    print(f"{kpi}: {value}")
