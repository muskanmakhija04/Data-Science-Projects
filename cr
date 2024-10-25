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
                                                                                                        
                                                                                                        
                                                                                                        
                                                                                                                                                                                                            
##################
Quill:

### 1. **Problem**
   - "At Amazon India, we faced a data security challenge with returned products. 
    The manual process of checking products for physical memory was time-consuming and error-prone, 
    risking user data if not handled properly. Example: camera, usb drive, mobile phone etc.
    It was important to ensure that returned items had their memory securely wiped before resale to protect user data."

### 2. **Solution**
   - "The solution I proposed was automate the identification of products with physical memory from their descriptions,
    enhancing efficiency and reliability."

### 3. **Techniques and Algorithms**
   - "I began by collecting data, including product descriptions and ASIN numbers, 
    and labeled it to indicate the presence of physical memory. 
    After preprocessing the data and creating new features, I explored various classification models. 
    Ultimately, we selected the XGBoost model, achieving an initial accuracy of 80%. 
    After hyperparameter tuning, we improved this to 95%."

### 4. **Results**
   - "The automated solution effectively identified products with physical memory, 
    ensuring secure data wiping before resale. 
    This not only streamlined our operations but also strengthened customer trust."




Smartdata – Medical prescription generator

### 1. **Problem**
   - "At SmartData Enterprises, I worked on generating medical prescriptions based on user inputs such as symptoms, 
    age, weight, and medical conditions. Build a solution so that a patient without even visiting the doctor can 
    get a medical prescription."

### 2. **Solution**
   - "I built a Chatbot that interacts with users in natural language to gather relevant information and generate accurate prescriptions."

### 3. **Techniques and Algorithms**
   - "I used Retrieval-Augmented Generation (RAG) technique. I chunked the provided PDFs—covering common diseases, 
    differential questions, medicine lists, and dosages. We built the knowledge base for our model using these PDFs.
   -The chatbot greets users and asks a series of questions. 
    Based on their responses, it retrieves the common disease name from the PDFs. 
    It then uses differential disease questions to confirm the user’s condition and extracts the corresponding 
    medicine name and dosage options according to the user’s age and weight."

### 4. **Results**
   - "The chatbot generates a complete prescription, including the disease name, medicine name, and dosage form. 
    This automated solution saves time for both users and healthcare providers."

### Conclusion
- "In summary, the Medical Prescription Generator Chatbot leverages advanced retrieval techniques 
    and user interaction to streamline the prescription process, enhancing healthcare accessibility."




Eclerx- marketingOps

### 1. **Problem**
   - "In my current role at Eclerx, we identified that creating marketing campaigns traditionally involves 
multiple people and teams, leading to delays and inefficiencies."

### 2. **Solution**
   - "We are developing a Marketing Campaign Builder using generative AI, 
    enabling users to create marketing campaigns in just minutes."

### 3. **Techniques and Features**
   - "Our solution leverages the database schema to extract marketing KPIs and visualize them through 
    graphs on the user’s dashboard. From these visualizations, we generate actionable insights to assist users 
    in making informed decisions."
   - "Based on the insights provided, the system will recommend a number of campaigns. 
Users can select one that fits their needs, edit it at any time, and customize details such as target persona,
audience, product, and country."
   - "Additionally, the tool will generate tailored email content for the selected marketing campaign."

### 4. **Expected Results**
   - "While the project is still in progress, the goal is to allow users to generate a complete marketing 
    campaign with just a few clicks, significantly streamlining the campaign creation process."



                                                                                                        
                                                                                                        
                                                                                                        
                                                                                                        
