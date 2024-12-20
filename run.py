import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Define the label encoder function
def label_encoder(df):
    label_encoder = LabelEncoder()
    df.drop(columns='CustomerID', inplace=True, errors='ignore')  
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    columns_to_encode = ['gender', 'subscription_type', 'contract_length']
    for col in columns_to_encode:
        if col in df.columns:  
            df[col] = label_encoder.fit_transform(df[col])
    
    discrete_col = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'last_interaction', 'churn']
    for col in discrete_col:
        if col in df.columns: 
            df[col] = df[col].astype(int)
    return df

def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_churn(model, df, threshold=0.5):
    df = label_encoder(df)
    
    X = df.drop(columns=['churn', 'predicted_churn'], errors='ignore')
    
    # Get predicted probabilities
    probabilities = model.predict_proba(X)[:, 1] 
    
   
    print("Predicted probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"Row {i + 1}: {prob}")
    

    predictions = (probabilities >= threshold).astype(int)
    df['predicted_churn'] = predictions
    
    return df

if __name__ == "__main__":
    model = load_model('neural_net.pkl')
    
    new_data_path = "sample.csv"  
    new_data = pd.read_csv(new_data_path)
    
   
    threshold = 0.5 
    result = predict_churn(model, new_data, threshold=threshold)
    

    new_data['predicted_churn'] = result['predicted_churn']
    
 
    output_file = "predicted_churn_results.csv"  
    new_data.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}' with threshold {threshold}.")
