import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Get the working directory of the ml_utility.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)

# Step 1: Read the data
def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        return df

# Step 2: Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    # Encode target column
    target_mapping = {value: idx for idx, value in enumerate(df[target_column].unique())}
    df[target_column] = df[target_column].map(target_mapping)

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if there are numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Choose the appropriate scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()

    # Initialize preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', scaler)
            ]), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder())
            ]), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    return X_train, X_test, y_train, y_test, preprocessor, target_mapping

# Step 3: Train the model
def train_model(X_train, y_train, preprocessor, model_to_be_trained, model_name):
    # Create a pipeline with preprocessor and classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_to_be_trained)
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    model_file_path = f"{parent_dir}/trainedmodel/{model_name}.pkl"
    with open(model_file_path, 'wb') as file:
        pickle.dump(model, file)

    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
