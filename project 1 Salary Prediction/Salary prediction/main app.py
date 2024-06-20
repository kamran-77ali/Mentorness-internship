import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("Salary Prediction of Data Professions.csv")

# Handle missing values
df.fillna({"LAST NAME": "Unknown",
           "AGE": df["AGE"].mean(),
           "LEAVES USED": 0,
           "LEAVES REMAINING": df["LEAVES REMAINING"].median(),
           "RATINGS": df["RATINGS"].mode()[0],
           "UNIT": "Unknown",
           "DESIGNATION": "Unknown",
           "PAST EXP": 0}, inplace=True)
df["DOJ"].ffill(inplace=True)
df["CURRENT DATE"] = pd.to_datetime("now")

# Additional feature engineering
df["DOJ"] = pd.to_datetime(df["DOJ"])
current_date = datetime.now()
df["Years of services"] = (current_date - df["DOJ"]).dt.days / 365.25
df["AGE at Joining"] = df["AGE"] - (current_date - df["DOJ"]).dt.days / 365.25

# Feature Engineering for new features
df['JOB_HOPPING_FREQUENCY'] = np.random.randint(0, 4, len(df))
df['HIERARCHY_LEVEL'] = np.random.choice(['Entry', 'Mid', 'Senior'], len(df))
df['PERFORMANCE_TREND'] = np.random.uniform(-1, 1, len(df))
df['TOP_PERFORMER'] = np.where(df['RATINGS'] >= df['RATINGS'].quantile(0.75), 1, 0)
df['LEAVE_UTILIZATION_RATE'] = df['LEAVES USED'] / (df['LEAVES USED'] + df['LEAVES REMAINING'])
bins = [20, 30, 40, 50, np.inf]
labels = ['20-30', '30-40', '40-50', '50+']
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels)

# Separating features and target variable
X = df.drop(columns=['SALARY'])
y = df['SALARY']

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create preprocessing and modeling pipeline
def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Define models
models = {
    'Linear Regression': create_pipeline(LinearRegression()),
    'Decision Tree': create_pipeline(DecisionTreeRegressor(random_state=42)),
    'Random Forest': create_pipeline(RandomForestRegressor(random_state=42)),
    'Gradient Boosting': create_pipeline(GradientBoostingRegressor(random_state=42))
}

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
evaluation_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    evaluation_results.append({
        'Model': name,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(evaluation_results)

# Identify the model with the best performance based on R2 score
best_model_name = results_df.loc[results_df['R2'].idxmax()]['Model']
best_model_pipeline = models[best_model_name]

# Save the best model to disk
joblib.dump(best_model_pipeline, 'best_model.pkl')

# Streamlit App
st.title("Salary Prediction Analysis for Data Professions")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Data Visualizations", "Model Evaluation", "Predict Salary"])

# Overview section
if section == "Overview":
    st.header("Dataset Overview")
    st.write(df.head())
    
    st.header("Statistical Summary")
    st.write(df.describe())
    
    st.header("Missing Values")
    st.write(df.isnull().sum())

# Data Visualizations section
if section == "Data Visualizations":
    st.header("Data Visualizations")
    
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["AGE"], kde=True, bins=10, ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Leaves Used vs Leaves Remaining")
    fig, ax = plt.subplots()
    sns.histplot(df["LEAVES USED"], kde=True, bins=10, color="blue", label="LEAVES USED", ax=ax)
    sns.histplot(df["LEAVES REMAINING"], kde=True, bins=10, color="red", label="LEAVES REMAINING", ax=ax)
    ax.set_title("Leaves Used vs Leaves Remaining")
    ax.set_xlabel("Number of Leaves")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="RATINGS", data=df, ax=ax)
    ax.set_title("Ratings Distribution")
    ax.set_xlabel("Ratings")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Sex Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="SEX", data=df, ax=ax)
    ax.set_title("Sex Distribution")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Average Age by Sex")
    fig, ax = plt.subplots()
    sns.barplot(x="SEX", y="AGE", data=df, ax=ax)
    ax.set_title("Average Age by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Average Age")
    st.pyplot(fig)

    st.subheader("Sex Distribution Pie Chart")
    fig, ax = plt.subplots()
    df['SEX'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['blue', 'red'], ax=ax)
    ax.set_title('Sex Distribution')
    ax.set_ylabel('')
    st.pyplot(fig)

    st.subheader("Age Distribution by Sex")
    fig, ax = plt.subplots()
    sns.boxplot(x='SEX', y='AGE', data=df, ax=ax)
    ax.set_title('Age Distribution by Sex')
    ax.set_xlabel('Sex')
    ax.set_ylabel('Age')
    st.pyplot(fig)

# Model Evaluation section
if section == "Model Evaluation":
    st.header("Model Evaluation Results")
    st.write(results_df)

# Predict Salary section
if section == "Predict Salary":
    st.header("Predict Salary")
    
    # Load the trained model
    model = joblib.load("best_model.pkl")
    
    # Collect user input
    st.subheader("Enter the details:")
    first_name = st.text_input("First Name", key="first_name_input")
    last_name = st.text_input("Last Name", "Unknown", key="last_name_input")
    sex = st.selectbox("Sex", ["Male", "Female"], key="sex_select")
    age = st.slider("Age", 20, 70, 30, key="age_slider")
    doj = st.date_input("Date of Joining", key="doj_date_input")
    leaves_used = st.slider("Leaves Used", 0, 30, 0, key="leaves_used_slider")
    leaves_remaining = st.slider("Leaves Remaining", 0, 30, 0, key="leaves_remaining_slider")
    ratings = st.slider("Ratings", 1, 5, 3, key="ratings_slider")
    unit = st.text_input("Unit", "Unknown", key="unit_input")
    designation = st.text_input("Designation", "Unknown", key="designation_input")
    past_exp = st.number_input("Past Experience (years)", 0, 50, 0, key="past_exp_input")

    # Additional feature calculations
    years_of_services = (current_date - pd.to_datetime(doj)).days / 365.25
    age_at_joining = age - years_of_services
    job_hopping_frequency = np.random.randint(0, 4)
    hierarchy_level = np.random.choice(['Entry', 'Mid', 'Senior'])
    performance_trend = np.random.uniform(-1, 1)
    top_performer = 1 if ratings >= df['RATINGS'].quantile(0.75) else 0
    leave_utilization_rate = leaves_used / (leaves_used + leaves_remaining) if (leaves_used + leaves_remaining) > 0 else 0

    # Create a DataFrame with the input data
    user_input = pd.DataFrame({
        'FIRST NAME': [first_name],
        'LAST NAME': [last_name],
        'SEX': [sex],
        'AGE': [age],
        'DOJ': [doj],
        'LEAVES USED': [leaves_used],
        'LEAVES REMAINING': [leaves_remaining],
        'RATINGS': [ratings],
        'UNIT': [unit],
        'DESIGNATION': [designation],
        'PAST EXP': [past_exp],
        'Years of services': [years_of_services],
        'AGE at Joining': [age_at_joining],
        'JOB_HOPPING_FREQUENCY': [job_hopping_frequency],
        'HIERARCHY_LEVEL': [hierarchy_level],
        'PERFORMANCE_TREND': [performance_trend],
        'TOP_PERFORMER': [top_performer],
        'LEAVE_UTILIZATION_RATE': [leave_utilization_rate]
    })

    # Predict the salary using the best model
    if st.button('Predict Salary', key="predict_button"):
        salary_prediction = model.predict(user_input)
        st.subheader(f"Predicted Salary: ${salary_prediction[0]:,.2f}")
