import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import GradientBoostingClassifier


# Load the dataset
df = pd.read_csv("E:\\projects\\sem 5\\ml\\final project\\Impact_of_Remote_Work_on_Mental_Health.csv")


# Display basic information about the dataset
print("Dataset Overview")
print(df.info())
print("\nFirst 5 rows of the dataset")
print(df.head())

# Check for missing values
print("\nMissing Values in Each Column")
print(df.isnull().sum())

# Basic statistics of the dataset
print("\nDescriptive Statistics")
print(df.describe())


# Print columns for debugging
print("Columns in DataFrame:", df.columns)

# Check for missing values in the target variable
if 'Mental_Health_Condition' not in df.columns:
    raise ValueError("Target variable 'Mental_Health_Condition' not found in DataFrame.")

# Drop Employee_ID if it exists
df = df.drop(columns=['Employee_ID'], errors='ignore')

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le  # Store the encoder for later use


# Display correlation matrix for all features (numerical and encoded categorical)
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Including All Features')
plt.show()


# Count the number of 'Prefer not to say' entries
prefer_not_to_say_count = df['Gender'].value_counts().get('Prefer not to say', 0)

# Replace 'Prefer not to say' with sampled values if any exist
if prefer_not_to_say_count > 0:
    males = df[df['Gender'] == 'Male']
    females = df[df['Gender'] == 'Female']
    non_binaries = df[df['Gender'] == 'Non-Binary']

    # Calculate how many of each gender to sample
    count_per_gender = prefer_not_to_say_count // 3

    # Sample from each gender to replace 'Prefer not to say'
    sampled_males = males.sample(count_per_gender, replace=True) if len(males) > 0 else pd.DataFrame(columns=df.columns)
    sampled_females = females.sample(count_per_gender, replace=True) if len(females) > 0 else pd.DataFrame(columns=df.columns)
    sampled_non_binaries = non_binaries.sample(count_per_gender, replace=True) if len(non_binaries) > 0 else pd.DataFrame(columns=df.columns)

    # Combine the samples into a single DataFrame
    replacement_df = pd.concat([sampled_males, sampled_females, sampled_non_binaries], ignore_index=True)

    # If the replacement DataFrame doesn't have enough rows, resample to match the required length
    replacement_df = replacement_df.sample(prefer_not_to_say_count, replace=True).reset_index(drop=True)

    # Replace 'Prefer not to say' in the original DataFrame
    df.loc[df['Gender'] == 'Prefer not to say', 'Gender'] = replacement_df['Gender'].values

# Feature selection using SelectKBest with chi-squared score
X = df.drop(columns=['Mental_Health_Condition'])
y = df['Mental_Health_Condition']

k_best = SelectKBest(score_func=chi2, k=10)  # Select top 10 features
k_best.fit(X, y)

# Get the indices of the selected features
selected_indices = k_best.get_support(indices=True)
top_k_features = X.columns[selected_indices]

# Print the names of the top 10 features
print("\nTop 10 Features:")
print(top_k_features.tolist())

# Create a bar plot of feature scores
feature_scores = k_best.scores_[selected_indices]
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_scores, y=top_k_features, palette='viridis')
plt.title('Top 10 Features based on Chi-Squared Scores')
plt.xlabel('Chi-Squared Score')
plt.ylabel('Features')
plt.show()

# Selecting relevant features
features = ['Gender', 'Job_Role', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings', 
            'Social_Isolation_Rating', 'Company_Support_for_Remote_Work', 'Physical_Activity', 'Sleep_Quality', 'Region']
X = df[features]
y = df['Mental_Health_Condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate and print model metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return accuracy

# Dictionary to store accuracies and models
accuracies = {}

# Train and evaluate Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
print("\nRandom Forest Metrics Before Tuning:")
rf_accuracy_before = evaluate_model(clf_rf, X_test, y_test)
accuracies['Random Forest Before Tuning'] = rf_accuracy_before

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(clf_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("\nRandom Forest Metrics After Tuning:")
rf_accuracy_after = evaluate_model(best_rf, X_test, y_test)
accuracies['Random Forest After Tuning'] = rf_accuracy_after

# Train and evaluate XGBoost Classifier
clf_xgb = xgb.XGBClassifier(random_state=42)
clf_xgb.fit(X_train, y_train)
print("\nXGBoost Metrics Before Tuning:")
xgb_accuracy_before = evaluate_model(clf_xgb, X_test, y_test)
accuracies['XGBoost Before Tuning'] = xgb_accuracy_before

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1]
}

grid_xgb = GridSearchCV(clf_xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print("\nXGBoost Metrics After Tuning:")
xgb_accuracy_after = evaluate_model(best_xgb, X_test, y_test)
accuracies['XGBoost After Tuning'] = xgb_accuracy_after


# Train and evaluate Gradient Boosting Classifier
clf_gb = GradientBoostingClassifier(random_state=42)
clf_gb.fit(X_train, y_train)
print("\nGradient Boosting Metrics Before Tuning:")
gb_accuracy_before = evaluate_model(clf_gb, X_test, y_test)
accuracies['Gradient Boosting Before Tuning'] = gb_accuracy_before


# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_gb = GridSearchCV(clf_gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_

print("\nGradient Boosting Metrics After Tuning:")
gb_accuracy_after = evaluate_model(best_gb, X_test, y_test)
accuracies['Gradient Boosting After Tuning'] = gb_accuracy_after



# Train and evaluate Logistic Regression model
clf_logreg = LogisticRegression(random_state=42, max_iter=1000)
clf_logreg.fit(X_train, y_train)
print("\nLogistic Regression Metrics Before Tuning:")
logreg_accuracy_before = evaluate_model(clf_logreg, X_test, y_test)
accuracies['Logistic Regression Before Tuning'] = logreg_accuracy_before

# Hyperparameter tuning for Logistic Regression
param_grid_logreg = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}

grid_logreg = GridSearchCV(clf_logreg, param_grid_logreg, cv=5, scoring='accuracy', n_jobs=-1)
grid_logreg.fit(X_train, y_train)

best_logreg = grid_logreg.best_estimator_
print("\nLogistic Regression Metrics After Tuning:")
logreg_accuracy_after = evaluate_model(best_logreg, X_test, y_test)
accuracies['Logistic Regression After Tuning'] = logreg_accuracy_after

# Summary of accuracies before and after tuning
print("\nAccuracies Summary:")
for model_name in accuracies:
    print(f"{model_name}: {accuracies[model_name]}")

# Print best accuracies of all models before and after tuning
best_model_name = max(accuracies.keys(), key=lambda k: accuracies[k])
best_accuracy = accuracies[best_model_name]
print(f"\nBest Model Overall: {best_model_name} with accuracy {best_accuracy}")

# GUI for prediction using Tkinter
root = tk.Tk()
root.title("Mental Health Prediction")

def make_prediction():
    try:
        # Get input values from GUI fields
        gender = gender_entry.get().strip()
        job_role = job_role_entry.get().strip()
        years_of_experience = float(years_of_experience_entry.get())
        hours_worked_per_week = float(hours_worked_per_week_entry.get())
        number_of_virtual_meetings = int(number_of_virtual_meetings_entry.get())
        social_isolation_rating = float(social_isolation_rating_entry.get())
        company_support_for_remote_work = float(company_support_for_remote_work_entry.get())
        physical_activity = physical_activity_entry.get().strip()
        sleep_quality = sleep_quality_entry.get().strip()
        region = region_entry.get().strip()

        # Prepare input for prediction
        input_data = np.array([[
            label_encoders['Gender'].transform([gender])[0],
            label_encoders['Job_Role'].transform([job_role])[0],
            years_of_experience,
            hours_worked_per_week,
            number_of_virtual_meetings,
            social_isolation_rating,
            company_support_for_remote_work,
            label_encoders['Physical_Activity'].transform([physical_activity])[0],
            label_encoders['Sleep_Quality'].transform([sleep_quality])[0],
            label_encoders['Region'].transform([region])[0]
        ]])

        # Make predictions using the best model selected earlier
        prediction_result = best_logreg.predict(input_data)
        prediction_label_inverse_encoded_value = label_encoders['Mental_Health_Condition'].inverse_transform(prediction_result)[0]

        messagebox.showinfo("Prediction Result", f"Mental Health Condition: {prediction_label_inverse_encoded_value}")

    except Exception as e:
        messagebox.showerror("Input Error", f"Error occurred: {str(e)}")

# GUI Layout
tk.Label(root, text="Gender (Male/Female/Non-Binary)").grid(row=0, column=0)
gender_entry = tk.Entry(root)
gender_entry.grid(row=0, column=1)

tk.Label(root, text="Job Role").grid(row=1, column=0)
job_role_entry = tk.Entry(root)
job_role_entry.grid(row=1, column=1)

tk.Label(root, text="Years of Experience").grid(row=2, column=0)
years_of_experience_entry = tk.Entry(root)
years_of_experience_entry.grid(row=2, column=1)

tk.Label(root, text="Hours Worked Per Week").grid(row=3, column=0)
hours_worked_per_week_entry = tk.Entry(root)
hours_worked_per_week_entry.grid(row=3, column=1)

tk.Label(root, text="Number of Virtual Meetings").grid(row=4, column=0)
number_of_virtual_meetings_entry = tk.Entry(root)
number_of_virtual_meetings_entry.grid(row=4, column=1)

tk.Label(root, text="Social Isolation Rating").grid(row=5, column=0)
social_isolation_rating_entry = tk.Entry(root)
social_isolation_rating_entry.grid(row=5, column=1)

tk.Label(root, text="Company Support for Remote Work").grid(row=6, column=0)
company_support_for_remote_work_entry = tk.Entry(root)
company_support_for_remote_work_entry.grid(row=6, column=1)

tk.Label(root, text="Physical Activity Rating").grid(row=7, column=0)
physical_activity_entry = tk.Entry(root)
physical_activity_entry.grid(row=7, column=1)

tk.Label(root, text="Sleep Quality Rating").grid(row=8, column=0)
sleep_quality_entry = tk.Entry(root)
sleep_quality_entry.grid(row=8, column=1)

tk.Label(root, text="Region").grid(row=9, column=0)
region_entry = tk.Entry(root)
region_entry.grid(row=9, column=1)

predict_button = tk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=10, column=0, columnspan=2)

root.mainloop()
