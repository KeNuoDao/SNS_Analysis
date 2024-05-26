import pandas as pd
import statsmodels.api as sm

# Define the data
data = {
    'Age Group': [17, 26, 23, 28, 30, 56, 19, 17, 45, 32, 22, 16, 26, 47, 23, 26, 51, 24, 28, 15, 19, 27, 51, 19, 27, 48, 15, 19, 19, 26, 24, 61, 19, 27, 15, 18, 25, 55, 19, 22],
    'Gender': [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    'Education Level': [0, 2, 2, 3, 3, 4, 1, 0, 3, 2, 2, 0, 3, 4, 2, 2, 3, 2, 3, 0, 2, 3, 4, 2, 3, 4, 0, 2, 1, 3, 3, 4, 2, 3, 0, 2, 3, 4, 2, 3],
    'Income Level': [1, 2, 2, 3, 4, 2, 2, 1, 4, 3, 2, 1, 3, 2, 2, 3, 3, 2, 4, 1, 2, 3, 2, 2, 3, 3, 1, 2, 2, 3, 4, 2, 2, 3, 1, 2, 3, 2, 2, 3],
    'Relatedness Satisfaction': [6, 4, 5, 3, 4, 2, 7, 5, 3, 6, 4, 6, 5, 3, 6, 4, 3, 5, 4, 5, 6, 4, 3, 7, 5, 2, 6, 4, 5, 3, 4, 2, 6, 4, 5, 6, 4, 3, 7, 5],
    'Self-Presentation Satisfaction': [5, 4, 6, 3, 4, 2, 6, 4, 3, 5, 5, 6, 5, 2, 7, 3, 3, 5, 4, 4, 6, 5, 3, 7, 4, 2, 6, 5, 6, 4, 5, 3, 6, 5, 5, 6, 4, 3, 6, 5],
    'SNS Addiction': [6, 5, 5, 4, 4, 2, 7, 5, 3, 6, 5, 7, 5, 3, 6, 4, 3, 6, 5, 5, 6, 5, 3, 7, 5, 2, 6, 5, 6, 4, 4, 2, 6, 5, 5, 6, 5, 3, 7, 5]
}

df = pd.DataFrame(data)

# Define the independent variables (X)
X = df[['Age Group', 'Gender', 'Education Level', 'Income Level']]

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the linear regression models for each dependent variable
model_relatedness = sm.OLS(df['Relatedness Satisfaction'], X).fit()
model_self_presentation = sm.OLS(df['Self-Presentation Satisfaction'], X).fit()
model_sns_addiction = sm.OLS(df['SNS Addiction'], X).fit()

# Print the summaries of the regressions
print("Relatedness Satisfaction Model Summary:")
print(model_relatedness.summary())
print("\nSelf-Presentation Satisfaction Model Summary:")
print(model_self_presentation.summary())
print("\nSNS Addiction Model Summary:")
print(model_sns_addiction.summary())

# Function to predict all three factors based on input data
def predict_factors(age_group, gender, education_level, income_level):
    input_data = pd.DataFrame({
        'const': [1],
        'Age Group': [age_group],
        'Gender': [gender],
        'Education Level': [education_level],
        'Income Level': [income_level]
    })
    
    relatedness_pred = model_relatedness.predict(input_data)[0]
    self_presentation_pred = model_self_presentation.predict(input_data)[0]
    sns_addiction_pred = model_sns_addiction.predict(input_data)[0]
    
    return relatedness_pred, self_presentation_pred, sns_addiction_pred

# Example of predicting factors for a new input
new_data = {
    'Age Group': 25,
    'Gender': 1,  # Male
    'Education Level': 3,  # Employed
    'Income Level': 3,  # 1000-3000
}

relatedness_pred, self_presentation_pred, sns_addiction_pred = predict_factors(
    new_data['Age Group'],
    new_data['Gender'],
    new_data['Education Level'],
    new_data['Income Level']
)

print(f"Predicted Relatedness Satisfaction: {relatedness_pred:.2f}")
print(f"Predicted Self-Presentation Satisfaction: {self_presentation_pred:.2f}")
print(f"Predicted SNS Addiction: {sns_addiction_pred:.2f}")
