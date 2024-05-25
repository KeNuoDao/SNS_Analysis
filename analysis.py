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

# Define the independent variables (X) and the dependent variable (y)
X = df[['Age Group', 'Gender', 'Education Level', 'Income Level', 'Relatedness Satisfaction', 'Self-Presentation Satisfaction']]
y = df['SNS Addiction']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())

# Function to predict SNS addiction level based on input data
def predict_sns_addiction(age_group, gender, education_level, income_level, relatedness_satisfaction, self_presentation_satisfaction):
    input_data = pd.DataFrame({
        'const': [1],
        'Age Group': [age_group],
        'Gender': [gender],
        'Education Level': [education_level],
        'Income Level': [income_level],
        'Relatedness Satisfaction': [relatedness_satisfaction],
        'Self-Presentation Satisfaction': [self_presentation_satisfaction]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Example of predicting SNS addiction level for a new input
new_data = {
    'Age Group': 25,
    'Gender': 1,  # Male
    'Education Level': 3,  # Employed
    'Income Level': 3,  # 1000-3000
    'Relatedness Satisfaction': 5,
    'Self-Presentation Satisfaction': 5
}

predicted_addiction = predict_sns_addiction(
    new_data['Age Group'],
    new_data['Gender'],
    new_data['Education Level'],
    new_data['Income Level'],
    new_data['Relatedness Satisfaction'],
    new_data['Self-Presentation Satisfaction']
)

print(f"Predicted SNS Addiction Level: {predicted_addiction:.2f}")
