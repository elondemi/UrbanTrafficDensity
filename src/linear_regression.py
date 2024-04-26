from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('cleaned_sample.csv')

df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Feature scaling
scaler = StandardScaler()
features = df.drop(['Traffic Density'], axis=1)  # 'Traffic Density' is the target column
target = df['Traffic Density']
scaled_features = scaler.fit_transform(features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
linear_regressor = LinearRegression()

# Train the model
linear_regressor.fit(X_train, y_train)
predictions_linear = linear_regressor.predict(X_test)
# Evaluate the model
mae_linear = mean_absolute_error(y_test, predictions_linear)
mse_linear = mean_squared_error(y_test, predictions_linear)
r2_linear = r2_score(y_test, predictions_linear)

# Print evaluation results
print('Linear Regression - Traffic Density Prediction:')
print(f'MAE: {mae_linear}')
print(f'MSE: {mse_linear}')
print(f'R-squared: {r2_linear}')
