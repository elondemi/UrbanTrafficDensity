import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('cleaned_sample.csv')

# Print data types and statistics
print(df.dtypes)
print(df.describe())

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Feature scaling
scaler = StandardScaler()
features = df.drop(['Traffic Density'], axis=1)
target = df['Traffic Density']
scaled_features = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Initialize the regressors
lgbm = LGBMRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
catboost = CatBoostRegressor(random_state=42, verbose=0)

# Train the models
lgbm.fit(X_train, y_train)
xgb.fit(X_train, y_train)
catboost.fit(X_train, y_train)

# Make predictions
predictions_lgbm = lgbm.predict(X_test)
predictions_xgb = xgb.predict(X_test)
predictions_catboost = catboost.predict(X_test)

# Evaluate the models
mae_lgbm = mean_absolute_error(y_test, predictions_lgbm)
mse_lgbm = mean_squared_error(y_test, predictions_lgbm)
r2_lgbm = r2_score(y_test, predictions_lgbm)

mae_xgb = mean_absolute_error(y_test, predictions_xgb)
mse_xgb = mean_squared_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)

mae_catboost = mean_absolute_error(y_test, predictions_catboost)
mse_catboost = mean_squared_error(y_test, predictions_catboost)
r2_catboost = r2_score(y_test, predictions_catboost)

# Print evaluation results
print('LightGBM - Traffic Density Prediction:')
print(f'MAE: {mae_lgbm}')
print(f'MSE: {mse_lgbm}')
print(f'R-squared: {r2_lgbm}')
print('------')
print('XGBoost - Traffic Density Prediction:')
print(f'MAE: {mae_xgb}')
print(f'MSE: {mse_xgb}')
print(f'R-squared: {r2_xgb}')
print('------')
print('CatBoost - Traffic Density Prediction:')
print(f'MAE: {mae_catboost}')
print(f'MSE: {mse_catboost}')
print(f'R-squared: {r2_catboost}')
