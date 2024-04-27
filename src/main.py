import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("futuristic_city_traffic.csv")

# number of rows
total_rows = df.shape[0]

# calculate size to sample 1% of the total dataset
sample_size = int(0.01 * total_rows)

# sample 1% of your dataset
sampled_data = df.sample(n=sample_size, random_state=42)  # 42 is random number for reproducibility

print('Top rows of the dataset: \n', sampled_data.head())

print('Data types: \n', sampled_data.info())

print(sampled_data.dtypes)
print(sampled_data.describe())

# missing values(fill with mean)
sampled_data.fillna(sampled_data.mean(numeric_only=True), inplace=True)

print('Check number of null values: \n', sampled_data.isnull().sum())

# begin with numerical columns
numerical_cols = ['Hour Of Day', 'Speed', 'Is Peak Hour', 'Random Event Occurred', 'Energy Consumption', 'Traffic Density']
numerical_data = sampled_data[numerical_cols]

# visual inspection
plt.figure(figsize=(12, 6))
sns.boxplot(data=numerical_data)
plt.title('Box plot for numerical columns')
plt.xticks(rotation=45)
plt.show()

summary_stats = numerical_data.describe()
print("Summary Statistics:")
print(summary_stats)

# IQR Method

Q1 = numerical_data.quantile(0.25)
Q3 = numerical_data.quantile(0.75)
IQR = Q3 - Q1
print("\nInterquartile Range (IQR):")
print(IQR)

outliers = ((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).any(axis=1)
print("\nOutliers detected using IQR method:")
print(numerical_data[outliers])

# Z-score method
z_scores = stats.zscore(numerical_data)
threshold = 3
outliers_zscore = (abs(z_scores) > threshold).any(axis=1)
print("\nOutliers detected using Z-score method:")
print(numerical_data[outliers_zscore])

# take traffic density as main column
traffic_density = sampled_data['Traffic Density'].values.reshape(-1, 1)

# threshold for contamination
contamination_threshold = sampled_data['Traffic Density'].quantile(0.75) + 0.05

# Isolation Forest Model
clf = IsolationForest(contamination=contamination_threshold, random_state=42)

# Fit the model and predict outliers
outliers = clf.fit_predict(traffic_density)

sampled_data.reset_index(drop=True, inplace=True)

# '1' indicates inliers, '-1' indicates outliers
outliers_indices = sampled_data.index[outliers == -1]

print("Outliers detected using Isolation Forest:")
print(sampled_data.iloc[outliers_indices])

# remove outliers
cleaned_df = sampled_data.drop(outliers_indices)

print("Cleaned DataFrame without outliers:")
print(cleaned_df)

# Calculate average traffic density for each hour of the day
hourly_avg_density = cleaned_df.groupby('Hour Of Day')['Traffic Density'].mean().reset_index()

cleaned_df.to_csv('cleaned_sample.csv', index=False)

plt.figure(figsize=(10, 6))
sns.lineplot(data=hourly_avg_density, x='Hour Of Day', y='Traffic Density')
plt.title('Average Traffic Density by Hour Of Day (Without Outliers)')
plt.xlabel('Hour Of Day')
plt.ylabel('Average Traffic Density')
plt.grid(True)
plt.show()
