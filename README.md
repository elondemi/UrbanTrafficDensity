# Futuristic City Traffic Analysis

## Introduction

This Python script analyzes traffic data from a futuristic city dataset (`futuristic_city_traffic.csv`). It performs data preprocessing, outlier detection, and visualization to gain insights into traffic patterns and identify potential anomalies.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/elondemi/UrbanTrafficDensity.git
```

2. Navigate to the project directory:

```bash
cd UrbanTrafficDensity
```

3. Ensure you have the required dependencies installed. You can install them using pip:

```bash
pip install pandas scikit-learn seaborn matplotlib
```

## Usage

Run the Python script `main.py` to analyze the futuristic city traffic dataset.

```bash
python main.py
```

The script will perform the following steps:

1. Load the dataset and sample 1% of the total dataset for analysis.
2. Handle missing values by filling them with the mean of each column.
3. Perform outlier detection using Interquartile Range (IQR) method, Z-score method, and Isolation Forest algorithm.
4. Visualize the data using box plots and line plots.
5. Output the detected outliers and the cleaned dataset without outliers.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- seaborn
- matplotlib

## Main Sections

1. **Data Loading and Sampling**:
```python
# Load your dataset
df = pd.read_csv("futuristic_city_traffic.csv")

# number of rows
total_rows = df.shape[0]

# calculate size to sample 1% of the total dataset
sample_size = int(0.01 * total_rows)

# sample 1% of your dataset
sampled_data = df.sample(n=sample_size, random_state=42)  # 42 is random number for reproducibility
```
Explanation:
- `pd.read_csv("futuristic_city_traffic.csv")`: This line loads the dataset from a CSV file named `"futuristic_city_traffic.csv"` using pandas.
- `df.shape[0]`: Returns the number of rows in the DataFrame `df`.
- `int(0.01 * total_rows)`: Calculates the size to sample 1% of the total dataset.
- `df.sample(n=sample_size, random_state=42)`: Randomly samples `sample_size` rows from the DataFrame `df`. The `random_state=42` ensures reproducibility of the results.

2. **Handling Missing Values**:
```python
# missing values(fill with mean)
sampled_data.fillna(sampled_data.mean(numeric_only=True), inplace=True)
```
Explanation:
- `sampled_data.mean(numeric_only=True)`: Calculates the mean of numeric columns in the DataFrame `sampled_data`.
- `sampled_data.fillna()`: Fills missing values in `sampled_data` with the mean values calculated above. The `inplace=True` parameter modifies the DataFrame in place.

3. **Isolation Forest Model**:
```python
# Isolation Forest Model
clf = IsolationForest(contamination=contamination_threshold, random_state=42)

# Fit the model and predict outliers
outliers = clf.fit_predict(traffic_density)
```
Explanation:
- `IsolationForest()`: Initializes an Isolation Forest model with the specified `contamination` (proportion of outliers) and `random_state` (for reproducibility).
- `clf.fit_predict(traffic_density)`: Fits the Isolation Forest model to the `traffic_density` data and predicts outliers. `-1` indicates outliers, while `1` indicates inliers.

## Figures

Attached figures:

- Box plot for numerical columns:

![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/488d9e62-c8fe-42b7-96a7-17ee688c8e3a)

- Line plot showing average traffic density by hour of the day:

![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/5ace6f37-9ab9-4f14-bf24-d48720f9ddb9)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Elon Demi](https://github.com/elondemi)

- [Lorent Sinani](https://github.com/lorentsinani)
