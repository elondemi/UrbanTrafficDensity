# Futuristic City Traffic Analysis

## Introduction

This Python script is part of a project for the Master’s degree in Computer and Software Engineering at the University of Prishtina. It is conducted under the guidance of Professor Lule Ahmedi and Assistant Mergim Hoti. The project is part of the course "Machine Learning" and is taken during the second semester.

This script analyzes traffic data from a futuristic city dataset (`futuristic_city_traffic.csv`) and aims to provide insights into traffic patterns and identify potential anomalies. The project can potentially demonstrate the synchronization of traffic lights based on city traffic density, offering insights into optimizing traffic light timing to improve traffic flow.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/elondemi/UrbanTrafficDensity.git
    ```

2. Navigate to the project directory:

    ```bash
    cd UrbanTrafficDensity
    ```

3. Install the required dependencies using pip:

    ```bash
    pip install pandas scikit-learn seaborn matplotlib
    ```

## Usage

To analyze the futuristic city traffic dataset, run the Python script `main.py`:

    ```bash
    python main.py
    ```

The script will:

1. Load the dataset and sample 1% of the total dataset for analysis.
2. Handle missing values by filling them with the mean of each column.
3. Perform outlier detection using the Interquartile Range (IQR) method, Z-score method, and Isolation Forest algorithm.
4. Visualize the data using box plots and line plots.
5. Output the detected outliers and the cleaned dataset without outliers.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- seaborn
- matplotlib

## Phase I: Model Preparation

Phase I involves preparing the model for analyzing traffic data. The dataset consists of 1,048,576 rows and was sourced from Kaggle. For the analysis, a specific percentage of the dataset was used as a sample.

- **Data Loading and Sampling**:
    ```python
    # Load your dataset
    df = pd.read_csv("futuristic_city_traffic.csv")

    # Calculate size to sample 1% of the total dataset
    sample_size = int(0.01 * df.shape[0])

    # Sample 1% of your dataset
    sampled_data = df.sample(n=sample_size, random_state=42)
    ```

    - Loads the dataset from `futuristic_city_traffic.csv` using pandas.
    - Calculates and samples 1% of the dataset for analysis, ensuring reproducibility with a random state.

- **Handling Missing Values**:
    ```python
    # Fill missing values with mean
    sampled_data.fillna(sampled_data.mean(numeric_only=True), inplace=True)
    ```

    - Calculates the mean of numeric columns and fills missing values with the mean.

- **Isolation Forest Model**:
    ```python
    # Initialize and fit the Isolation Forest model
    clf = IsolationForest(contamination=contamination_threshold, random_state=42)
    outliers = clf.fit_predict(traffic_density)
    ```

    - Initializes an Isolation Forest model and predicts outliers.

## Figures

Attached figures:

- **Box plot for numerical columns**:

    ![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/488d9e62-c8fe-42b7-96a7-17ee688c8e3a)

- **Line plot showing average traffic density by hour of the day**:

    ![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/5ace6f37-9ab9-4f14-bf24-d48720f9ddb9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributors

- [Elon Demi](https://github.com/elondemi)
- [Lorent Sinani](https://github.com/lorentsinani)
