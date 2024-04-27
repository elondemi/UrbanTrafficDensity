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

## Phase II: Model Training

### Overview
This phase of the project focuses on training and evaluating machine learning models to predict traffic density. Three popular gradient boosting algorithms (LightGBM, XGBoost, and CatBoost) are utilized, along with a baseline linear regression model, to compare their performance in predicting traffic density.

### Data Preparation
- The dataset is loaded from a CSV file named `cleaned_sample.csv`.
- Categorical variables are encoded using label encoding.
- Missing values in numerical columns are filled with the mean of each column.
- Features are scaled using standard scaling to ensure uniformity and comparability across different features.

### Model Training
- The dataset is split into training and test sets with a ratio of 80:20.
- Three gradient boosting models (LightGBM, XGBoost, and CatBoost) and a baseline linear regression model are trained using the training data.
- Hyperparameters for the models are kept at default values for simplicity.

## Model Evaluation
- The trained models are evaluated using the test set.
- Evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2).
- Results are printed out for each model, providing insights into their performance in predicting traffic density.

### Visualization
- The evaluation results are visualized using bar plots to compare the performance of the models across different metrics (MAE, MSE, R2).
- Bar plots provide a clear comparison between the models, aiding in model selection and interpretation of results.

### Files Included
- `cleaned_sample.csv`: Input dataset containing cleaned data for traffic density prediction.
- `gradient_boosting_model_training.py`: Python script containing code for model training and evaluation.
- `linear_regression.py`: Python script containing code for linear regression model training and evaluation.

### Usage
1. Ensure Python environment is set up with necessary libraries installed (`pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`).
2. Run the `gradient_boosting_model_training.py` and `linear_regression.py` script to train the models and evaluate their performance.
3. Review the printed evaluation results and visualizations to compare the models.
4. Analyze the performance of each model and make informed decisions for further model tuning or deployment.


## Figures

Attached figures:

- **Box plot for numerical columns**:

    ![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/488d9e62-c8fe-42b7-96a7-17ee688c8e3a)

- **Line plot showing average traffic density by hour of the day**:

    ![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/5ace6f37-9ab9-4f14-bf24-d48720f9ddb9)

- The visualization in Phase II consists of three bar plots, each representing different evaluation metrics for the trained machine learning models. Here's a breakdown of each bar plot:

    1. **Mean Absolute Error (MAE) Plot**: This plot displays the MAE values for each model (LightGBM, XGBoost, CatBoost) on the y-axis. Each model is represented by a colored bar, with the height of the bar indicating the MAE value. The x-axis shows the names of the           models.

    2. **Mean Squared Error (MSE) Plot**: Similar to the MAE plot, this plot shows the MSE values for each model. The height of each bar represents the MSE value for the corresponding model.

    3. **R-squared (R2) Plot**: This plot visualizes the R-squared values for each model. The R-squared value indicates the proportion of variance in the target variable (traffic density) that is explained by the model. Again, each model is represented by a bar, with         the height indicating the R-squared value.

    The purpose of these bar plots is to provide a visual comparison of the performance of different models across various evaluation metrics. They allow you to easily identify which model performs better in terms of MAE, MSE, and R-squared, aiding in the selection and     interpretation of the best model for predicting traffic density.

  ![image](https://github.com/lorentsinani/UrbanTrafficDensity/assets/66006296/4bbd9691-d24b-4d8a-9ccf-4caed6b62f19)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributors

- [Elon Demi](https://github.com/elondemi)
- [Lorent Sinani](https://github.com/lorentsinani)
