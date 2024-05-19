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
This phase focuses on training and evaluating machine learning models to predict traffic density. We explore three popular gradient boosting algorithms (LightGBM, XGBoost, and CatBoost) and a baseline linear regression model to compare their performance in predicting traffic density.

### Data Preparation
- The dataset is loaded from a CSV file named `cleaned_sample.csv`.
- Categorical variables are encoded using label encoding.
- Missing values in numerical columns are filled with the mean of each column.
- Features are scaled using standard scaling for uniformity and comparability across features.

### Model Training
- The dataset is split into training and test sets with an 80:20 ratio.
- Three gradient boosting models (LightGBM, XGBoost, CatBoost) and a baseline linear regression model are trained using the training data.
- Hyperparameters for the models are set to default values for simplicity.

### Model Evaluation
- The trained models are evaluated using the test set.
- Evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2).
- Results are presented for each model, providing insights into their performance in predicting traffic density.

### Comparison and Analysis
- **Linear Regression**: The linear regression model serves as a baseline. It exhibits a higher error rate and a negative R-squared value, indicating a poor fit for the dataset. This suggests linear regression may not be an effective algorithm for predicting traffic density in our data.
- **Gradient Boosting Models**: LightGBM, XGBoost, and CatBoost deliver significantly better performance compared to linear regression, with lower MAE and MSE values and higher R-squared scores. These models excel at capturing complex patterns in the data and offer more accurate predictions.

The comparison between linear regression and gradient boosting models demonstrates the superiority of gradient boosting techniques for this task. Using different boosting models provides flexibility in exploring multiple approaches to identify the best strategy for predicting traffic density.

### Visualization
- Evaluation results are visualized using bar plots comparing the performance of models across different metrics (MAE, MSE, and R2).
- Bar plots offer a clear comparison between models, facilitating model selection and interpretation of results.

### Files Included
- `cleaned_sample.csv`: Input dataset containing cleaned data for traffic density prediction.
- `gradient_boosting_model_training.py`: Python script for model training and evaluation using gradient boosting algorithms.
- `linear_regression.py`: Python script for linear regression model training and evaluation.

### Usage
1. Set up your Python environment with the required libraries (`pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`).
2. Run the `gradient_boosting_model_training.py` and `linear_regression.py` scripts to train the models and evaluate their performance.
3. Review the printed evaluation results and visualizations to compare models.
4. Analyze each model's performance to make informed decisions for further model tuning or deployment.

### Figures

- **Box plot for numerical columns**:

![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/488d9e62-c8fe-42b7-96a7-17ee688c8e3a)

- **Line plot showing average traffic density by hour of the day**:

![image](https://github.com/elondemi/UrbanTrafficDensity/assets/66006296/5ace6f37-9ab9-4f14-bf24-d48720f9ddb9)

- **Bar plots**: Phase II includes three bar plots showing the performance of models across different evaluation metrics:

    1. **Mean Absolute Error (MAE) Plot**: This plot shows MAE values for each model (LightGBM, XGBoost, CatBoost) on the y-axis. Each model is represented by a colored bar, with the height of the bar indicating the MAE value.

    2. **Mean Squared Error (MSE) Plot**: Similar to the MAE plot, this plot displays MSE values for each model, with bar heights representing MSE values for each model.

    3. **R-squared (R2) Plot**: This plot visualizes R-squared values for each model. The R-squared value indicates the proportion of variance in the target variable (traffic density) that is explained by the model. Models are represented by colored bars, with height representing the R2 value.

These bar plots provide a visual comparison of different models across various evaluation metrics, aiding in model selection and interpretation.

- **Linear Regression Results Visualization**:

    ![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/f0d2bfc8-7ebb-4d7c-a230-e64b7148e1fd)

    This image visualizes the results of the linear regression model, providing insights into its performance in predicting traffic density. The plot shows how the model performs compared to actual traffic density values, highlighting areas where the model may underperform.

![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/224a02c9-21c1-471d-bbdf-e73e80a38782)


## Phase III: Application of Machine Learning Tool

Phase III involves the application of machine learning tools to analyze and visualize the futuristic city traffic dataset. In this phase, we utilize Python libraries such as pandas, numpy, seaborn, and matplotlib to perform data analysis and visualization.
Phase III involves evaluating and selecting the best-performing machine learning models for predicting traffic density in the futuristic city environment. We use various regression techniques and ensemble methods to build and assess the performance of these models.


### Data Analysis and Visualization

We start by loading the dataset `futuristic_city_traffic.csv` using pandas:

```python
import pandas as pd

dataset = pd.read_csv('futuristic_city_traffic.csv')
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/d5cb593c-d6cf-49e8-8f84-41822403457d)




# Visualization 
## Density of traffic based on the city
The visualization below displays the average traffic density based on different cities in the dataset.


```python
traffic_city = dataset.groupby('City')['Traffic Density'].mean().reset_index()
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/ebc28a87-fb36-412a-bd93-a242a23e1b7f)


```python

plt.bar(data=traffic_city, x=traffic_city['City'], height=traffic_city['Traffic Density'])
plt.title('Average Traffic Density Based on City')
plt.xticks(rotation=90)
plt.show()
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/d5fa1b8a-3dc2-4515-aa80-85ae5f8e70b5)

The bar plot illustrates significant variations in traffic density across different cities. Observing these disparities can provide valuable insights for urban planners and policymakers in understanding traffic patterns and allocating resources efficiently. For instance, cities with higher average traffic density may require infrastructure improvements or alternative transportation solutions to mitigate congestion and enhance mobility.

By examining this visualization, we can identify cities where traffic density is particularly high or low, allowing stakeholders to prioritize interventions and investments accordingly.

In future iterations of this project, further analysis could explore the underlying factors contributing to variations in traffic density among different cities. Additionally, integrating real-time traffic data and predictive modeling techniques could enhance the accuracy and relevance of traffic management strategies.



## Density of traffic based on the Weather
The following visualization illustrates the average traffic density categorized by different weather conditions.


```python
traffic_weather = dataset.groupby('Weather')['Traffic Density'].mean().reset_index()
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/59852d8b-d91e-48a5-9d2a-2e2d5a9b973d)

```python

plt.bar(data=traffic_weather, x=traffic_weather['Weather'], height=traffic_weather['Traffic Density'])
plt.title('Average Traffic Density Based on Weather')
plt.xticks(rotation=90)
plt.show()
```

![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/13bc981e-672a-4d78-8d68-55254b47ed28)

This bar plot provides insights into how weather conditions impact traffic density.

In urban planning and transportation management, understanding the relationship between weather and traffic density is crucial for optimizing traffic flow and ensuring road safety. Inclement weather conditions such as rain, snow, or fog often lead to decreased visibility and increased travel time, resulting in higher traffic congestion. By analyzing historical traffic data under different weather conditions, authorities can implement targeted strategies such as adaptive traffic signal control or weather-responsive routing to alleviate congestion and enhance road safety during adverse weather events.

## Amount of vehicle types from each category
The visualization below illustrates the distribution of vehicle types across different categories in the dataset.


```python
cars = dataset['Vehicle Type'].value_counts().reset_index()
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/c1e242bf-f969-416e-b935-73ad9e672166)

```python

plt.pie(labels=cars['index'], x=cars['Vehicle Type'], autopct='%1.1f%%')
plt.legend()
plt.show()
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/685268b1-7449-430a-825a-001af646a78e)

This pie chart provides a comprehensive overview of the composition of vehicles in the dataset, categorized by type. Understanding the distribution of vehicle types is essential for various stakeholders involved in urban transportation planning, infrastructure development, and environmental impact assessment.

By analyzing the proportion of different vehicle types, policymakers and transportation authorities can make informed decisions regarding infrastructure investments, public transit planning, and emissions reduction strategies. For instance, a high percentage of private vehicles may indicate a need for improved public transportation options or incentives for carpooling and ridesharing to alleviate traffic congestion and reduce carbon emissions.



# Correlation analysis
## Does Traffic density increase of decrease energy consumption
To explore the relationship between traffic density and energy consumption, we calculated the correlation coefficient between these two variables.
```python

correlation = dataset['Energy Consumption'].corr(dataset['Traffic Density'])
//0.01584834966727215
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/733456bd-a10c-44e9-9cb5-150b21cf07e3)


The correlation coefficient of approximately 0.0158 suggests a very weak positive correlation between traffic density and energy consumption. While this correlation is statistically significant due to the large dataset, its practical significance is minimal.

This weak correlation indicates that changes in traffic density have only a marginal impact on energy consumption. Other factors such as vehicle efficiency, driving behaviors, road conditions, and infrastructure play more substantial roles in determining energy consumption patterns.

From a policy perspective, understanding the relationship between traffic density and energy consumption is essential for developing sustainable transportation strategies. While reducing traffic congestion can potentially lead to energy savings, it is crucial to consider holistic approaches that promote energy-efficient vehicles, alternative transportation modes, and infrastructure improvements to achieve meaningful reductions in energy consumption and environmental impact.

In future analyses, exploring the interplay between traffic density, energy consumption, and environmental factors such as air quality and carbon emissions could provide deeper insights into the complex dynamics of urban transportation systems and inform evidence-based policy decisions.


## Visualization

```python

plt.scatter(data=dataset, x='Energy Consumption', y='Traffic Density')
plt.xlabel('Energy Consumption')
plt.ylabel('Traffic Density')
plt.show()
print("Correlation between Energy Consumption and Traffic Density:", correlation)

```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/5f61976c-760d-432a-b78c-96f9fb78a67a)

This scatter plot depicts the relationship between energy consumption and traffic density. Despite a weak positive correlation (correlation coefficient ≈ 0.0158), as shown in the scatter plot, no clear linear trend emerges. Understanding this relationship is crucial for optimizing urban transportation and energy efficiency strategies.


## Average speed of every vehicle
```python
every_hour_density = dataset.groupby('hour/day')['Traffic Density'].mean().sort_values(ascending=False)



average_speed = dataset.groupby('Vehicle Type')['Speed'].mean()

print("Average Speed of Every Vehicle Type:\n", average_speed)
```
![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/e5d4bcb9-f417-491c-bfcb-716f348dfd30)

This bar plot displays the average speed of each vehicle type. Understanding the speed characteristics of different vehicle types is essential for traffic management and infrastructure planning.

Analyzing the average speed of vehicles can help identify congestion hotspots, optimize traffic flow, and improve road safety. Additionally, it informs decisions regarding speed limits, lane configurations, and traffic signal timing to enhance overall transportation efficiency.

## Does a car consume more energy when moving faster?
```python

correlation_speed_energy = dataset['Energy Consumption'].corr(dataset['Speed'])


plt.scatter(x='Speed', y='Energy Consumption', data=dataset)
plt.xlabel('Speed of the Car')
plt.ylabel('Energy Consumption')
plt.show()



print("Correlation between Speed and Energy Consumption:", correlation_speed_energy)
```

![image](https://github.com/elondemi/UrbanTrafficDensity/assets/84631230/94fd1bc1-63c4-454c-b494-dbdd5f0a2021)

This scatter plot illustrates the relationship between car speed and energy consumption. We calculate the correlation coefficient to measure the strength of this relationship.

Analyzing this correlation helps us understand the energy efficiency of different driving speeds. While the scatter plot indicates a positive correlation, suggesting that cars tend to consume more energy at higher speeds, the correlation coefficient quantifies this relationship.

By comprehending the impact of speed on energy consumption, policymakers and vehicle manufacturers can develop strategies to promote energy-efficient driving behaviors and improve fuel economy.

# Encoding Categorical Variables
```python

object_data = ohe.fit_transform(object_data)

```

# Removing Outliers (Z-Score Method)
```python

z_scores = stats.zscore(numeric_data)
threshold = 3
outliers_zscore = (abs(z_scores) > threshold).any(axis=1)
numeric_data = numeric_data[~outliers_zscore]
```

# Removing Outliers (IQR Method)
```python

Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
numeric_data = numeric_data[~outliers]
```
# Model Training and Evaluation
## 1. Ridge Regression
```python

ridge = Ridge()
grid = GridSearchCV(estimator=ridge,param_grid=param_grid,cv=5,n_jobs=-1)
grid.fit(X_train,y_train)

```
## 2. Elastic Net Regression
```python

elastico = ElasticNet()
elasticgrid = GridSearchCV(estimator=elastico,param_grid=param_grid,n_jobs=-1,scoring='neg_mean_squared_error',cv=10)
elasticgrid.fit(X_train,y_train)
```

## 3. Random Forest Regressor
```python

rfr = RandomForestRegressor()
randomforestgrid = GridSearchCV(estimator=rfr,param_grid=random_grid,cv=3,n_jobs=-1,scoring='neg_mean_squared_error')
randomforestgrid.fit(X_train,y_train)
```

## 4. XGBoost Regressor
```python

xgb_model = XGBRegressor()
grid_searchxgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
grid_searchxgb.fit(X_train,y_train)
```

## 5. LightGBM Regressor
```python

lgb_model = LGBMRegressor()
grid_searchlgb = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',n_jobs=-1)
grid_searchlgb.fit(X_train,y_train)
```

## 6. Linear Regression
```python

normal_model = LinearRegression(n_jobs=-1)
normal_model.fit(X_train,y_train)
```

# Model Performance Evaluation

We evaluate the performance of each model using Mean Absolute Error (MAE) and Mean Squared Error (MSE) to assess accuracy and precision in predicting traffic density.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributors

- [Elon Demi](https://github.com/elondemi)
- [Lorent Sinani](https://github.com/lorentsinani)
