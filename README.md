# Time-Series-Forecasting with Python and Power BI Integration

### Table of Contents

Part 1 - Time-Series-Forecasting with Python 
- [Introduction](#introduction)
- [Prediction Models using Python](#prediction-models-using-python)
- [Hyperparameters and cross-validation](#hyperparameters-and-cross-validation)
- [Evaluation metrics](#evaluation-metrics)
- [Conclusion and Next steps](#conclusion-and-next-steps)

Part 2 - Power BI Integration
- [Power BI Integration](#power-bi-integration)

## Introduction
We analyse the number of treatments occured in a veterinaries clinics and try to predict the future number of animals treated.
Our fictional vet clinic ( VetGroup ) wants to understand if there is a reliable way to predict the inflow of pacients coming through their doors.
The predictions of the model could have an important impact in the company by anticipating high and low peaks of treatment. Also it can help to plan staff leaves, stock for meds and accomodation necessary for the animals.

## Prediction Models using Python
The aim of this analysis is to try to find the model that predicts most accurately the future number of treatments.
To achieve this goal I will explore three different models:
- Exponential smoothing [more info](https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html)
- Sarimax [more info](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- XG Boosting for time series regression [more info](https://www.analyticsvidhya.com/blog/2024/01/xgboost-for-time-series-forecasting/)

The first two are statistical models while XG Boosting is a machine learning model.

Each model will follow these stages

### Spliting the data

The data will be splitted in two parts.

- Training data: for the model to understand the patterns present
- Test data: where we will confront the prediction of the model with the actuals values of our test data

## Hyperparameters and cross-validation 

Which model contaings their specifications for diferent parameters that they can accept.

Grid search will be employed in order to test all possible combinations and find the best predictions.

To minimize the risk of overfitting our data, meaning that we will explain the training data very well but the prediction will not generalize well, the time series validator will be used.
The validator will use the technic of cross-validation.

This is how the cross-validation for time series works: in simple terms the data is divided into folds and each fold is ordered cronologically.

Example: if we specify 5 folds, the training data will create 5 steps for validation.

This process ensures that the data does not overfits and will generalize well.

## Evaluation metrics

The evaluation metrics used for comparing model performance are the following:

- Mean Absolute Error (MAE) [more info](https://en.wikipedia.org/wiki/Mean_absolute_error)
- Mean Squarred Error (MSE) [more info](https://en.wikipedia.org/wiki/Mean_squared_error)
- Root Mean squarred Error (RMSE) [more info](https://en.wikipedia.org/wiki/Root_mean_square_deviation)

RMSE will be used as the main metric of evaluation while the others will be a complementary indicators of performance.

### Conclusion and Next steps
Model  |RMSE |MAE |MSE
-----|-----|-----|-----| 
XG_Boosting|23.596686|15.324856|593.103966|
SARIMAX|28.654489|21.131146|847.697908|
Exponential smoothing|32.776729|25.498562|1166.163153|

We can check in this example that XG Boosting had the best performance of the three models.

For this model feature engineering was essential to produce good results. Always check if your features have an importance for the model, if not, remove them, or create new ones.

All the details about the code used are in the Jupyter Notebook ;)

# Power BI Integration

This project combines Python and Power Query for time series forecasting. The workflow includes data preprocessing, feature engineering, model training, and forecasting using the **XGBoost** algorithm. The script is designed to work within Power BI by leveraging Python scripts in Power Query.

## Workflow Overview

1. **Data Preprocessing**  
   The data is loaded and preprocessed using **Pandas**. This includes parsing dates and grouping data by the `Date` column to create a time series with business day frequency (`asfreq('B')`).

    ```python
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
    ts = df[['Date', 'Treatments']].groupby('Date').sum().asfreq('B').ffill()
    ```

2. **Feature Engineering**  
   New features are added to the dataset, such as time-based features (e.g., day of the week, month, year), lag features (e.g., previous day’s treatments), and **Fourier transforms** for capturing seasonality.

    ```python
    def date_features(data):
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        # More features...
        return data
    ```

    **Lag features** and **Fourier transforms** capture historical dependencies and seasonality patterns.

    ```python
    data['lag_day'] = data.index - pd.offsets.BDay(1)
    data['lag_day'] = data['lag_day'].map(target_map)
    
    # Fourier Transform to capture periodic patterns
    fourier_transform = fft(values)
    data['fourier'] = np.abs(fourier_transform)
    ```

3. **Model Training (XGBoost)**  
   The model is trained using **XGBoost** to predict future values of `Treatments`. All features are used to predict the target.

    ```python
    model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.03, max_depth=2)
    model.fit(X_all, y_all)
    ```

4. **Forecasting Future Values**  
   A future date range is generated, and the trained model is used to predict future treatments. Features are computed for the future dates using the same functions.

    ```python
    future = pd.date_range(start_date, end_date, freq='B')
    future_with_features['forecast'] = model.predict(future_with_features[features])
    ```

5. **Power Query Integration**  
   The resulting forecasted values are merged with the original dataset using Power Query. Additional transformations such as column renaming, type conversions, and merging with external datasets are done here.

    ```powerquery
    #"Merged Queries" = Table.NestedJoin(#"Changed Type2", {"index"}, vet_data_avg_decision, {"List_of_dates"}, "vet_data_avg_decision", JoinKind.LeftOuter),
    #"Renamed Columns" = Table.RenameColumns(#"Changed Type",{{"index", "Date"}, {"Avg_treatments", "Avg treatment time in minutes"}})
    ```

## Key Features and Concepts

- **Date and Time Features**: Extract temporal aspects like day of the week, month, year.
- **Lag Features**: Capture the delayed impact of previous treatments on the current period.
- **Fourier Transforms**: Identify seasonal patterns in the data.
- **Rolling Means**: Smoothing the data with moving averages to capture trends.

## Final Forecast

The final result merges historical data with future forecasts, providing insights into future treatments, which can be used for planning and decision-making.

```powerquery
#"Added Custom1" = Table.AddColumn(#"Expanded vet_data_avg_decision", "Treatments", each Number.Round([forecast]*[Distribution],0)),
#"Changed Type" = Table.TransformColumnTypes(#"Added Custom1",{{"Treatments", Int64.Type}}),


In the next part of this project I will implement the final model in Power BI in order for business users to have an easy way to check the predictions of our model.



