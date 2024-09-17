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
- [Data Visualization](#data-visualization)

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

[Download Vet Group - forecast.pbix](Vet%20Group%20-%20forecast.pbix)

The link above will redirect you to another page. Once there, click on Download Raw in the top-right corner to download the file.

This project integrates Python and Power Query for time series forecasting. The workflow encompasses data preprocessing, feature engineering, model training, and forecasting with the XGBoost algorithm. The script is optimized to run within Power BI, utilizing Python scripts embedded in Power Query.

## Final Model Integration in Power Query

1. **Data Preprocessing**
   Use connect to our csv file through power query and then we use python. 
   The data is loaded and preprocessed using **Pandas**. This includes parsing dates and grouping data by the `Date` column to create a time series with business day frequency (`asfreq('B')`).
   In this step we aggregate the values by date to generate the number of prediction as whole.
   
    ```python
    # 'dataset' holds the input data for this script
    df = dataset
   
    # For data manipulation
    import pandas as pd
    import numpy as np
   
    # For data modeling
    import xgboost as xgb
    from scipy.fft import fft
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
    ts = df[['Date', 'Treatments']].groupby('Date').sum().asfreq('B').ffill()
    ```
   
3. **Feature Engineering**  
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

4. **Model Training (XGBoost)**  
   The model is trained using **XGBoost** to predict future values of `Treatments`. All features are used to predict the target.

    ```python
    model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.03, max_depth=2)
    model.fit(X_all, y_all)
    ```

5. **Forecasting Future Values**  
   A future date range is generated, and the trained model is used to predict future treatments. Features are computed for the future dates using the same functions.
   At the end of this process we have the predictions in 
    ```python
    future = pd.date_range(start_date, end_date, freq='B')
    future_with_features['forecast'] = model.predict(future_with_features[features])
    ```

6. **Power Query Integration**  
   The resulting forecasted values are merged with the helper query used to determine the number of treatments and average treatment time per Country and pacient category. Additional transformations such as column renaming, type conversions, and merging with external datasets are done here.

    ```powerquery
    #"Merged Queries" = Table.NestedJoin(#"Changed Type2", {"index"}, vet_data_avg_decision, {"List_of_dates"}, "vet_data_avg_decision", JoinKind.LeftOuter),
    #"Expanded vet_data_avg_decision" = Table.ExpandTableColumn(#"Merged Queries", "vet_data_avg_decision", {"Country", "Pacient_Category", "Avg_treatments", "Distribution"}, {"Country", "Pacient_Category", "Avg_treatments", "Distribution"}),
    #"Added Custom1" = Table.AddColumn(#"Expanded vet_data_avg_decision", "Treatments", each Number.Round([forecast]*[Distribution],0)),
    #"Changed Type" = Table.TransformColumnTypes(#"Added Custom1",{{"Treatments", Int64.Type}}),
    #"Renamed Columns" = Table.RenameColumns(#"Changed Type",{{"index", "Date"}, {"Avg_treatments", "Avg treatment time in minutes"}}),
    #"Removed Columns1" = Table.RemoveColumns(#"Renamed Columns",{"Distribution", "forecast"})
    ```

## Final Forecast

The final result are then merged in a final query that aggregates actuals and forecast data

 ```powerquery
    let
    Source = main,
    #"Added Custom" = Table.AddColumn(Source, "Nature", each "Actuals"),
    #"Changed Type1" = Table.TransformColumnTypes(#"Added Custom",{{"Nature", type text}}),
    #"Appended Query" = Table.Combine({#"Changed Type1", forecast}),
    #"Changed Type" = Table.TransformColumnTypes(#"Appended Query",{{"Date", type datetime}})
   in
    #"Changed Type"
 ```
For more details about the queries check the PBIX file in this repository.

## Data Visualization

The PBI report is divided into three pages:

1. **Actuals and Forecast Distribution**

![PBI - page 1](PBI%20-%20page%201.gif)
   
This page provides statistical insights into the data distribution. Metrics such as the average, 99th quantile, 1st quantile, and interquartile range (IQR) are dynamically updated based on the selected view.
Field Parameters are used to control the x-axis of the graph, while the y-axis (minimum and maximum values) adjusts dynamically depending on the selected field parameter view.
   
 ```DAX
format_max_axis = 
VAR yea = 
ROUNDUP(
    CALCULATE(
        MAXX(
            VALUES(calendar_table[year]),
            CALCULATE(SUM(vet_treatments[Treatments]))
        )
    ),-4
)

VAR qua = 
ROUNDUP(
     CALCULATE(
        MAXX(
            VALUES(calendar_table[Year_Quarter]),
            CALCULATE(SUM(vet_treatments[Treatments]))
        )
    ),-3
)

VAR mon = 
ROUNDUP(
    CALCULATE(
        MAXX(
            VALUES(calendar_table[Year_Month]),
            CALCULATE(SUM(vet_treatments[Treatments]))
        )
    ),-2
)

VAR day = 
ROUNDUP(
    CALCULATE(
        MAXX(
            VALUES(calendar_table[Date]),
            CALCULATE(SUM(vet_treatments[Treatments]))
        )
    ),-1
)



RETURN
    SWITCH(
        TRUE(),
        MAX(Params_table[Fields]) = "Daily View", day,
        MAX(Params_table[Fields]) = "Monthly View", mon,   
        MAX(Params_table[Fields]) = "Quarterly View", qua,
        MAX(Params_table[Fields]) = "Yearly View", yea               
    )
 ```
   
2. **FTE Work Capacity**

![PBI - page 2](PBI%20-%20page%202.gif)

On this page, we aim to assess whether the available workforce in the vet clinics is sufficient to meet the demand, based on both actual and forecasted values. Five key metrics are used to evaluate this.

1- Total_monthly volume

This metric calculates the total treatment volume per month by summing the treatment time and man-time for all treatments, then averaging the monthly totals. The result provides a measure of the workload in minutes per month, accounting for both the actual treatment time and the time between treatments.

Key Variables:
vet_treatments[Avg treatment time in minutes]: The average time spent on a single treatment.
Man_time[Man_time Value]: The break or setup time (in minutes) between treatments.
vet_treatments[Treatments]: The number of treatments performed.

 ```DAX
Total_monthly volume=
    AVERAGEX(
            ADDCOLUMNS(
            SUMMARIZE(vet_treatments, vet_treatments[year_month]),
            "Total_product",
                SUMX(
                    vet_treatments,
                    (vet_treatments[Avg treatment time in minutes] + Man_time[Man_time Value]) * vet_treatments[Treatments]
                    )
    ),
    [Total_product]
)
 ```

2- Needed work units

This metric calculates the number of work units (or full-time equivalent staff members) required to manage the total monthly treatment volume. It is based on the available working hours per day, excluding breaks.

 ```DAX
Needed Work units =
VAR ag = [Total_monthly volume]
VAR minutes = 60
VAR work_hours_without_breaks = 7.5
VAR total_minutes = minutes*work_hours_without_breaks

RETURN
DIVIDE(ag,total_minutes)
 ```

3- FTE's work units

This measure calculates the total number of work units provided by full-time employees (FTEs) over a specified period, such as a month. It excludes non-working days, such as weekends.

In the FTE Monthly Planner, you can find two field parameters that allow you to adjust the number of FTEs and the man-time (the time between treatments in minutes). These parameters provide the flexibility to explore how variations in these variables affect the remaining capacity.

 ```DAX
FTE's work units = 
VAR FTE = 'FTE''s'[FTE's Value]
VAR Days = 
CALCULATE (
    COUNTROWS ( calendar_table ),
    FILTER (
        calendar_table,
        WEEKDAY ( calendar_table[Date], 2 ) <= 5
    )
)

RETURN
IF([Needed Work units] = 0, BLANK(),FTE*Days)
 ```

4- Number of treatments

5- Average time of treatments in minutes

On this page, field parameters allow you to adjust the graphs in the top-right corner to display either treatments, average time, or cumulative values.

3. **ABC Classification by Country**

![PBI - page 3](PBI%20-%20page%203.gif)

This page aims to classify countries based on the volume of treatments, with A representing countries with the highest number of treatments and C representing those with the lowest.
This analysis helps business leaders identify which countries should receive more focus or resources.

To achieve this 4 DAX measures are necessary.

1- Create a rank of treatments

 ```DAX
ABC_rank = RANKX(ALL(vet_treatments[Country]),[Number Treatments])
 ```

2- Create a running total based on the rank

The measure calculates a cumulative running total of treatments based on the [ABC_rank] ranking. It shows how the number of treatments accumulates as you progress from the top-ranked countries to those with lower ranks.

 ```DAX
ABC_running_total = 
VAR rk = [ABC_rank]
RETURN
    CALCULATE(
        [Number Treatments],
        FILTER(
            ALL(vet_treatments[Country]),
            [ABC_rank] <= rk
        )
    )
 ```
The running total aggregates treatments for countries where the rank is less than or equal to the current rank (rk).

3- Create a Percentage of the running total by the total for the selected countries.

This measure calculates the percentage of the cumulative running total relative to the total for the selected countries. It provides insight into the proportion of treatments contributed by each ranked country relative to the overall total.

 ```DAX
ABC_% =

VAR ABC_total_by_country = 
 CALCULATE(
    [Number Treatments],
    ALLSELECTED(vet_treatments[Country])
 )

RETURN
DIVIDE([ABC_running_total],[ABC_total_by_country])
 ```
4- Create the ABC classes

Use a simply switch function to create the classification.

 ```DAX
ABC_class = 
SWITCH(
    TRUE(),
    [ABC_%] <= 'Pareto_%'[Pareto_% Value], "A",
    [ABC_%] <= 'Pareto_%'[Pareto_% Value]+0.1, "B",
    "C"
)
 ```
Field parameters are also used to allow users to choose the optimal threshold for classification.
