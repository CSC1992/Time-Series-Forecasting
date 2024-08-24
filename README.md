# Time-Series-Forecasting with Python Part 1

### We analyse the number of treatments occured in a veterinaries clinics and try to predict the future number of animals treated.
Our fictional vet clinic ( VetGroup ) wants to understand if there is a reliable way to predict the inflow of pacients coming through their doors.
The predictions of the model could have an important impact in the company by anticipating high and low peaks of treatment. Also it can help to plan staff leaves, stock for meds and accomodation necessary for the animals.

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

### Hyperparameters and cross-validation 

Which model contaings their specifications for diferent parameters that they can accept.

Grid search will be employed in order to test all possible combinations and find the best predictions.

To minimize the risk of overfitting our data, meaning that we will explain the training data very well but the prediction will not generalize well, the time series validator will be used.
The validator will use the technic of cross-validation.

This is how the cross-validation for time series works: in simple terms the data is divided into folds and each fold is ordered cronologically.

Example: if we specify 5 folds, the training data will create 5 steps for validation.

This process ensures that the data does not overfits and will generalize well.

### Evaluation metrics

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

In the next part of this project I will implement the final model in Power BI in order for business users to have an easy way to check the predictions of our model.



