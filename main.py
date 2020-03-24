import math
import pandas as pd
import numpy as np

def read_data_training():
  df_train = pd.read_csv('train.csv', index_col='Id')
  df_train['Date'] = pd.to_datetime(df_train['Date'], format='%Y-%m-%d')
  return df_train

def prophet_calculation(df_train, label):
  from fbprophet import Prophet
  selected_df_train = df_train[['Date', label]]
  selected_df_train.rename(columns={'Date': 'ds'}, inplace=True)
  selected_df_train.rename(columns={label: 'y'}, inplace=True)
  # Train the model
  model = Prophet(growth='linear')
  model.fit(selected_df_train)
  # Predict the future
  future_df = model.make_future_dataframe(periods=50)
  forecast = model.predict(future_df)
  forecast.index = range(1, len(future_df) + 1)
  forecast = forecast[['ds', 'yhat']]
  forecast['y'] = selected_df_train['y']
  forecast['yhat'] = pd.to_numeric(forecast['yhat'])
  forecast.loc[forecast['yhat'] < 0, ['yhat']] = 0
  forecast = forecast.round({'yhat': 0})
  forecast.rename(columns={'ds': 'Date'}, inplace=True)
  forecast = forecast.set_index('Date')

  return forecast

def get_resonable_value(row):
  if math.isnan(row['y']):
    return row['yhat']
  return row['y']

def rmsle(actual_value, predicted_value):
  '''
  root mean square logarithmic error
  '''
  y_true, y_predict = np.array(actual_value), np.array(predicted_value)
  return math.sqrt(np.mean((np.log(y_predict + 1) - np.log(y_true + 1))**2))

if __name__ == "__main__":
  # Create submission dataframe
  submission_df = pd.DataFrame(columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])

  # Read test file
  df_test = pd.read_csv('test.csv')
  df_test['Date'] = pd.to_datetime(df_test['Date'], format='%Y-%m-%d')

  # Read data to train
  df_train = read_data_training()
  coordinates = list(zip(df_train['Lat'], df_train['Long'], df_train['Country/Region']))
  coordinates = list(dict.fromkeys(coordinates))
  count = 0
  for coordinate in coordinates:
    print(coordinate)
    df_train_region = df_train[((df_train['Lat'] == coordinate[0]) & (df_train['Long'] == coordinate[1]) & (df_train['Country/Region'] == coordinate[2]))]
    df_test_region = df_test[((df_test['Lat'] == coordinate[0]) & (df_test['Long'] == coordinate[1]) & (df_test['Country/Region'] == coordinate[2]))]
    df_test_region = df_test_region.set_index('Date')

    forecast_case = prophet_calculation(df_train_region, "ConfirmedCases")
    forecast_fatalities = prophet_calculation(df_train_region, "Fatalities")
    df_test_region['ConfirmedCases'] = forecast_case.apply(get_resonable_value, axis=1)
    df_test_region['Fatalities'] = forecast_fatalities.apply(get_resonable_value, axis=1)
    print(df_test_region)
    df_test_region = df_test_region[['ForecastId', 'ConfirmedCases', 'Fatalities']]
    # df_test_region = df_test_region.set_index('ForecastId')
    df_test_region['ConfirmedCases'] = df_test_region['ConfirmedCases'].astype(np.int64)
    df_test_region['Fatalities'] = df_test_region['Fatalities'].astype(np.int64)
    submission_df = submission_df.append(df_test_region, ignore_index=True)
    if count == 0:
      break
    count += 1
  submission_df.to_csv('submission.csv', index=False)
