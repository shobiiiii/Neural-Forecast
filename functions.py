import os
import yfinance as yf
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ta.volatility import AverageTrueRange
from ta.trend import CCIIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import pytz

import holidays
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from datetime import timedelta
import datetime

import scipy.stats as stats

# --- Fetch Data and Preprocess
def process_asset_symbol(asset_type, asset):
    """
    Transforms the asset symbol based on the asset type.

    :param asset_type: The type of the asset ('Cryptocurrencies', 'Forex', or 'Stocks')
    :param asset: The asset symbol
    :return: The transformed asset symbol
    """
    if asset_type == 'Cryptocurrencies':
        symbol = asset + '-USD'
    elif asset_type == 'Forex':
        symbol = asset.replace('/', '') + '=X'
    elif asset_type == 'Stocks':
        symbol = asset  # No transformation needed for stocks
    return symbol

def extend_time_window(start_date, end_date):
    # Create business date range from start_date to end_date
    date_range = pd.bdate_range(start=start_date, end=end_date)

    # Calculate 80% of the date range length
    extension_length = int(len(date_range) * 1) # it should be multiply by 8, for demonstration purpose it is now

    # Calculate the new start date as 80% earlier
    new_start_date = date_range[0] - pd.offsets.BDay(extension_length)

    # Return the new start date and end date as strings
    return new_start_date.strftime('%Y-%m-%d'), end_date

def save_to_csv(data, symbol, start_date, end_date, interval, dir_name='data'):
    """
    Save the provided data to a CSV file. The file is saved in a directory specified by 'dir_name' and the
    file name is constructed using the symbol, start and end dates, and interval. If the file already exists,
    the function does not overwrite it and instead prints a message to the console. If the directory does not exist,
    it is created.

    :param data: A pandas DataFrame containing the data to be saved.
    :param symbol: A string representing the asset's symbol.
    :param start_date: A string representing the start date in the format 'YYYY-MM-DD'.
    :param end_date: A string representing the end date in the format 'YYYY-MM-DD'.
    :param interval: A string representing the interval for the data.
    :param dir_name: A string representing the name of the directory where the file will be saved. The default is 'data'.
    """
    # Check if directory exists, if not, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Construct the file name and path
    file_name = f"{symbol}_from_{start_date}_to_{end_date}_interval_{interval}.csv"
    file_path = os.path.join(dir_name, file_name)

    # Check if the file already exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 42:
        # print("CSV File Already Exist.")
        return
    else:
        # Save the data to a CSV file
        data.to_csv(file_path)

def fetch_data(symbol, start_date, end_date, interval='1d'):
    """
    Fetch historical price data for the given asset type and symbol.

    :param symbol: A string representing the asset's symbol (e.g., "EURUSD" for forex, "BTC-USD" for crypto, "TSLA" for stock)
    :param start_date: A string representing the start date in the format 'YYYY-MM-DD'
    :param end_date: A string representing the end date in the format 'YYYY-MM-DD'
    :return: A pandas DataFrame containing historical price data
    """
    dir_name = 'data'
    csv_filename = f"{symbol}_from_{start_date}_to_{end_date}_interval_{interval}.csv"
    csv_filepath = os.path.join(dir_name, csv_filename)

    if os.path.exists(csv_filepath) and os.path.getsize(csv_filepath) > 42:
        print("CSV File Already Exist. Loading data from CSV file.")
        hist_data = pd.read_csv(csv_filepath, parse_dates=[0], index_col=0)

    else:
        print("CSV File does not exist. Querying data from yfinance.")
        yf_symbol = yf.Ticker(symbol)
        hist_data = yf_symbol.history(start=start_date, end=end_date, interval=interval)
        save_to_csv(data=hist_data, symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)

    return hist_data

def plot_data(symbol, hist_data, start_date, end_date):

    """
    Plot the historical price data for the given symbol.

    :param symbol: A string representing the asset's symbol
    :param hist_data: A pandas DataFrame containing historical price data
    :param start_date: A string representing the start date in the format 'YYYY-MM-DD'
    :param end_date: A string representing the end date in the format 'YYYY-MM-DD'
    """
    fig, ax = plt.subplots()
    ax.plot(hist_data.index, hist_data['Close'])

    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Use a more precise date string for the x axis locations
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f"{symbol} Price ({start_date} to {end_date})")
    ax.grid(True)

    dir_name = "/var/www/html/static/images"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_path = f"/var/www/html/static/images/{symbol}_{start_date}_{end_date}_price.png"
    plt.savefig(file_path)
    plt.show()
    plt.close(fig)

    return 'file_path'

def add_features(data, asset_type):
    """
    Adds technical indicators to a DataFrame of price data.

    :param data: A pandas DataFrame containing price data
    :param asset_type: A string representing the type of the asset ("stock", "Forex", or "Cryptocurrencies")

    :returns: The input DataFrame with added technical indicators
    """

    # convert index to a DatetimeIndex
    data.index = pd.to_datetime(data.index, utc=True)

    # Calculate returns
    data['Returns'] = data['Close'].pct_change().dropna()

    periods = [3, 7, 14, 21]

    for period in periods:
        # Calculate SMA and EMA
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()

        # Calculate RSI
        rsi = RSIIndicator(data['Close'], window=period).rsi()
        data[f'RSI_{period}'] = rsi

        # Calculate ATR
        atr = AverageTrueRange(data['High'], data['Low'], data['Close'], window=period).average_true_range()
        data[f'ATR_{period}'] = atr

        # Calculate CCI
        cci = CCIIndicator(data['High'], data['Low'], data['Close'], window=period).cci()
        data[f'CCI_{period}'] = cci

    # Calculate OBV if asset_type is "stock"
    if asset_type != "Forex":
        obv = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        data[f'OBV'] = obv

        # Calculate OBV_SMA for different periods
        for period in periods:
            data[f'OBV_SMA_{period}'] = data['OBV'].rolling(window=period).mean()

    # Drop any NaN rows created due to the indicator calculations
    data.dropna(inplace=True)

    # Create the categorical features
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter

    # Perform one-hot encoding
    data = pd.get_dummies(data, columns=['day_of_week', 'month', 'quarter'])

    return data

def select_features(data, asset_type):
    """
    Selects the relevant features for the given asset type.

    :param data: A pandas DataFrame containing the data.
    :param asset_type: A string representing the type of asset ("Forex", "Cryptocurrencies", or "Stocks").
    :return: A tuple containing the features DataFrame and the target Series.
    """
    # Select columns to drop based on the asset type
    columns_to_drop = ['Dividends', 'Stock Splits']
    if asset_type == "Forex":
        columns_to_drop.append('Volume')

    # Drop the selected columns from the features
    features = data.drop(columns=columns_to_drop)

    # The target is always 'Lagged_Returns'
    target = data[['Returns']]

    return features, target

def normalize_data(features, target, feature_range=(-1, 1)):
    """
    Normalizes the features and target using MinMaxScaler.

    :param features: A pandas DataFrame containing the features.
    :param target: A pandas Series or DataFrame containing the target.
    :param feature_range: A tuple representing the desired range of transformed data.
    :return: A tuple containing the scaled features and target as numpy arrays.
    """
    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=feature_range)

    # Fit and transform the features and target
    scaled_features = scaler.fit_transform(features)
    scaled_target = scaler.fit_transform(target)

    return scaled_features, scaled_target, scaler

def split_data(features, target, test_size=0.2):
    """
    Split data into training and testing sets.

    Parameters:
    features (pd.DataFrame): DataFrame with features.
    target (pd.DataFrame): DataFrame with target.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    tuple: (features_train, features_test, target_train, target_test)
    """
    split_point = int(len(features) * (1 - test_size))
    features_train, features_test = features[:split_point], features[split_point:]
    target_train, target_test = target[:split_point], target[split_point:]
    return features_train, features_test, target_train, target_test

def create_sequences(data, target, feature_seq_length):
    """
    Create sequences from data and targets.

    Parameters:
    data (pd.DataFrame): DataFrame with data.
    target (pd.DataFrame): DataFrame with targets.
    seq_length (int): Length of the sequences.

    Returns:
    tuple: (np.array, np.array): Sequences of data and corresponding targets.
    """
    data_sequences = []
    target_vector = []
    for i in range(len(data) - feature_seq_length): # can use different seq_length for the prediction
        data_sequences.append(data[i:(i + feature_seq_length)])
        target_vector.append(target[i + feature_seq_length])
    return np.array(data_sequences), np.array(target_vector)

# --- Train Model and Backtest
def create_model(X_train, optimizer=Adam(lr=0.001), lstm_layers=[50], dropout_rate=0.2):
    """
    Creates an LSTM model with specified hyperparameters.

    Args:
        X_train (numpy array): The training data to infer input shape for the LSTM layer.
        optimizer (keras.optimizers.Optimizer): The optimizer for the model. Default is Adam(lr=0.001).
        lstm_layers (list): A list of units for each LSTM layer. Default is [50].
        dropout_rate (float): Dropout rate between LSTM layers. Default is 0.2.

    Returns:
        keras.models.Model: The compiled Keras model.
    """
    model = Sequential()
    for i, units in enumerate(lstm_layers):
        if i == 0:
            model.add(LSTM(units, activation='relu', return_sequences=True if len(lstm_layers) > 1 else False, input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            model.add(LSTM(units, activation='relu', return_sequences=True if i < len(lstm_layers) - 1 else False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def hyperparameter_tuning(X_train, y_train, search_method='random'):
    """
    Performs hyperparameter tuning using Grid Search or Random Search.

    Args:
        X_train (numpy array): The input training data.
        y_train (numpy array): The target training data.
        search_method (str): The hyperparameter tuning method ('grid' for Grid Search, 'random' for Random Search). Default is 'grid'.

    Returns:
        dict: A dictionary containing the best score and best hyperparameters.
    """
    # Define the hyperparameter search space
    param_grid = {
        'optimizer': [Adam(lr=0.01)],
        'lstm_layers': [[50]],
        'dropout_rate': [0.4],
        'batch_size': [20],
        'epochs': [10]
    }

    # Wrap Keras model with KerasRegressor
    model = KerasRegressor(build_fn=create_model, verbose=0)

    # Perform Grid Search or Random Search
    if search_method == 'grid':
        tscv = TimeSeriesSplit(n_splits=3)
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    elif search_method == 'random':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5, cv=3, scoring='neg_mean_squared_error')

    search.fit(X_train, y_train)

    # Print the best hyperparameters
    best_score = search.best_score_
    best_params = search.best_params_
    print(f"Best score: {best_score} using {best_params}")

    return {'best_score': best_score, 'best_params': best_params}

def train_model(X_train, y_train, optimizer=Adam(lr=0.001), lstm_layers=[50], dropout_rate=0.2):
    """
    Trains the LSTM model with the given training data and parameters.

    Args:
        X_train (numpy array): The training data features.
        y_train (numpy array): The training data target variable.
        best_params (dict): The best parameters for the model training.

    Returns:
        tuple: The trained model and its history.
    """
    best_model = create_model(
        X_train,
        optimizer=optimizer,
        lstm_layers=lstm_layers,
        dropout_rate=dropout_rate
    )

    history = best_model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=20,
        verbose=0
    )

    return best_model, history

def plot_prediction(symbol, start_date, end_date, y_actual, y_predicted, title="Model Performance", xlabel="Time", ylabel="Returns"):
    """Plot actual values against predicted values.

    Args:
        y_actual (numpy array): The actual target values.
        y_predicted (numpy array): The predicted target values.
        title (str): The plot's title.
        xlabel (str): The x-axis label.
        ylabel (str): The y-axis label.
    """
    plt.figure(figsize=(7 , 4))
    plt.title(title)
    plt.plot(y_actual, label="Actual Prices")
    plt.plot(y_predicted, label="Predicted Prices")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    file_path = f"/var/www/html/static/images/{symbol}_{start_date}_{end_date}_{title}.png"
    plt.savefig(file_path)
    plt.show()
    plt.close()

def calculate_metrics(y_test, y_pred):
    """Calculate evaluation metrics.

    Args:
        y_test (numpy array): The actual target test values.
        y_pred (numpy array): The predicted target test values.

    Returns:
        dict: A dictionary with calculated metrics.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nMAPE: {mape}")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

def plot_loss(history, symbol, start_date, end_date):
    """Plot the training loss over epochs.

    Args:
        history (History): The history object from the trained model.
    """
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    file_path = f"/var/www/html/static/images/{symbol}_{start_date}_{end_date}_Loss.png"
    plt.savefig(file_path)
    plt.show()
    plt.close()

def predict_price_on_date(date, features, X_test, model, scaler, print_details=False, train_size=0.8, sequence_length=15):

    # Convert the input date to a timezone-aware object
    tz = pytz.timezone('UTC')
    date = pd.to_datetime(date).tz_localize(features.index.tz)
    print('date transformed to :', date)

    # Find the index of the given date
    date_index = features.index.get_loc(date)
    print(date_index)

    # Check if there is enough historical data for the given date
    if date_index < sequence_length:
        print("Not enough historical data for the given date.")
        return None

    # Calculate the index for X_test and check if it's within the valid range
    x_test_index = date_index - train_size - sequence_length
    if x_test_index < 0 or x_test_index >= len(X_test):
        print(f"No data available for the date {date}.")
        print(date_index - train_size - sequence_length, date_index, train_size, sequence_length, len(X_test))
        return None

    # Assigning the matching record of X_test to our target date
    row = X_test[x_test_index]

    # Reshape the input data to match the input shape required by the model
    input_data = row[None, :, :]

    # Predict the return for the given date using the model
    predicted_return = model.predict(input_data, verbose=0)

    # Unscale the return using the appropriate scaler.inverse_transform method
    unscaled_return = scaler.inverse_transform(predicted_return)

    # Calculate the predicted price for the given date
    previous_close_price = features.iloc[date_index - 1]["Close"]
    predicted_price = (1 + unscaled_return) * features.iloc[-1]['Close']

    # Calculate the actual price for the given date
    actual_price = features.iloc[date_index]["Close"]

    # Calculate the prediction error
    error = actual_price - predicted_price

    # Calculate the actual and predicted directions
    actual_direction = "up" if actual_price > previous_close_price else ("down" if actual_price < previous_close_price else "none")
    predicted_direction = "up" if predicted_price > previous_close_price else ("down" if predicted_price < previous_close_price else "none")

    # Print the results
    if print_details:
      print(f"Date: {date}")
      print(f"Predicted price: {predicted_price[0][0]:.5f}")
      print(f"Actual price: {actual_price:.5f}")
      print(f"Error: {error[0][0]:.5f}")
      print(f"Actual direction: {actual_direction}")
      print(f"Predicted direction: {predicted_direction}")

    return predicted_price[0][0], actual_price, error[0][0], actual_direction, predicted_direction

def is_holiday_or_weekend(date):
    # Get US holidays
    us_holidays = holidays.US()
    # Convert the input date to a pandas Timestamp object
    pd_date = pd.Timestamp(date)

    # Check if date is a public holiday
    is_public_holiday = pd_date in us_holidays

    # Check if date is a weekend (Saturday or Sunday)
    is_weekend = pd_date.weekday() in (5, 6)

    # Return True if date is either a public holiday or a weekend
    return is_public_holiday or is_weekend

    is_holiday = date in us_holidays
    print(date, "date is holiday or weekend?: ", is_public_holiday, is_weekend)
    return is_holiday

def calculate_win_rate(predictions):
    total_predictions = len(predictions)
    correct_predictions = sum([1 for prediction in predictions if prediction[3] == prediction[4]])
    win_rate = correct_predictions / total_predictions
    return win_rate

def calculate_reward_risk_ratio(errors):
    gains = [error for error in errors if error > 0]
    losses = [error for error in errors if error <= 0]
    reward_risk_ratio = np.mean(gains) / np.abs(np.mean(losses))
    return reward_risk_ratio

def calculate_sharpe_ratio(errors, risk_free_rate=0.01):
    excess_returns = np.array(errors) - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

def calculate_alpha(errors, benchmark_return=0.01):
    actual_return = np.mean(errors)
    alpha = actual_return - benchmark_return
    return alpha

def calculate_max_drawdown(errors):
    cumulative_returns = np.cumsum(errors)
    max_return = np.maximum.accumulate(cumulative_returns)
    drawdown = (max_return - cumulative_returns) / max_return
    max_drawdown = np.max(drawdown)
    return max_drawdown

def plot_errors(symbol, start_date, end_date, errors):
    plt.plot(errors, marker='o')
    plt.title("Prediction Errors")
    plt.xlabel("Days")
    plt.ylabel("Error")
    file_path = f"/var/www/html/static/images/{symbol}_{start_date}_{end_date}_Erros.png"
    plt.savefig(file_path)
    plt.close()

def is_date_between(date, start_date, end_date):
    # Convert the input strings to datetime objects
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    # Check if the date is between the start_date and end_date (inclusive)
    return start_date <= date <= end_date

def get_index(date, features, start_date, end_date):

    if is_date_between(date, start_date, end_date):
        # print(f"{date} is between {start_date} and {end_date}")
        pass
    else:
        print(f"{date} is NOT between {start_date} and {end_date}")
        return -1

    # Convert the input date to a timezone-aware object
    tz = pytz.timezone('UTC')
    date = pd.to_datetime(date).tz_localize(features.index.tz)
    # print('date transformed to :', date)

    # Find the index of the given date
    date_index = features.index.get_loc(date)
    return date_index

def predict_on_date (date, start_date, end_date, features, features_norm, sequence_length, model, scaler, train_size=0.8, print_details=False):

  start_date = features.index[0].strftime('%Y-%m-%d')
  end_date = features.index[-1].strftime('%Y-%m-%d')

  date_index = get_index(date, features, start_date, end_date)
  if date_index < sequence_length:
    # print('too soon to predict dear sir! you need at least a sequence to predict!')
    return None


  input_for_prediction = features_norm [date_index - sequence_length : date_index]
  input_for_prediction = input_for_prediction [None, :, :]

  predicted_return = model.predict(input_for_prediction, verbose=0)

  # Unscale the return using the appropriate scaler.inverse_transform method
  unscaled_return = scaler.inverse_transform(predicted_return)

  # Calculate the predicted price for the given date
  previous_close_price = features.iloc[date_index - 1]["Close"]
  predicted_price = (1 + unscaled_return) * features.iloc[-1]['Close']

  # Calculate the actual price for the given date
  actual_price = features.iloc[date_index]["Close"]

  # Calculate the prediction error
  error = actual_price - predicted_price

  # Calculate the actual and predicted directions
  actual_direction = "up" if actual_price > previous_close_price else ("down" if actual_price < previous_close_price else "none")
  predicted_direction = "up" if predicted_price > previous_close_price else ("down" if predicted_price < previous_close_price else "none")

  # Print the results
  if print_details:
    print(f"Date: {date}")
    print(f"Predicted price: {predicted_price[0][0]:.5f}")
    print(f"Actual price: {actual_price:.5f}")
    print(f"Error: {error[0][0]:.5f}")
    print(f"Actual direction: {actual_direction}")
    print(f"Predicted direction: {predicted_direction}")

  return predicted_price[0][0], actual_price, error[0][0], actual_direction, predicted_direction

def backtest(symbol, start_date, end_date, features, features_norm, sequence_length, model, scaler, plot_error_flag=True):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    results = []
    errors = []
    print('lezgo')
    for date in pd.date_range(start_date, end_date):
        if not is_holiday_or_weekend(date):
            prediction = predict_on_date(date.strftime('%Y-%m-%d'), start_date, end_date, features, features_norm, sequence_length, model, scaler, sequence_length)
            if prediction is not None:
                predicted_price, actual_price, error, actual_direction, predicted_direction = prediction
                results.append(prediction)
                errors.append(error)

    # Calculate performance metrics
    win_rate = calculate_win_rate(results)
    reward_risk_ratio = calculate_reward_risk_ratio(errors)
    sharpe_ratio = calculate_sharpe_ratio(errors)
    alpha = calculate_alpha(errors)
    max_drawdown = calculate_max_drawdown(errors)

    # Print performance metrics
    print(f"Win Rate: {win_rate}")
    print(f"Reward/Risk Ratio: {reward_risk_ratio}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Alpha: {alpha}")
    print(f"Maximum Drawdown: {max_drawdown}")

    #plot errors
    if plot_error_flag:
      plot_errors(symbol, start_date, end_date, errors)

    return results, win_rate, reward_risk_ratio
