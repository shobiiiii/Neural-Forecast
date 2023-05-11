import os
import yfinance as yf
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



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
        pass  # No transformation needed for stocks
    return symbol

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
        hist_data = pd.read_csv(csv_filepath)
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

    file_path = f"static/images/{symbol}_{start_date}_{end_date}.png"
    plt.savefig(file_path)
    plt.close(fig)

    return file_path

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
        print("CSV File Already Exist.")
        return
    else:
        # Save the data to a CSV file
        data.to_csv(file_path)
