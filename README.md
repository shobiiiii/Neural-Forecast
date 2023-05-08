# Stock Price Prediction with LSTM
This project aims to predict stock prices using an LSTM (Long Short-Term Memory) model. We use historical stock price data and technical indicators to train our LSTM model to predict future price movements.

## Features
The model takes the following features as input:

- Close
- High
- Low
- Volume (for stocks only)
- RSI (Relative Strength Index) for periods: 3, 7, 14, and 21
- ATR (Average True Range) for periods: 3, 7, 14, and 21
- CCI (Commodity Channel Index) for periods: 3, 7, 14, and 21
- SMA (Simple Moving Average) for periods: 3, 7, 14, and 21
- EMA (Exponential Moving Average) for periods: 3, 7, 14, and 21
- OBV (On-Balance Volume) (for stocks only)
- Day of the week (as a categorical variable)
- Month of the year (as a categorical variable)
- Quarter (as a categorical variable)


## Dependencies
To run this project, you'll need the following Python libraries:

- pandas
- numpy
- scikit-learn
- tensorflow
- keras
- yfinance
- seaborn
- matplotlib
- ta
- holidays
- pytz

## Backtest Results

Visit our website (coming soon) to access backtest results for the model across various time periods. We are actively developing the project and continuously updating the backtest results and enhancing the model.

On our website, you'll be able to customize the following parameters:

- Ticker symbol
- Start and end dates for data retrieval

Upcoming features:

- Sequence length (number of past time steps used for prediction)
- Hyperparameters for the LSTM model

## License

This project is licensed under the GNU General Public License v3.0. This means that you are allowed to use, modify, and distribute the code, as long as any derivative works or applications that use the code are also released under the same GPL license. This ensures that any changes or improvements made to the code remain open-source. For more information, please see the [LICENSE](LICENSE) file.
