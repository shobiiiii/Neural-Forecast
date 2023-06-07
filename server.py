from flask import Flask, request, jsonify
from flask_cors import CORS
from functions import *

app = Flask(__name__)
CORS(app)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    print('Received data:', data)

    asset = data['asset']
    asset_type = data['assetType']
    start_date = data['startDate']
    end_date = data['endDate']

    # Transform the asset name to symbol
    symbol = process_asset_symbol(asset_type, asset)

    # extend the time window to acquire data for train
    print('user start and end dates are: ', start_date, end_date)
    start_date, end_date = extend_time_window(start_date, end_date)
    print('application dates are: ', start_date, end_date)

    # Fetch Data
    data = fetch_data(symbol, start_date, end_date)

    # Plot the data
    plot_file_path = plot_data(symbol, data, start_date, end_date)

    # Add featues
    data = add_features(data, asset_type)

    # Select relevant features
    features, target = select_features(data, asset_type)

    # Normalize data
    features_norm, target_norm, scaler = normalize_data(features, target)
    # print("---features_norm[-5:]---\n",features_norm[-5:])
    # print("\n\n\n---Returns[-5:]---\n",target_norm[-5:])

    # split the data
    features_norm_train, features_norm_test, target_norm_train, target_norm_test = split_data(features_norm, target_norm)

    # define sequence length
    seq_length = 3

    # create sequences for train and test features and target
    features_norm_train_seq, target_norm_train_vec = create_sequences(features_norm_train, target_norm_train, seq_length)
    features_norm_test_seq, target_norm_test_vec = create_sequences(features_norm_test, target_norm_test, seq_length)
    # print("\n\n\n---features_norm_test_seq[-1:]---\n", features_norm_test_seq[-1], "\n\n\n---target_norm_test_vec[-1:]---\n", target_norm_test_vec[-1])

    # Train the model
    model, history = train_model(features_norm_train_seq, target_norm_train_vec)

    # Predict the train data and plot the result
    # Y_train_pred = model.predict(features_norm_train_seq, verbose=0)
    # plot_prediction(symbol, start_date, end_date, target_norm_train_vec, Y_train_pred, title="Performance of model over trained data")

    # Predict the test data and plot the result
    y_pred = model.predict(features_norm_test_seq, verbose=0)
    plot_prediction(symbol, start_date, end_date, target_norm_test_vec, y_pred, title="Performance of model over test data")

    # calculate statistical metrics
    # metrics = calculate_metrics(target_norm_test_vec, y_pred)

    # Plot the training loss
    plot_loss(history, symbol, start_date, end_date)

    # backtest the strategy
    results, win_rate, reward_risk = backtest(symbol, start_date, end_date, features, features_norm, seq_length, model, scaler)
    image_url = f"/static/images/{symbol}_{start_date}_{end_date}_Performance of model over test data.png"

    return jsonify({'message': 'finished!', 'image_url': image_url, 'win_rate': float(win_rate), 'reward_risk': float(reward_risk) }), 200

if __name__ == '__main__':
    app.run(debug=True)
