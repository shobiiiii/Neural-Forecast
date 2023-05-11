from flask import Flask, request, jsonify
from flask_cors import CORS
from functions import fetch_data, plot_data, process_asset_symbol

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

    # Fetch Data
    data = fetch_data(symbol, start_date, end_date)

    # Plot the data
    plot_file_path = plot_data(symbol, data, start_date, end_date)
    image_url = request.url_root + plot_file_path

    return jsonify({'message': 'Data fetched and plotted successfully.', 'image_url': image_url}), 200

if __name__ == '__main__':
    app.run(debug=True)
