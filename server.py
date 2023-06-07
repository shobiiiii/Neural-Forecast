from flask import Flask, request, jsonify
from flask_cors import CORS
import functions

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
    symbol = functions.process_asset_symbol(asset_type, asset)

    # Fetch Data
    data = functions.fetch_data(symbol, start_date, end_date)

    # Plot the data
    plot_file_path = functions.plot_data(symbol, data, start_date, end_date)
    image_url = request.url_root + plot_file_path

    # Add Technical indicators
    functions.add_technical_indicators(data, asset_type)

    # Select relevant features
    features, target = functions.select_features(data, asset_type)

    return jsonify({'message': 'technical added.', 'image_url': image_url}), 200

if __name__ == '__main__':
    app.run(debug=True)
