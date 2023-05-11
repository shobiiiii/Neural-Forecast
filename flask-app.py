from flask import Flask, request, jsonify
from flask_cors import CORS
from functions import fetch_data, plot_data

app = Flask(__name__)
CORS(app)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    print('Received data:', data)
    # temporarily add '-USD', need to check the asset type then transform properly
    asset = data['asset']+'-USD'
    start_date = data['startDate']
    end_date = data['endDate']
    data = fetch_data('cypto', asset, start_date, end_date)
    plot_file_path = plot_data(asset, data, start_date, end_date)
    image_url = request.url_root + plot_file_path
    return jsonify({'message': 'Data fetched and plotted successfully.', 'image_url': image_url}), 200

if __name__ == '__main__':
    app.run(debug=True)
