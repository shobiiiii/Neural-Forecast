from flask import Flask, request

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form.get('data')
    print('Received data:', data)
    return 'Data received!'

if __name__ == '__main__':
    app.run(debug=True)
