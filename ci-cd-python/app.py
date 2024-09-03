from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello from ci/cd powered Flask app! V2"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')