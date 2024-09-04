from flask import Flask, request, jsonify, render_template
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
import plotly
import json
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

# Load your trained model
model = joblib.load('airquality.joblib')

# OpenWeather API Key
API_KEY = '7d436901b0c9caab8af7164c3ddb1cb0'
MODEL_PATH = 'weather_model.pkl'
SCALER_PATH = 'scaler.pkl'
IMG_SIDEBAR_PATH = "./static/img.jpg"
DATA_PATH = "weather_dataset.csv"

def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

model1 = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

@app.route('/a')
def index():
    data = pd.read_csv(DATA_PATH)
    slider_data = {
        "precipitation": {"min": 0, "max": float(data["precipitation"].max()), "value": float(data["precipitation"].mean())},
        "temp_max": {"min": 0, "max": float(data["temp_max"].max()), "value": float(data["temp_max"].mean())},
        "temp_min": {"min": 0, "max": float(data["temp_min"].max()), "value": float(data["temp_min"].mean())},
        "wind": {"min": 0, "max": float(data["wind"].max()), "value": float(data["wind"].mean())}
    }
    return render_template('index.html', slider_data=slider_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([data['precipitation'], data['temp_max'], data['temp_min'], data['wind']]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    pred_result = int(model1.predict(input_data_scaled)[0])
    probabilities = model1.predict_proba(input_data_scaled)[0]
    response = {
        "pred_result": pred_result,
        "probabilities": {
            "drizzle": round(probabilities[0], 2),
            "rain": round(probabilities[1], 2),
            "sun": round(probabilities[2], 2),
            "snow": round(probabilities[3], 2),
            "fog": round(probabilities[4], 2),
        }
    }
    return jsonify(response)

@app.route('/radar_chart', methods=['POST'])
def radar_chart():
    data = request.json
    categories = ['Precipitation', 'Max Temperature', 'Min Temperature', 'Wind']
    radar_data = [data['precipitation'], data['temp_max'], data['temp_min'], data['wind']]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_data,
        theta=categories,
        fill='toself',
        name='Input Data'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

@app.route('/bar_chart')
def bar_chart():
    df = pd.read_csv("./assets/weather_classes.csv")
    fig = px.bar(df, x='Weather', y='Number of that Class', color='Weather', title="Weather Classes")
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

@app.route('/donut_chart')
def donut_chart():
    df = pd.read_csv("./assets/weather_classes.csv")
    fig = px.pie(df, values='Number of that Class', names='Weather', hole=0.3, title="Weather Classes")
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/predict_manually', methods=['POST','GET'])
def predict_manually():
    if request.method == 'POST':
        # Extract data from form
        pm25 = float(request.form['PM2.5'])
        pm10 = float(request.form['PM10'])
        o3 = float(request.form['O3'])
        no2 = float(request.form['NO2'])
        co = float(request.form['CO'])
        so2 = float(request.form['SO2'])

        # Prepare data for prediction
        sample = [[pm25, pm10, o3, no2, co, so2]]
        prediction = model.predict(sample)[0]

        # Determine Air Quality Index based on prediction
        result, conclusion = determine_air_quality(prediction)

        # Return the result to the user
        return render_template('results.html', prediction=prediction, result=result, conclusion=conclusion)
    else:
        return render_template('index1.html')
@app.route('/predict_automatically', methods=['GET', 'POST'])
def predict_automatically():
    if request.method == 'POST':
        city_name = request.form.get('city_name')
        if not city_name:
            error_message = "Missing city name parameter"
            error_code = 400
            return render_template('error.html', error=error_message ,error_code=error_code), 400

        # Geocoding API to get lat and lon from city name
        geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
        geocode_response = requests.get(geocode_url)
        if geocode_response.status_code != 200:
            error_message = "Failed to fetch location data"
            error_code = 500
            return render_template('error.html', error=error_message ,error_code=error_code), 500
        
        geocode_data = geocode_response.json()
        if not geocode_data:
            error_message = "City not found"
            error_code = 404
            return render_template('error.html', error=error_message ,error_code=error_code), 404


        # Assuming the first result is the most relevant
        lat = geocode_data[0]['lat']
        lon = geocode_data[0]['lon']

        # Now use lat and lon to get air pollution data
        air_quality_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        air_quality_response = requests.get(air_quality_url)
        if air_quality_response.status_code != 200:
            error_message = "Failed to fetch Air Quality Index data"
            error_code = 500
            return render_template('error.html', error=error_message ,error_code=error_code), 500


        air_quality_data = air_quality_response.json()['list'][0]['components']
        sample = [
            [air_quality_data['pm2_5'], air_quality_data['pm10'], air_quality_data['o3'],
             air_quality_data['no2'], air_quality_data['co'], air_quality_data['so2']]
        ]
        prediction = round(model.predict(sample)[0],2)

        result, conclusion = determine_air_quality(prediction)

        return render_template('results.html', prediction=prediction, result=result, conclusion=conclusion)

    else:
        return render_template('city.html')

def determine_air_quality(prediction):
    if prediction < 50:
        return 'Air Quality Index is Good', 'The Air Quality Index is excellent. It poses little or no risk to human health.'
    elif 51 <= prediction < 100:
        return 'Air Quality Index is Satisfactory', 'The Air Quality Index is satisfactory, but there may be a risk for sensitive individuals.'
    elif 101 <= prediction < 200:
        return 'Air Quality Index is Moderately Polluted', 'Moderate health risk for sensitive individuals.'
    elif 201 <= prediction < 300:
        return 'Air Quality Index is Poor', 'Health warnings of emergency conditions.'
    elif 301 <= prediction < 400:
        return 'Air Quality Index is Very Poor', 'Health alert: everyone may experience more serious health effects.'
    else:
        return 'Air Quality Index is Severe', 'Health warnings of emergency conditions. The entire population is more likely to be affected.'

if __name__ == '__main__':
    app.run(debug=True)
