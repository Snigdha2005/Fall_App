from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import requests
import os
from twilio.rest import Client
from .models import Caretaker, Patient, FallHistory

deepsense_model = load_model('deepsense_model.h5')
xgb_model = joblib.load('xgboost_model.pkl')

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
twilio_phone_number = os.environ["TWILIO_PHONE_NUMBER"]
client = Client(account_sid, auth_token)

def evaluate_fall(request):
    if request.method == 'POST':
        data = request.POST.get('sensor_data')
        patient_phone = request.POST.get('patient_phone')
        if not data or not patient_phone:
            return JsonResponse({'status': 'error', 'message': 'Missing data'}, status=400)

        try:
            # Convert sensor data to numpy array
            sensor_data = np.array(data).reshape((1, -1, 1))  # Assuming data is provided in the correct shape

            # Predict with DeepSense model
            deep_sense_predictions = deepsense_model.predict(sensor_data)

            # Predict with XGBoost model
            xgb_predictions = xgb_model.predict(deep_sense_predictions)
            fall_detected = xgb_predictions[0] == 1

            if fall_detected:
                response = send_fall_alert(request, patient_phone)
                FallHistory.objects.create(patient=Patient.objects.get(phone_number=patient_phone), timestamp=timezone.now())
                return JsonResponse({'status': 'success', 'message': 'Fall detected and alert sent'})
            else:
                return JsonResponse({'status': 'success', 'message': 'No fall detected'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def get_client_ip(request):
    """Get the client's IP address from the request."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def location_coordinates(ip):
    """Get location coordinates based on the provided IP address."""
    try:
        response = requests.get(f'https://ipinfo.io/{ip}/json')
        data = response.json()
        loc = data['loc'].split(',')
        lat, long = float(loc[0]), float(loc[1])
        city = data.get('city', 'Unknown')
        state = data.get('region', 'Unknown')
        return lat, long, city, state
    except Exception as e:
        print(f"Error obtaining location data: {e}")
        return None, None, None, None

def send_fall_alert(request, patient_phone):
    if request.method == 'POST':
        try:
            patient = Patient.objects.get(phone_number=patient_phone)
            caretaker_phone = patient.caretaker.caretaker_phone_no
            patient_name = patient.patient_name
        except Patient.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Patient not found'}, status=404)

        client_ip = get_client_ip(request)
        lat, long, city, state = location_coordinates(client_ip)
        if lat is None or long is None:
            return JsonResponse({'status': 'error', 'message': 'Unable to determine location'}, status=500)

        gps_location = f"Latitude: {lat}, Longitude: {long} in {city}, {state}"

        message_body = f"Fall detected for patient {patient_name} at location: {gps_location}. Please check immediately."

        message = client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=caretaker_phone,
        )

        return JsonResponse({'status': 'success', 'message': message.sid})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
