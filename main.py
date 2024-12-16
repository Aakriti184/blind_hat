import torch
import cv2
import numpy as np
import pyttsx3
import requests
import geopy
from geopy.exc import GeocoderTimedOut
from geopy.distance import geodesic
import speech_recognition as sr
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to convert text to speech
def say_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to get voice input
def get_voice_input(prompt):
    recognizer = sr.Recognizer()
    say_text(prompt)
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)  # Adjust for background noise
        try:
            print("Listening...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Allow more time for input
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            say_text("Sorry, I couldn't understand. Please try again.")
            return get_voice_input(prompt)
        except sr.RequestError:
            say_text("Speech recognition service is not available.")
            return None
        except sr.WaitTimeoutError:
            say_text("I didn't hear anything. Please try again.")
            return None

# Function to convert location to coordinates
def get_coordinates(location_name):
    geolocator = geopy.geocoders.Nominatim(user_agent="blind_hat")
    try:
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        else:
            say_text("Location not found. Please try again.")
            return None
    except GeocoderTimedOut:
        say_text("Geocoding service timed out. Please try again.")
        return None

# Function to speak navigation directions (with a delay)
def say_directions(directions):
    for direction in directions:
        say_text(direction)
        time.sleep(3)  # Add a delay between each direction

# Function to get directions from OpenRouteService
def get_navigation_route(current_coords, destination_coords, api_key):
    url = f'https://api.openrouteservice.org/v2/directions/driving-car?'
    params = {
        'api_key': api_key,
        'start': f"{current_coords[1]},{current_coords[0]}",
        'end': f"{destination_coords[1]},{destination_coords[0]}"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'features' in data and len(data['features']) > 0:
            route = data['features'][0]
            if 'properties' in route and 'segments' in route['properties'] and len(route['properties']['segments']) > 0:
                steps = route['properties']['segments'][0]['steps']
                directions = [step['instruction'] for step in steps]
                return directions
            else:
                say_text("No route segments found.")
                return []
        else:
            say_text("No routes found or invalid response from the routing service.")
            return []
    except requests.exceptions.RequestException as e:
        say_text("Failed to retrieve navigation data.")
        return []

# Initialize YOLO model for obstacle detection
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
except Exception as e:
    say_text("Object detection model failed to load.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    say_text("Unable to access the camera.")
    exit()

# Replace with actual API key for OpenRouteService
ORS_API_KEY = '5b3ce3597851110001cf6248d3a64af6f5f34b9dad74b188c5a74f07'

# Get starting and destination coordinates by voice input
start_location = get_voice_input("Please say your starting location.")
if start_location:
    current_coords = get_coordinates(start_location)

destination_location = get_voice_input("Please say your destination location.")
if destination_location:
    destination_coords = get_coordinates(destination_location)

if current_coords and destination_coords:
    navigation_directions = get_navigation_route(current_coords, destination_coords, ORS_API_KEY)
    if navigation_directions:
        say_directions(navigation_directions)

# Start video stream and obstacle detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Detect objects in the video feed
    results = model(frame)
    if len(results.pandas().xyxy[0]) > 0:
        detected_objects = results.pandas().xyxy[0]['name'].unique().tolist()
        if detected_objects:
            for obj in detected_objects:
                say_text(f"{obj} ahead")

    # Render detection results on the frame
    frame = np.squeeze(results.render())

    # Display the frame
    cv2.imshow('Blind Hat', frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
