import os
import json
import openai
import io
import traceback
import threading
import serial
from threading import Thread
from serial import Serial, SerialException
from serial.tools import list_ports
from time import sleep
import numpy as np
import pygame
from pydub import AudioSegment
from pydub.utils import mediainfo
from dotenv import load_dotenv, set_key
import requests
import webbrowser
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, render_template
import tyr_util as tyr
from enum import IntEnum

# Flask Controls
Tyr = Flask(__name__)
Tyr.config['UPLOAD_FOLDER'] = 'static/data'
ALLOWED_EXTENSIONS = {'pdf', 'mp3'}

class Personality(IntEnum):
    NORMAL = 0
    SAFE = 1
    HAPPY = 2
    SATIRICAL = 3
    Dan = 4
    WRITING_HELPER = 5
    JOE_ROGAN = 6

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
elabs_apikey = os.getenv("elabs_apikey")
elabs_voiceID = os.getenv("elabs_voice")
global current_temperature
global current_personality
global file_embeddings

# OpenAI Controls
current_temperature = 0.7
current_personality = Personality.NORMAL
file_embeddings = None

# Arduino Controls
# Set up the serial connection with your Arduino
""" def find_and_connect_serial():
    for com_port in list_ports.comports():
        try:
            ser = Serial(com_port.device, 9600)
            print(f"Connected to {com_port.device}")
            return ser
        except SerialException:
            pass
            print("Error: COM port not found.") """

arduino_serial = serial.Serial('COM6', 9600)

# ---------------  ARDUINO FUNCTIONS  ---------------
def map_amplitude_to_angle(amplitude, min_angle, max_angle):
    min_amplitude = 0
    max_amplitude = 32767
    angle = (amplitude - min_amplitude) * (max_angle - min_angle) / (max_amplitude - min_amplitude) + min_angle
    return int(angle)

def process_audio(audio_file, duration, arduino_serial):
    CHUNK_SIZE_MS = 10  # The size of each chunk in milliseconds
    MIN_ANGLE = 0  # The minimum angle for the jaw servo
    MAX_ANGLE = 180  # The maximum angle for the jaw servo

    audio = AudioSegment.from_file(audio_file, format='mp3')

    for i in range(0, duration, CHUNK_SIZE_MS):
        chunk = audio[i:i + CHUNK_SIZE_MS]
        amplitude = np.mean(np.abs(np.frombuffer(chunk.raw_data, np.int16)))
        angle = map_amplitude_to_angle(amplitude, MIN_ANGLE, MAX_ANGLE)

        # Send the angle to the Arduino
        arduino_serial.write(str(angle).encode() + b'\n')
        sleep(CHUNK_SIZE_MS / 1000)

def play_audio_file(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# ---------------  OPENAI FUNCTIONS  ---------------
# Set Tyr personality
with open("static/prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

messages = [
    [{"role": "system", "content": data["Prompts"]["Normal"]}],
    [{"role": "system", "content": data["Prompts"]["Safe"]}],
    [{"role": "system", "content": data["Prompts"]["Happy"]}],
    [{"role": "system", "content": data["Prompts"]["Satirical"]}],
    [{"role": "system", "content": data["Prompts"]["Dan"]}],
    [{"role": "system", "content": data["Prompts"]["WRITING_HELPER"]}],
    [{"role": "system", "content": data["Prompts"]["Joe_Rogan"]}]
]

# Auhtenticate the API key
def test_oai_key(api_key):
    url = "https://api.openai.com/v1/engines"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True
    else:
        return False
    
def test_elab_key(api_key):
    url = "https://api.elevenlabs.io/v1/user"
    headers = {
        "accept": "application/json",
        "xi-api-key": api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True
    else:
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Generate response with personality 1
def generate_response(input: str, personality: Personality):
    if input:
        for i in range(len(messages)):
            messages[i].append({"role": "user", "content": input})

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages[int(personality)], temperature=current_temperature
        )
        reply = chat.choices[0].message.content
        for i in range(len(messages)):
            messages[i].append({"role": "assistant", "content": reply})

        return reply   

# ---------------  FLASK FUNCTIONS  ---------------
@Tyr.route('/')
def index():
    # Render the index page.
    return render_template('TyrPage.html')

@Tyr.route('/ask', methods=['POST'])
def ask():
    global current_personality
    conversation = str(request.get_json(force=True).get("conversation", ""))
    print("Received data:", conversation, current_personality.name)
    # Call the appropriate function based on the selected personality
    reply = generate_response(conversation, current_personality)

    print("Generated response:", reply)

    return jsonify({'text': reply})

@Tyr.route('/process_file', methods=['POST'])
def process_file():
    global file_embeddings
    file = request.files.get('file')
    # Save the file to the /static/data folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(Tyr.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Process the file using store_doc_embeds function
    file_embeddings = tyr.getDocEmbeds(filepath)
    print("done")
    return jsonify(success=True)

@Tyr.route('/get_answer', methods=['POST'])
def get_answer():
    global file_embeddings
    query = str(request.get_json(force=True).get("conversation", ""))
    history = []
    if file_embeddings:
        answer = tyr.conversational_chat(file_embeddings, query, history)
        history.append(query, answer)
        return jsonify({'text': answer})
    else:
        return jsonify(error='No file has been processed yet.')

@Tyr.route('/update-openai-api-key', methods=['POST'])
def update_openai_api_key():
    new_key = request.get_json(force=True).get("api_key", "")
    if new_key and test_oai_key(new_key):
        openai.api_key = new_key
        set_key(".env", "OPENAI_API_KEY", new_key)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error"})

@Tyr.route('/update-elevenlabs-api-key', methods=['POST'])
def update_elevenlabs_api_key():
    new_key = request.get_json(force=True).get("api_key", "")
    if new_key and test_elab_key(new_key):
        os.environ["elabs_apikey"] = new_key
        set_key(".env", "elabs_apikey", new_key)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error"})

@Tyr.route('/update_temperature', methods=['POST'])
def update_temperature():
    data = request.get_json()
    new_temperature = float(data.get('temperature', 0.7))
    current_temperature = new_temperature
    return jsonify({'temperature': current_temperature})

@Tyr.route('/update_personality', methods=['POST'])
def update_personality():
    global current_personality
    try:
        personality_number = int(request.json['personality'])
        current_personality = Personality(personality_number)
        print(current_personality.name)
        return jsonify(personality=current_personality.name), 200
    except ValueError:
        print("Invalid personality number")
        return jsonify(error="Invalid personality number"), 400

@Tyr.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    ELABS_STAB = 0.70
    ELABS_SIMIL = 0.75
    try:
        text = request.get_json(force=True).get("text","")
        voice_id = elabs_voiceID # Replace with your desired voice ID
        api_key = elabs_apikey # Replace with your Eleven Labs API key
        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': api_key,
            'Content-Type': 'application/json'
        }
        data = {
            "text": text,
            "voice_settings": {
                "stability": ELABS_STAB,
                "similarity_boost": ELABS_SIMIL
            }
        }
        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            with open('temp_audio.mp3', 'wb') as f:
                f.write(response.content)

            audio_data = 'temp_audio.mp3'

            # Get the duration of the audio file
            audio = AudioSegment.from_file(audio_data, format='mp3')
            duration = int(len(audio) / 1000)

            play_audio_thread = threading.Thread(target=play_audio_file, args=(audio_data,))
            process_audio_thread = threading.Thread(target=process_audio, args=(audio_data, duration, arduino_serial))
            try:
                # Create a thread to play the audio
                play_audio_thread.start()
                # Create a thread to send jaw movement commands to the Arduino
                process_audio_thread.start()
                # Wait for both threads to finish
                play_audio_thread.join()
                process_audio_thread.join()
            finally: 
                #os.remove('temp_audio.mp3')
                return send_file(audio_data, mimetype='audio/mpeg', as_attachment=True, download_name='response.mp3')

        else:
            print(f"Error in text_to_speech: Eleven Labs API returned {response.status_code} - {response.text}")
            return jsonify({"status": "error", "text": "Text-to-speech conversion failed"}), response.status_code
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        traceback.print_exc()
        print(request.get_json())
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    Tyr.run()