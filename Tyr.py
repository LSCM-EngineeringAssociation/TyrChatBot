import os
import json
import openai
import pyttsx3
import uuid
import webbrowser
from flask import Flask, request, jsonify, send_file, render_template
import speech_recognition as sr
from enum import Enum

class Personality(Enum):
    SAFE = 1
    NORMAL = 2
    HAPPY = 3
    SATIRICAL = 4
    HIGHLY_SATIRICAL = 5

openai.api_key = "Enter api"
global current_temperature
global current_personality

current_temperature = 0.7
current_personality = Personality.NORMAL

Tyr = Flask(__name__)

# Set Tyr personality
with open("prompts.json", "r") as f:
    data = json.load(f)

messages = [
    {"role": "system", "content": data["Prompts"]["Safe"]},
    {"role": "system", "content": data["Prompts"]["Normal"]},
    {"role": "system", "content": data["Prompts"]["Happy"]},
    {"role": "system", "content": data["Prompts"]["Satirical"]},
    {"role": "system", "content": data["Prompts"]["Highly_Satirical"]},
    {"role": "system", "content": data["Prompts"]["Joe_Rogan"]}
]


# Auhtenticate the API key
def test_api_key(api_key):
    openai.api_key = api_key
    test = openai.Completion.create(
        engine="text-ada-001", prompt="test", max_tokens=5, n=1)
    if 'Invalid API key' in str(test):
        return False
    else:
        return True


# Generate response with personality 1
def generate_response(input: str, personality: Personality):
    if input:
        for i in range(6):
            messages[i].append({"role": "user", "content": input})

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages[current_personality], temperature=current_temperature
        )
        reply = chat.choices[current_personality].message.content
        for i in range(6):
            messages[i].append({"role": "assistant", "content": reply})

        return reply


# Function that Listens to user using Whisper
def transcribe_audio(filename: str) -> str:
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text


# Function that Listens to user locally using Google Speech Recognition
def listen_to_user():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        user_message = r.recognize_google(audio, language='en-US')
        print("You: " + user_message)
        return user_message
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Error requesting results; {0}".format(e))
        return None


# Create function to handle "Voice Ask" button click
def TyrChat():
    engine = pyttsx3.init()
    while True:
        user_message = listen_to_user()
        if user_message is None:
            continue
        if user_message.lower() in ["quit", "bye", "exit"]:
            print("Tyr: Goodbye!")
            exit()
        prompt = (f"{user_message}")
        chatbot_response = generate_response(prompt)
        engine.say(chatbot_response)
        engine.runAndWait()

# ---------------  FLASK FUNCTIONS  ---------------
@Tyr.route('/')
def index():
    # Render the index page.
    return render_template('TyrPage.html', voices=Tyr)


@Tyr.route('/transcribe', methods=['POST'])
def transcribe():
    # Transcribe the given audio to text using Whisper.
    if 'file' not in request.files:
        return 'No file found', 400
    file = request.files['file']
    recording_file = f"{uuid.uuid4()}.wav"
    recording_path = f"uploads/{recording_file}"
    os.makedirs(os.path.dirname(recording_path), exist_ok=True)
    file.save(recording_path)
    transcription = transcribe_audio(recording_path)
    return jsonify({'text': transcription})


@Tyr.route('/ask', methods=['POST'])
def ask():
    # Get the selected personality from the request data
    personality_text = str(request.get_json(force=True).get("personality", ""))
    current_personality = Personality(int(personality_text))
    conversation = str(request.get_json(force=True).get("conversation", ""))
    print("Received data:", conversation, current_personality)
    # Call the appropriate function based on the selected personality
    reply = generate_response(conversation, current_personality)

    print("Generated response:", reply)
    return jsonify({'text': reply})


@Tyr.route('/update-api-key', methods=['POST'])
def update_api_key():
    new_key = request.get_json(force=True).get("api_key", "")
    if new_key and test_api_key(new_key):
        openai.api_key = new_key
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error"})


@Tyr.route('/update_temperature', methods=['POST'])
def update_temperature():
    data = request.get_json()
    new_temperature = float(data.get('temperature', 0.5))
    current_temperature = new_temperature
    return jsonify({'temperature': current_temperature})


@Tyr.route('/listen/<filename>')
def listen(filename):
    # Return the audio file located at the given filename.
    return send_file(f"outputs/{filename}", mimetype="audio/mp3", as_attachment=False)


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    Tyr.run()
