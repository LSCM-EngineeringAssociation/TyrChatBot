import requests
import serial
import time
from pygame import mixer

CHUNK_SIZE = 1024
url = "https://api.elevenlabs.io/v1/text-to-speech/BPEBxITmwVdxpLyK0AMA/stream"

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "4753f56971761bc63e359fe255282b97"
}

data = {
    "text": "As an AI language model, I cannot directly solve or perform experiments to achieve cold fusion. However, I can provide you with information and help you understand the challenges and current state of research in the field. Cold fusion, also referred to as low-energy nuclear reactions (LENR), is a hypothetical form of nuclear reaction that occurs at or near room temperature. In contrast, conventional fusion reactions require extremely high temperatures and pressures, such as those found in the core of the sun. If cold fusion were possible, it could potentially provide a nearly limitless, clean, and safe source of energy. The concept of cold fusion gained significant attention in 1989 when Martin Fleischmann and Stanley Pons claimed to have observed it in a laboratory experiment. However, their findings were met with skepticism, and many subsequent attempts to replicate their results were unsuccessful. Despite the skepticism and controversy surrounding cold fusion, there are still some researchers investigating the phenomenon. They aim to understand the underlying mechanisms and explore novel materials and experimental techniques that might enable cold fusion. However, as of my last knowledge update in September 2021, no conclusive or widely accepted evidence of cold fusion has been found, and it remains a controversial and elusive area of research. If you have specific questions about cold fusion, I'd be happy to help answer them or provide further information.",
    "voice_settings": {
        "stability": 0,
        "similarity_boost": 0
    }
}

response = requests.post(url, json=data, headers=headers, stream=True)

with open('output.mp3', 'wb') as f:
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            f.write(chunk)

mixer.init()
mixer.music.load('output.mp3')

ser = serial.Serial('COM6', 9600)  # Change 'COM3' to the correct port for your Arduino
time.sleep(2)  # Wait for the Arduino to initialize

mixer.music.play()

while mixer.music.get_busy():
    ser.write(b'0')  # Jaw closed
    time.sleep(0.1)
    ser.write(b'180')  # Jaw open
    time.sleep(0.1)

ser.close()
