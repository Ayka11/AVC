import os
import time
import base64
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
from colorsys import rgb_to_hsv
from scipy.io.wavfile import write as write_wav

app = Flask(__name__)
OUTPUT_DIR = "static/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 44100
DURATION_PER_STEP = 0.3  # seconds per X step (column)

NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
    'E': 4, 'F': 5, 'F#': 6, 'G': 7,
    'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
note_names = list(NOTE_TO_SEMITONE.keys())

def hue_to_note_name(hue):
    index = int((hue % 360) / 30)
    return note_names[index]

def brightness_to_octave(brightness):
    return int(3 + brightness * 3)

def color_to_frequency(r, g, b):
    h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)
    hue_deg = h * 360
    note_name = hue_to_note_name(hue_deg)
    octave = brightness_to_octave(v)
    midi_note = 12 + octave * 12 + NOTE_TO_SEMITONE[note_name]
    return 440 * 2 ** ((midi_note - 69) / 12)

def generate_tone(frequencies, duration=DURATION_PER_STEP):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    waveform = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    waveform /= max(np.abs(waveform)) if np.max(np.abs(waveform)) != 0 else 1
    return waveform

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data['image'].split(',')[1]
    img = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGBA')
    width, height = img.size
    pixels = img.load()

    # Build a map: x → list of (frequency at y)
    timeline = {}
    for x in range(width):
        freqs = []
        for y in range(height):
            r, g, b, a = pixels[x, y]
            if a > 10:  # ignore transparent / empty pixels
                freq = color_to_frequency(r, g, b)
                freqs.append(freq)
        if freqs:
            timeline[x] = freqs

    # Generate sound from left to right (time axis)
    audio_segments = [generate_tone(timeline[x]) for x in sorted(timeline.keys())]
    audio = np.concatenate(audio_segments)
    audio_int16 = np.int16(audio * 32767)

    filename = f"sound_{int(time.time() * 1000)}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    write_wav(filepath, SAMPLE_RATE, audio_int16)

    return jsonify({"url": f"/static/audio/{filename}"})

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
