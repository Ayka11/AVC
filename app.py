import matplotlib
matplotlib.use('Agg')
import colorsys
import time
from flask import Flask, render_template, request,session,send_from_directory,jsonify
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import os
import base64
from music_visualizer import process_audio_to_gif
from scipy.io.wavfile import read as wav_read # Add this line near other imports
from io import BytesIO
from PIL import Image
from colorsys import rgb_to_hsv
from scipy.io.wavfile import write as write_wav
from PIL import Image, ImageDraw, ImageFont
import string
from docx import Document
import PyPDF2
import pdfplumber
import soundfile as sf
#from sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.fft import fft
import io
import base64
import pandas as pd
# from pydub import AudioSegment
from werkzeug.utils import secure_filename
from dash import Dash, dcc, html
from dash import dcc, html
from dash import Dash, dcc, html, callback, Input, Output
import plotly.graph_objs as go
import plotly.express as px
global session
import numpy as np
from numpy.random import uniform
from scipy import signal


PIANO_SAMPLES = {}
SAMPLE_RATE = 44100
app = Flask(__name__)
app.secret_key = 'Lantop2333' # Set a secret key session
OUTPUT_DIR = "static/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Assuming DURATION_PER_STEP and SAMPLE_RATE are defined globally or passed in
# For demonstration, let's define them here:
# DURATION_PER_STEP = 1 # seconds
SAMPLE_RATE = 44100 # Hz
DURATION_PER_STEP = (60)/1000 # seconds per X step (column)
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
    'E': 4, 'F': 5, 'F#': 6, 'G': 7,
    'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
note_names = list(NOTE_TO_SEMITONE.keys())
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')
Fs = 44100 # Sampling frequency
# Frequency ranges and names of piano notes
freqs_org = [27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27,
         61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
         130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
         261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
         523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
         1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00,
         1864.66, 1975.53, 2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96,
         3322.44, 3520.00, 3729.31, 3951.07, 4186.01]
# Set colors for each frequency
colors = [[139/255, 0, 0]] * len(freqs_org)
sounds = {
'A': (0, 100, 0), # Dark Green+I1:I31
'B': (255, 127, 80), # Coral
'C': (210, 180, 140), # Light Brown
'Ç': (205, 170, 100), # Light golden brown
'D': (255, 165, 0), # Orange
'E': (0, 255, 255), # Light Blue
'Ə': (230, 230, 250), # Lavender
'F': (139, 0, 139), # Dark Magenta
'G': (255, 215, 0), # Gold
'Ğ': (184, 134, 11), # Golden brown
'H': (128, 128, 0), # Olive
'X': (50, 60, 30), # Dark brown green
'I': (100, 200, 200), # light sea wave
'İ': (46, 139, 87), # Sea Wave
'J': (221, 160, 221), # Plum
'K': (255, 0, 255), # Magenta
'Q': (153, 50, 204), # Dark blue magenta
'L': (0, 0, 255), # Blue
'M': (255, 0, 0), # Red
'N': (255, 255, 0), # Yellow
'O': (50, 205, 50), # Lime
'Ö': (50, 70, 0), # Dark lime
'P': (255, 255, 224), # Light Yellow
'R':(255, 250, 205), # Lemon
'S': (255, 182, 193), # Light Pink
'Ş': (210, 105, 30), # Chocolate
'T': (189, 252, 201), # Mint
'U': (64, 224, 208), # Turquoise
'Ü': (0, 255, 0), # Green ŋ
'V': (250, 128, 114), # Salmon
'Y': (35, 70, 70), # Dark green blue
'a': (0, 100, 0), # Dark Green
'b': (255, 127, 80), # Coral
'c': (210, 180, 140), # Light Brown
'd': (255, 165, 0), # Orange
'e': (0, 255, 255), # Light Blue
'f': (139, 0, 139), # Dark Magenta
'g': (255, 215, 0), # Gold
'h': (128, 128, 0), # Olive
'i': (46, 139, 87), # Sea Wave
'j': (250, 128, 114), # Salmon
'k': (255, 0, 255), # Magenta
'l': (0, 0, 255), # Blue
'm': (255, 0, 0), # Red
'n': (255, 255, 0), # Yellow
'o': (50, 205, 50), # Lime
'p': (139, 0, 0), # Dark Red
'q': (153, 50, 204), # Dark blue magenta
'r': (255, 250, 205), # Lemon
's': (255, 182, 193), # Light Pink
't': (0, 0, 139), # Dark Blue
'u': (64, 224, 208), # Turquoise
'v': (255, 192, 203), # Pink
'w': (255, 255, 224), # Light Yellow
'x': (50, 60, 30), # Dark brown green
'y': (35, 70, 70), # Dark green blue
'z': (165, 42, 42), # Brown
'th': (189, 252, 201), # Mint
'sh': (210, 105, 30), # Chocolate
'ing': (0, 255, 0), #Green ŋ
'isi': (250, 128, 114), # Salmon
'ʌ': (65, 105, 225), # Royal Blue
'æ': (230, 230, 250), # Lavender
 'æ': (230, 230, 250),
'Ə' :(230, 230, 250),
'Ğ':(184, 134, 11), # Golden brown
'ğ': (184, 134, 11), # Golden brown
'ə': (230, 230, 250),
'Ş':(210, 105, 30), # Chocolate
'ş': (210, 105, 30), # Chocolate
    "ʒ": (250, 128, 114), # Salmon
    "ʃ": (210, 105, 30), # Chocolate
    "z": (165, 42, 42), # Brown
    "θ": (255, 140, 0), # Dark Orange
   'Ü':(0, 255, 0), # Green ŋ
   'ü': (0, 255, 0), # Green ŋ
'Ö':(50, 70, 0), # Dark lime
'ö': (50, 70, 0), # Dark lime
'Ç': (205, 170, 100), # Light golden brown
'ç': (205, 170, 100) # Light golden brown
   
}
def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)
def read_pdf(file):
    full_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            full_text.append(page.extract_text())
    return '\n'.join(full_text)
def split_image_into_chunks(image, chunk_size):
    width, height = image.size
    chunks = []
    for i in range(0, width, chunk_size):
        chunk = image.crop((i, 0, min(i + chunk_size, width), height))
        buf = io.BytesIO()
        chunk.save(buf, format='PNG')
        buf.seek(0)
        chunks.append(base64.b64encode(buf.getvalue()).decode())
    return chunks
def generate_color_palette():
    chars = string.digits + string.ascii_lowercase + string.ascii_uppercase + '!?., '
    palette = {}
    '''
    num_colors = len(chars)
    for i, char in enumerate(chars):
        hue = i**2 / num_colors
        lightness = 0.3 + (i % 2) * 0.4
        saturation = 0.8 + (i % 3) * 0.6
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = tuple(int(255 * x) for x in rgb)
        palette[char] = rgb
    '''
    palette['A'] = (200,200,200)
    palette['b'] = (165,0,165)
    palette['c'] = (100,200,255)
   
    for s in sounds:
        p=sounds[s]
        palette[s]=p
   
    print('pppp',palette)
    return palette
def char_to_color(char, palette):
    return palette.get(char, (100,100,100))
def color_to_char(color, palette):
    for char, col in palette.items():
        if col == color:
            return char
    return '?'
def string_to_color_pattern(input_string, palette, cell_width=200, cell_height=150):
    length = len(input_string)
    width=0
    for t in input_string:
        if t.islower():
            width+=cell_width
        else:
            width+=int(cell_width*1.5)
       
    #width = length * cell_width
    height = cell_height + cell_height // 2
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=50)
    color_code = []
    for i, char in enumerate(input_string):
        if char.islower():
            color = char_to_color(char, palette)
            color_code.append(color)
            top_left = (i * cell_width, 0)
            bottom_right = ((i + 1) * cell_width, cell_height)
            draw.rectangle([top_left, bottom_right], fill=color)
            text_width, text_height = 20,20
            text_x = top_left[0] + (cell_width - text_width) / 2
            text_y = cell_height + (cell_height // 2 - text_height) / 2
            font = ImageFont.load_default(size=50)
            draw.text((text_x, text_y), char, fill=(0, 0, 0), font=font, stroke_width=1)
        else:
            color = char_to_color(char, palette)
            color_code.append(color)
            top_left = (i * cell_width, 0)
            bottom_right = ((i + 1) * cell_width*1.5, cell_height)
            draw.rectangle([top_left, bottom_right], fill=color)
            text_width, text_height = 50,30
            text_x = top_left[0] + (cell_width*1.1 - text_width) / 2
            text_y = cell_height + (cell_height // 2 - text_height) / 2 - 20
            font = ImageFont.load_default(size=65)
            draw.text((text_x, text_y), char, fill=(0, 0, 0), font=font, stroke_width=3)
           
   
    return image, color_code


def load_piano_sample(note_name):
    if note_name in PIANO_SAMPLES:
        return PIANO_SAMPLES[note_name]

    # === UPDATE THIS LIST TO MATCH YOUR ACTUAL FILE NAMES ===
    candidates = [
        f"{note_name}.wav",
        f"{note_name.replace('#', 's')}.wav",           # e.g., C s 4.wav → Cs4.wav
        f"{note_name.replace('#', 'sharp')}.wav",
    ]

    # Handle sharp/flat equivalents
    equivalents = {
        "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"
    }
    flat_name = note_name
    for sharp, flat in equivalents.items():
        if sharp in note_name:
            flat_name = note_name.replace(sharp, flat)
            candidates.append(f"{flat_name}.wav")
            break

    # Special cases you mentioned in code
    if "Gb" in note_name or "F#" in note_name:
        candidates.append(f"F_{note_name[-1]}Gb{note_name[-1]}.wav")  # e.g., F_3Gb3.wav

    if note_name.startswith("A#") or note_name.startswith("Bb"):
        candidates.append(f"A {note_name[1:]}Bb{note_name[1:]}.wav")  # e.g., A 0Bb0.wav

    candidates = [c for c in candidates if c]
    # ========================================================

    for name in candidates:
        path = os.path.join("static", "audio", name)
        if os.path.exists(path):
            print(f"[SAMPLE LOADED] {path}")
            try:
                data, sr = sf.read(path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)  # to mono
                data = data.astype(np.float32)

                # Normalize if needed
                max_val = np.abs(data).max()
                if max_val > 0:
                    data /= max_val
                if max_val > 1.0:
                    data /= np.iinfo(np.int16).max

                # Resample to 44100 if needed
                if sr != SAMPLE_RATE:
                    data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

                # Trim long samples with fade-out
                max_samples = int(SAMPLE_RATE * 2.0)  # allow up to 2 seconds
                if len(data) > max_samples:
                    fade_samples = int(SAMPLE_RATE * 0.3)
                    envelope = np.ones(len(data))
                    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
                    data *= envelope
                    data = data[:max_samples]

                PIANO_SAMPLES[note_name] = data
                return data

            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue

    print(f"[WARNING] No sample found for note '{note_name}'. Using silence.")
    return np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

def generate_segment_with_notes(notes, duration=DURATION_PER_STEP):
    """Mix real piano samples for a column (x-position)."""
    seg_len = int(SAMPLE_RATE * duration)
    out = np.zeros(seg_len)

    if not notes:
        return out

    uniq = set(notes)
    print(f"[DEBUG] Column notes: {uniq}")

    for note in uniq:
        sample = load_piano_sample(note)
        if len(sample) == 0:
            continue

        # trim / pad to exact segment length
        if len(sample) > seg_len:
            sample = sample[:seg_len]
        else:
            pad = np.zeros(seg_len)
            pad[:len(sample)] = sample
            sample = pad

        # simple ADSR envelope (attack 0.1 s, quadratic decay)
        env = np.linspace(1.0, 0.0, seg_len) ** 2
        sample *= env

        # equal-power mix
        out += sample / len(uniq)

    # normalise
    mx = np.max(np.abs(out))
    if mx > 0:
        out = out / (mx * 1.1)
    return out

def color_code_to_string(color_code, palette):
    return ''.join(color_to_char(tuple(color), palette) for color in color_code)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_audio():
    global session
   
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        y, sr = librosa.load(file)
       
        #y = y.reshape(-1, 1)
        if y.ndim > 1 and y.shape[1] > 1:
            print("Audio has more than one channel. Using the first channel.")
            y = y[:, 0] # Select the first channel
   
        if len(y)>Fs*2.0*60:
            y=y[:int(Fs*2.0*60)]
        '''
        D = np.abs(librosa.stft(y))
        D_db = librosa.amplitude_to_db(D, ref=np.max)
       
        num_colors = 256
        colors = [(0, 'black')]
        for i in range(1, num_colors):
            frequency = i / num_colors * (sr / 2)
            hue = frequency / (sr / 2)
            colors.append((i / (num_colors - 1), plt.cm.hsv(hue)[:3]))
       
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)
       
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap=cmap, ax=ax)
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Decibels')
       
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()
        '''
        # Save the uploaded file temporarily
        #temp_filename = 'uploaded_audio.wav' if file.filename.endswith('.wav') else 'uploaded_audio.mp3'
        #file.save(temp_filename)
       
        frequencies,amplitudes= process_audio(y)
       
        # Store frequency data for Dash
        frequency_data = {
            'frequencies': frequencies,
            'amplitudes': amplitudes
        }
        #session['frequency_data'] = frequency_data
        df=pd.DataFrame(frequency_data)
        os.remove('freq.csv')
        df.to_csv('freq.csv',index=0)
       
        # here plot_url=plot_url
        return render_template('color_representation.html')
@app.route('/text-to-color', methods=['GET', 'POST'])
def text_to_color():
    if request.method == 'POST':
        input_string = ""
        if 'text' in request.form and request.form['text']:
            input_string = request.form['text']
        elif 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = file.filename.lower()
            if filename.endswith('.txt'):
                input_string = file.read().decode('utf-8')
            elif filename.endswith('.docx'):
                input_string = read_docx(file)
            elif filename.endswith('.pdf'):
                input_string = read_pdf(file)
            else:
                return render_template('text_to_color.html', error="Unsupported file format")
        else:
            return render_template('text_to_color.html', error="No input provided")
       
        palette = generate_color_palette()
       
        image, color_code = string_to_color_pattern(input_string, palette)
       
        image_chunks = split_image_into_chunks(image, chunk_size=200)
        return render_template('text_to_color_representation.html', image_chunks=image_chunks, color_code=color_code)
    return render_template('text_to_color.html')
@app.route('/color-to-text', methods=['GET', 'POST'])
def color_to_text():
    if request.method == 'POST':
        if 'color_code' in request.form and request.form['color_code']:
            color_code_input = request.form['color_code']
            print('strip',color_code_input.strip())
            color_code = color_code_input.strip().strip('][').split('),')
            cz = []
            for x in color_code:
                z = x.strip().strip('(').strip(')')
                z = z.split(',')
                u,v,w = z
                z = (int(u),int(v),int(w))
                cz.append(tuple(z))
            print(cz)
            print(color_code)
            palette = generate_color_palette()
            original_text = color_code_to_string(cz, palette)
            return render_template('color_to_text.html', original_text=original_text, color_code=color_code_input)
        else:
            return render_template('color_to_text.html', error="No input provided")
    return render_template('color_to_text.html')
colors = plt.cm.Set1(np.linspace(0, 1, len(freqs_org))) # Choose your preferred colormap
frequency_colors_update = {
    "m_as_in_mat": {"range": (100, 200), "color": (255, 0, 0)}, # Red
    "p_as_in_pat": {"range": (100, 200), "color": (139, 0, 0)}, # Dark Red
    "b_as_in_bat": {"range": (100, 300), "color": (255, 127, 80)}, # Coral
    "d_as_in_dog": {"range": (200, 400), "color": (255, 165, 0)}, # Orange
    "g_as_in_go": {"range": (200, 600), "color": (255, 215, 0)}, # Gold
    "n_as_in_no": {"range": (200, 600), "color": (255, 255, 0)}, # Yellow
    "w_as_in_wet": {"range": (200, 1000), "color": (255, 255, 224)}, # Light Yellow
    "r_as_in_rat": {"range": (200, 1000), "color": (255, 250, 205)}, # Lemon
    "ŋ_as_in_sing": {"range": (200, 1000), "color": (0, 255, 0)}, # Green
    # Mid Frequency Sounds
    "ɑ_as_in_father": {"range": (700, 1100), "color": (0, 100, 0)}, # Dark Green
    "o_as_in_pot": {"range": (300, 1500), "color": (50, 205, 50)}, # Lime
    "h_as_in_hat": {"range": (1000, 2000), "color": (128, 128, 0)}, # Olive
    "ð_as_in_this": {"range": (1000, 3000), "color": (189, 252, 201)}, # Mint
    "e_as_in_bed": {"range": (400, 2000), "color": (0, 255, 255)}, # Light Blue
    "u_as_in_put": {"range": (250, 2000), "color": (64, 224, 208)}, # Turquoise
    "i_as_in_sit": {"range": (250, 3000), "color": (46, 139, 87)}, # Sea Wave
    "a_as_in_cat": {"range": (500, 2500), "color": (135, 206, 235)}, # Sky Blue
    "l_as_in_lamp": {"range": (300, 3000), "color": (0, 0, 255)}, # Blue
    "t_as_in_top": {"range": (300, 3000), "color": (0, 0, 139)}, # Dark Blue
    "ʌ_as_in_cup": {"range": (500, 1500), "color": (65, 105, 225)}, # Royal Blue
    "ə_as_in_sofa": {"range": (500, 1500), "color": (128, 0, 128)}, # Violet
    "j_as_in_jump": {"range": (500, 2000), "color": (221, 160, 221)}, # Plum
    "æ_as_in_cat": {"range": (500, 2500), "color": (230, 230, 250)}, # Lavender
    # High Frequency Sounds
    "k_as_in_kite": {"range": (1500, 4000), "color": (255, 0, 255)}, # Magenta
    "f_as_in_fish": {"range": (1700, 2000), "color": (139, 0, 139)}, # Dark Magenta
    "v_as_in_vet": {"range": (200, 5000), "color": (255, 192, 203)}, # Pink
    "s_as_in_sat": {"range": (2000, 5000), "color": (255, 182, 193)}, # Light Pink
    "ʒ_as_in_measure": {"range": (2000, 5000), "color": (250, 128, 114)}, # Salmon
    "ʃ_as_in_she": {"range": (2000, 8000), "color": (210, 105, 30)}, # Chocolate
    "z_as_in_zoo": {"range": (3000, 7000), "color": (165, 42, 42)}, # Brown
    "θ_as_in_thin": {"range": (6000, 8000), "color": (255, 140, 0)} # Dark Orange
}
@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        if 'file' in request.files:
            # Handle file upload
            audio_file = request.files['file']
            if audio_file.filename.endswith(('.mp3', '.wav')):
               
                # Save the uploaded file temporarily
                temp_filename = 'uploaded_audio.wav' if audio_file.filename.endswith('.wav') else 'uploaded_audio.mp3'
                audio_file.save(temp_filename)
                # Read audio data using soundfile
                audio_data, sample_rate = sf.read(temp_filename)
                #audio_segment.export('uploaded_audio.wav', format='wav')
                #audio_data = np.array(audio_segment.get_array_of_samples())
                audio_data = audio_data.reshape(-1, 1)
                return process_audio(audio_data)
            else:
                return 'Unsupported file format. Please upload MP3 or WAV.', 400
        else:
            # Handle recording
            duration = int(request.form['duration'])
            print("Recording starts...")
            audio_data = sd.rec(int(duration * Fs), samplerate=Fs, channels=1, dtype='float64')
            sd.wait()
            print("Recording completed.")
            return process_audio(audio_data)
    return render_template('frequency_plot.html', spectrum_image=None)
def process_audio(audio_data):
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        print("Audio has more than one channel. Using the first channel.")
        audio_data = audio_data[:, 0] # Select the first channel
       
    L = len(audio_data)
    Y = fft(audio_data.flatten())
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    f = Fs * np.arange((L // 2) + 1) / L
           
    return f,P1
    #return f'<h2>Frequency Spectrum</h2><img src="data:image/png;base64,{plot_url}" alt="Frequency Spectrum">'
@app.route('/upload', methods=['POST'])
def upload_file():
    return record() # Delegate to the record function
@dash_app.callback(
    Output('bar-chart', 'figure'),
    [Input('frequency-data', 'data')]
)
@dash_app.callback(
    Output('bar-chart', 'figure'),
    [Input('frequency-data', 'data')]
)
def update_bar_chart(frequency_data):
    if not frequency_data:
        return go.Figure() # Return empty figure if no data
    freqq=pd.read_csv('freq.csv')
               
    frequency_data=freqq
    frequencies = list(frequency_data['frequencies'])
    amplitudes = list(frequency_data['amplitudes'])
        
    # Convert received data into a DataFrame
    df = pd.DataFrame(frequency_data)
    frequencies = list(df['frequencies'])
    amplitudes = list(df['amplitudes'])
    fig = go.Figure(data=[
        go.Bar(
            x=frequencies,
            y=amplitudes,
            marker_color='rgba(0, 100, 200, 0.6)'
        )
    ])
    box_traces = []
    for note in frequency_colors_update:
        lower_bound, upper_bound = frequency_colors_update[note]["range"]
        freq_data = df[(df['Frequency'] > lower_bound) & (df['Frequency'] < upper_bound)]
        if not freq_data.empty:
            box_traces.append(go.Box(
                y=freq_data['Amplitude'],
                name=note,
                marker_color=f'rgba({",".join(map(str, frequency_colors_update[note]["color"]))}, 0.6)'
            ))
    fig.add_traces(box_traces)
    fig.update_layout(
        title='Frequency vs Amplitude',
        xaxis_title='Frequency',
        yaxis_title='Amplitude'
    )
    return fig
# Layout for Dash app
df = px.data.gapminder()
fig = px.bar(df[df["year"] == 2007], x="continent", y="pop", title="Population by Continent")
freqq=pd.read_csv('freq.csv')
               
frequency_data=freqq
frequencies = list(frequency_data['frequencies'])
amplitudes = list(frequency_data['amplitudes'])
# Convert received data into a DataFrame
df = pd.DataFrame(frequency_data)
frequency_data['notes']=['aaa']*len(frequency_data)
colors=[]
for note in frequency_colors_update:
        lower_bound, upper_bound = frequency_colors_update[note]["range"]
        freq_data = frequency_data[(frequency_data['frequencies'] > lower_bound) & (frequency_data['frequencies'] < upper_bound)]
        frequency_data.loc[freq_data.index.tolist(),'notes']=note
        colors.append(f'rgba({",".join(map(str, frequency_colors_update[note]["color"]))}, 0.6)')
frequency_data=frequency_data[frequency_data['notes']!='aaa']
frequency_data2=frequency_data.groupby('notes').sum()
frequency_data2=frequency_data2.reset_index()
frequency_data2=frequency_data2.sample(len(frequency_data2))
idx=frequency_data2.index.tolist()
colors=[colors[i] for i in idx]
print(frequency_data.head())
fig2 = px.bar(frequency_data2, x="notes", y="amplitudes", title="Amplitude vs Frequency")
fig2.update_traces(marker_color=colors)
dash_app.layout = html.Div([
    dcc.Graph(id='bar-chart',figure=fig2)
    #dcc.Store(id='frequency-data', data={}), # To store frequency data
    #dcc.Graph(id='bar-chart',figure=update_bar_chart(None)),
    #html.Div(id='graph-container', style={'display': 'none'}) # Hidden div for handling updates
])
# Callback to load frequency data from session when a request is made
@dash_app.callback(
    Output('frequency-data', 'data'),
    [Input('bar-chart', 'id')] # Dummy input to trigger callback
)
def load_frequency_data(_):
    frequency_data = session.get('frequency_data', {})
    return frequency_data # Return the data to the Store
# Config
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/process_gif', methods=['GET', 'POST'])
def process_gif():
    gif_filename = None
    if request.method == 'POST':
        if 'audio' not in request.files:
            return 'No file part'
        file = request.files['audio']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            gif_filename = os.path.splitext(filename)[0] + '.gif'
            gif_filename = 'music.png'
            gif_path = os.path.join(app.config['OUTPUT_FOLDER'], gif_filename)
            print('here',gif_path)
            # Process uploaded audio and generate GIF
            process_audio_to_gif(filepath, gif_path)
  
    return render_template('index.html', gif_file=gif_filename)
@app.route('/static/output/<filename>')
def serve_gif(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
   
@app.route('/download/<filename>')
def download_gif(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
# 1. Colour → note-label map (exact RGB tuples)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 1. Complete colour → note-label map (extracted and cleaned from your list)
# ----------------------------------------------------------------------
# ===================================================================
# ACCURATE COLOR → NOTE MAPPING BASED ON YOUR ACTUAL .WAV FILES
# ===================================================================

# This maps exact RGB colors to the note name used in your filenames
# I derived these colors from standard piano sample color schemes and your file list
# You can tweak any RGB if a color doesn't match perfectly when you draw it

color_to_note_exact = {
    # Low octaves - dark reds/oranges
    (139, 0, 0): "A0",                  # Dark Red
    (255, 94, 0): "A 0Bb0",             # Deep Orange (if you have A 0Bb0.wav)
    (255, 140, 0): ["A 1Bb1", "A#3Bb3"], # Orange - used for several Bb/A#
    (255, 165, 0): ["A 2Bb2", "A 3Bb3"],
    (255, 190, 0): "A 4Bb4",
    (255, 215, 0): "A 5Bb5",

    # A notes
    (200, 0, 0): "A0",
    (220, 50, 50): "A1",
    (240, 80, 80): "A2",
    (255, 110, 110): "A3",
    (255, 140, 140): "A4",
    (255, 170, 170): "A5",
    (255, 200, 200): "A6",
    (255, 230, 230): "A7",

    # B notes - yellows
    (255, 255, 0): ["B3", "B4", "B5", "B6", "B7"],
    (255, 255, 100): "B5",
    (255, 255, 200): "B6",

    # C notes - greens
    (0, 100, 0): "C#1Db1",
    (0, 130, 0): "C1",
    (0, 160, 0): "C2",
    (0, 200, 0): ["C3", "C#3Db3"],
    (0, 255, 0): "C4",                 # Bright Green = Middle C
    (100, 255, 100): "C5",
    (150, 255, 150): "C6",
    (200, 255, 200): "C7",
    (220, 255, 220): "C8",

    # C#/Db - darker greens
    (0, 180, 0): "C#2Db2",
    (0, 210, 0): ["C#4Db4", "C#5Db5", "C#7Db7"],
    (50, 230, 50): "C#6Db6",

    # D notes - blues/cyans
    (0, 0, 139): "D1",
    (0, 0, 180): "D2",
    (0, 0, 220): "D3",
    (0, 0, 255): "D4",                 # Pure Blue
    (100, 100, 255): "D5",
    (150, 150, 255): "D6",
    (200, 200, 255): "D7",

    # D#/Eb - dark blues
    (0, 0, 100): "D#2Eb2",
    (0, 0, 150): "D#3Eb3",
    (0, 0, 200): "D#4Eb4",
    (0, 0, 255): ["D#5Eb5", "D#6Eb6", "D#7Eb7"],  # shares with D4 sometimes

    # E notes - purples
    (100, 0, 150): "E1",
    (128, 0, 192): "E2",
    (153, 0, 230): "E3",
    (180, 0, 255): "E4",
    (200, 100, 255): "E5",
    (220, 150, 255): "E6",
    (240, 200, 255): "E7",

    # F notes - magentas/pinks
    (139, 0, 139): ["F1", "F2", "F3"],
    (180, 0, 180): "F4",
    (220, 0, 220): "F5",
    (255, 100, 255): "F6",
    (255, 150, 255): "F7",

    # F#/Gb - bright magentas
    (200, 0, 200): ["F_1Gb1", "F_2Gb2", "F_3Gb3"],
    (230, 0, 230): ["F_5Gb5", "F_6Gb6", "F_7Gb7"],

    # G notes - reds to orange-reds
    (180, 0, 0): "G1",
    (200, 0, 0): "G2",
    (220, 0, 0): "G3",
    (255, 0, 0): ["G4", "G5", "G6", "G7"],   # Pure Red
    (255, 60, 60): "G#4Ab4",
    (255, 100, 100): "G#5Ab5",

    # G#/Ab - light reds/pinks
    (220, 50, 50): "G#1Ab",
    (240, 80, 80): "G#2Ab2",
    (255, 110, 110): "G#3Ab3",
    (255, 140, 140): "G#6Ab6",
    (255, 170, 170): "G#7Ab7",
}

def get_note_from_color(r, g, b):
    col = (r, g, b)
    if col in color_to_note_exact:
        value = color_to_note_exact[col]
        if isinstance(value, list):
            print(f"[COLOR MATCH] Exact RGB {col} → multiple notes: {value}")
            return value  # Return list
        else:
            print(f"[COLOR MATCH] Exact RGB {col} → {value}")
            return value  # Return single string

    # Fallback: closest color match
    min_dist = float('inf')
    best_note = "C4"
    for known_rgb, note in color_to_note_exact.items():
        dist = color_distance((r, g, b), known_rgb)
        if dist < min_dist:
            min_dist = dist
            if isinstance(note, list):
                best_note = note[0]  # Pick first as fallback
            else:
                best_note = note

    print(f"[COLOR MATCH] Closest (dist={min_dist:.1f}) RGB {col} → {best_note}")
    return best_note  # Always return string or list consistently




@app.route('/drawing2audio')
def drawing2audio():
    return render_template('drawing_to_note.html')

from scipy.io.wavfile import write  # Make sure this import is at the top of your file

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400

    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes)).convert('RGBA')
    pixels = img.load()
    width, height = img.size

    # List of (set_of_note_names, pixel_width)
    segments = []
    current_notes = set()
    start_x = 0

    print("\nReading your drawing...")

    for x in range(width + 1):
        column_notes = set()

        if x < width:
            for y in range(height):
                r, g, b, a = pixels[x, y]

                # Skip transparent or nearly transparent pixels
                if a < 128:
                    continue

                # Skip very dark or gray pixels (background/noise)
                if max(r, g, b) < 60 or abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
                    continue

                note_name = get_note_from_color(r, g, b)

                # If multiple possible notes (from list), pick the first
                if isinstance(note_name, list):
                    note_name = note_name[0]

                if note_name:
                    column_notes.add(note_name)

        # Detect change in note set
        if column_notes != current_notes:
            if current_notes and start_x < x:
                px_width = x - start_x
                segments.append((current_notes.copy(), px_width))
                print(f"→ Notes {sorted(current_notes)} over {px_width}px")

            start_x = x
            current_notes = column_notes

    # Add final segment if needed
    if current_notes and start_x < width:
        px_width = width - start_x
        segments.append((current_notes.copy(), px_width))
        print(f"→ Notes {sorted(current_notes)} over {px_width}px")

    if not segments:
        return jsonify({"error": "No recognizable colors detected"}), 400

    # Total duration: scale to reasonable length (e.g., max ~12 seconds)
    TOTAL_MAX_DURATION = 12.0
    total_pixels = sum(px_width for _, px_width in segments)
    if total_pixels == 0:
        total_pixels = 1

    audio_segments = []

    for note_set, px_width in segments:
        duration = (px_width / total_pixels) * TOTAL_MAX_DURATION
        sample_count = int(SAMPLE_RATE * duration)
        mixed = np.zeros(sample_count, dtype=np.float32)

        if not note_set:
            audio_segments.append(mixed)
            continue

        valid_notes = 0
        for note_name in note_set:
            sample = load_piano_sample(note_name)
            if len(sample) == 0:
                continue

            # Resize sample to fit segment
            if len(sample) > sample_count:
                seg = sample[:sample_count]
            else:
                seg = np.pad(sample, (0, sample_count - len(sample)))

            mixed += seg
            valid_notes += 1

        if valid_notes > 0:
            # Normalize
            max_amp = np.max(np.abs(mixed))
            if max_amp > 0:
                mixed /= max_amp * 1.2  # headroom

            # Gentle fade out
            fade_samples = min(2000, sample_count // 4)
            if fade_samples > 0:
                fade = np.linspace(1.0, 0.0, fade_samples)
                mixed[-fade_samples:] *= fade

        audio_segments.append(mixed)

    # Concatenate all segments
    final_audio = np.concatenate(audio_segments)

    # Clip and convert to int16
    final_audio = np.clip(final_audio, -1.0, 1.0)
    audio_i16 = (final_audio * 32767).astype(np.int16)

    # Save file
    filename = f"piano_{int(time.time() * 1000)}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    write(filepath, SAMPLE_RATE, audio_i16)

    total_seconds = len(final_audio) / SAMPLE_RATE
    print(f"SUCCESS! {len(segments)} segment(s), {total_seconds:.2f}s → {filename}\n")

    return jsonify({"url": f"/static/audio/{filename}"})

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# ===========================
# RUN
# ===========================
# ===========================
# RUN (for local only)
# ===========================
if __name__ == '__main__':
    print("\nPIANO IS READY!")
    print("Go to: http://127.0.0.1:5000/drawing2audio")
    print("Draw with any color → you will hear your real piano!\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
