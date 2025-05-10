import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import imageio
import os
import re
from concurrent.futures import ThreadPoolExecutor

# Existing imports and dictionaries (frequency_colors_update, note_number_dict) remain unchanged

# Updated freq_symbols with dynamic symbol selection based on duration
freq_symbols = {
    "A0": {"frequency": 27.50, "color": [139, 0, 0], "range": [27.50, 29.14]},
    "A#0/Bb0": {"frequency": 29.14, "color": [255, 69, 0], "range": [29.14, 30.87]},
    # ... (include all notes as in the original freq_symbols)
    "C8": {"frequency": 4186.01, "color": [200, 255, 0], "range": [4186.01, 4434.92]},
}

# Define note symbols based on duration (in beats)
NOTE_SYMBOLS = {
    "whole": {"symbol": "𝅝", "beats": 4.0},
    "half": {"symbol": "𝅗𝅥", "beats": 2.0},
    "quarter": {"symbol": "𝅘𝅥", "beats": 1.0},
    "eighth": {"symbol": "𝅘𝅥𝅮", "beats": 0.5},
    "sixteenth": {"symbol": "𝅘𝅥𝅯", "beats": 0.25},
    "thirty-second": {"symbol": "𝅘𝅥𝅰", "beats": 0.125},
    "sixty-fourth": {"symbol": "𝅘𝅥𝅱", "beats": 0.0625},
}

# Function to map duration to note symbol
def get_note_symbol(duration_beats, tempo=120):
    """
    Map note duration (in beats) to the appropriate musical symbol.
    
    Args:
        duration_beats (float): Duration of the note in beats.
        tempo (int): Tempo in beats per minute (BPM), default 120.
    
    Returns:
        str: Musical note symbol.
    """
    # Sort note types by duration (descending) to find the closest match
    for note_type, info in sorted(NOTE_SYMBOLS.items(), key=lambda x: x[1]["beats"], reverse=True):
        if duration_beats >= info["beats"]:
            return info["symbol"]
    return NOTE_SYMBOLS["sixty-fourth"]["symbol"]  # Default to shortest note if duration is very small

def get_notes_from_audio(y, sr, tempo=120):
    """
    Extract notes and their durations from audio data.
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
        tempo (int): Tempo in beats per minute (BPM), default 120.
    
    Returns:
        list: List of tuples (note, duration_beats).
    """
    # Perform pitch detection
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    
    # Get the dominant pitch for each frame
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()  # Get index of max magnitude
        pitch.append(pitches[index, t])
    
    pitch = np.array(pitch)
    
    # Convert pitches to notes and estimate durations
    notes = []
    current_note = None
    note_start = 0
    frame_duration = 1.0 / (sr / 512)  # Approximate frame duration in seconds (assuming hop_length=512)
    
    for t, p in enumerate(pitch):
        if p > 0:
            note = librosa.hz_to_note(p)
            if note != current_note:
                if current_note is not None:
                    # Calculate duration of the previous note
                    duration_seconds = (t - note_start) * frame_duration
                    duration_beats = duration_seconds * (tempo / 60.0)
                    notes.append((current_note, duration_beats))
                current_note = note
                note_start = t
        else:
            if current_note is not None:
                # End of a note
                duration_seconds = (t - note_start) * frame_duration
                duration_beats = duration_seconds * (tempo / 60.0)
                notes.append((current_note, duration_beats))
                current_note = None
                note_start = t
    
    # Handle the last note if it exists
    if current_note is not None:
        duration_seconds = (len(pitch) - note_start) * frame_duration
        duration_beats = duration_seconds * (tempo / 60.0)
        notes.append((current_note, duration_beats))
    
    return notes

def process_audio_to_gif(audio_path, gif_path, tempo=120):
    audio_data_org, sr = librosa.load(audio_path)      
    counter = 0

    # Canvas setup
    width = 1000
    height = 220
        
    t = len(audio_data_org) // sr
    
    # Clear the music folder
    files = os.listdir('music')
    for f in files:
        os.remove(os.path.join('music', f))

    nn = int(sr * 0.7)
    d = len(list(range(0, sr * t - nn, nn)))
    canvas_size = (width, height * d)
    artwork = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(artwork)
    
    for tt in range(0, sr * t - nn, nn):
        audio_data = audio_data_org[tt:tt + nn]
        notes = get_notes_from_audio(audio_data, sr, tempo)
        
        lower_freq_lines = [120, 130, 140, 150, 160, 170]
        lower_freq_lines = [counter * height + ii for ii in lower_freq_lines]
        
        higher_freq_lines = [30, 40, 50, 60, 70, 80]
        higher_freq_lines = [counter * height + ii for ii in higher_freq_lines]

        # Draw staff lines
        num_lines = 6
        for i in range(num_lines):           
            y = lower_freq_lines[i]
            draw.line([0, y, width, y], fill="black", width=2)
            
        for i in range(num_lines):           
            y = higher_freq_lines[i]
            draw.line([0, y, width, y], fill="black", width=2)
            
        draw.line([0, 0, 0, height * d], fill="black", width=7)
        draw.line([width, 0, width, height * d], fill="black", width=7)

        C = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        font = ImageFont.truetype("static/Bravura.otf", size=50)
        y_position = lower_freq_lines[0] - 80
        draw.text((5, y_position), "𝄢", fill=C, font=font)  

        y_position = higher_freq_lines[0] - 60
        draw.text((5, y_position), "𝄞", fill=C, font=font)
        counter += 1
        
        for i, (note, duration_beats) in enumerate(notes):
            n = note.replace("♯", "#")
            
            if n not in freq_symbols:
                continue
            string_number = get_string_number_mod12(n[:3])
            
            # Get the appropriate symbol based on duration
            symbol = get_note_symbol(duration_beats, tempo)
            color = tuple(freq_symbols[n]["color"])
            freq_value = freq_symbols[n]["frequency"]
           
            total_notes = len(notes)
            spacing = width / total_notes
            note_index = list(freq_symbols.keys()).index(n)
            x_position = 5 + int(width * 1 / len(notes)) + int(width * i / len(notes))
            
            print(n, string_number, symbol, duration_beats)
            
            if string_number < 6:
                y_position = lower_freq_lines[0] - 60 - 2 * string_number
                draw.text((x_position, y_position), symbol, fill=color, font=font)                
            else:
                y_position = higher_freq_lines[0] - 60 - (2 * (string_number - 6))
                draw.text((x_position, y_position), symbol, fill=color, font=font)
                
    # Save the artwork
    artwork.save('static/output/music.png', "PNG")

    # Create GIF
    image_folder = 'music'
    output_filename = gif_path
    create_gif(image_folder, output_filename, duration=0.5)

# Existing create_gif function remains unchanged
def create_gif(image_folder, output_filename, duration=0.5):
    filenames = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(imageio.imread, filenames))
    imageio.mimsave(output_filename, images, duration=duration, loop=0)
    print(f"GIF saved as {output_filename}")

# Example usage
# process_audio_to_gif("input_audio.wav", "output_music.gif", tempo=120)
