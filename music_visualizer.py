import librosa
import numpy as np
import pygame
import sounddevice as sd
import queue
import os
import subprocess
import re
from collections import deque
import tempfile
import shutil
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

# import random
import matplotlib.pyplot as plt
import imageio
#import imageio as iio
import os
import math
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
FRAME_RATE = 20  # Frames per second for the GIF
FRAME_WIDTH = 800
FRAME_HEIGHT = 300 # Height for Grand Staff + some margin
STAFF_LINE_SPACING = 12 # Pixels between staff lines
STAFF_TOP_MARGIN = 40 # Margin above the top staff line
STAFF_BOTTOM_MARGIN = 40 # Margin below the bottom staff line
CLEF_WIDTH = 50
TIME_SIG_WIDTH = 40
PIXELS_PER_BEAT = 80 # How many pixels represent one beat horizontally

# Note detection parameters
MIN_NOTE_DURATION_SEC = 0.08 # Minimum duration for a detected pitch to be considered a note
MAGNITUDE_THRESHOLD_RATIO = 0.1 # Minimum magnitude relative to max magnitude in a frame

# Quantization tolerance: how close a detected duration must be to a standard duration (as a ratio)
QUANTIZATION_TOLERANCE = 0.2 # e.g., 0.2 means +/- 20% of the target duration

# --- Musical Notation Mapping ---

# Vertical pixel offset relative to the bottom line of the staff (E4 for Treble, G2 for Bass)
# Each step is STAFF_LINE_SPACING / 2
# Treble Clef (G clef): E4 is on the bottom line (offset 0 relative to E4)
# Bass Clef (F clef): G2 is on the bottom line (offset 0 relative to G2)
NOTE_STAFF_STEPS = {
    ('E', 4): 0, ('F', 4): 1, ('G', 4): 2, ('A', 4): 3, ('B', 4): 4,
    ('C', 5): 5, ('D', 5): 6, ('E', 5): 7, ('F', 5): 8, ('G', 5): 9,
    ('A', 5): 10, ('B', 5): 11, ('C', 6): 12, ('D', 6): 13, ('E', 6): 14,
    ('F', 6): 15, ('G', 6): 16, ('A', 6): 17, ('B', 6): 18, ('C', 7): 19,

    ('D', 4): -1, ('C', 4): -2, ('B', 3): -3, ('A', 3): -4, ('G', 3): -5,
    ('F', 3): -6, ('E', 3): -7, ('D', 3): -8, ('C', 3): -9, ('B', 2): -10,

    # Bass Clef Mapping (relative to G2)
    ('G', 2): 0, ('A', 2): 1, ('B', 2): 2, ('C', 3): 3, ('D', 3): 4,
    ('E', 3): 5, ('F', 3): 6, ('G', 3): 7, ('A', 3): 8, ('B', 3): 9,
    ('C', 4): 10, ('D', 4): 11, ('E', 4): 12, ('F', 4): 13, ('G', 4): 14,

    ('F', 2): -1, ('E', 2): -2, ('D', 2): -3, ('C', 2): -4, ('B', 1): -5,
    ('A', 1): -6, ('G', 1): -7, ('F', 1): -8, ('E', 1): -9, ('D', 1): -10,
    ('C', 1): -11,
}

# Define Y positions for the bottom line of each staff relative to FRAME_HEIGHT
TREBLE_STAFF_BOTTOM_Y = FRAME_HEIGHT * 0.35
BASS_STAFF_BOTTOM_Y = FRAME_HEIGHT * 0.75

# Note colors based on frequency (using a simplified version of the provided map)
# Mapping note name (C, C#, D, etc.) to a base color
NOTE_COLORS = {
    'C': (255, 0, 0), 'C#': (255, 69, 0), 'D': (255, 140, 0), 'D#': (255, 215, 0),
    'E': (255, 255, 0), 'F': (173, 255, 47), 'F#': (0, 255, 0), 'G': (0, 128, 0),
    'G#': (0, 255, 255), 'A': (0, 0, 255), 'A#': (75, 0, 130), 'B': (148, 0, 211)
}
# Standard musical note symbols (requires a font like Bravura)
# Using standard symbols based on duration, ignoring the pitch-based symbols in the user's dict

DURATION_SYMBOLS = {
    'whole': '𝅝',
    'half': '𝅗𝅥',
    'quarter': '♩',
    'eighth': '♪',
    'sixteenth': '𝅘𝅥𝅮',
    # Add more if needed (32nd: '𝅘𝅥𝅯', 64th: '𝅘𝅥𝅰', etc.)
}

# --- Helper Functions ---

def get_staff_y_position(note_name_octave, clef):
    """Calculates the vertical pixel position for a note on the staff."""
    match = re.match(r'^([A-Ga-g]{1}[#b]?)(-?\d+)$', note_name_octave)
    if not match:
        print(f"Warning: Invalid note name format: {note_name_octave}")
        return -9999 # Indicate invalid note

    name = match.group(1).capitalize()
    octave = int(match.group(2))
    base_name = name.replace('#', '').replace('b', '') # Use base name for step calculation

    if (base_name, octave) not in NOTE_STAFF_STEPS:
        # Handle notes outside the defined range - might need more steps or raise error
        print(f"Warning: Note {note_name_octave} outside standard staff range.")
        return -9999 # Indicate it's off-staff

    steps = NOTE_STAFF_STEPS[(base_name, octave)]
    offset_from_bottom_line = steps * (STAFF_LINE_SPACING / 2)

    if clef == 'treble':
        return int(TREBLE_STAFF_BOTTOM_Y - offset_from_bottom_line)
    elif clef == 'bass':
        return int(BASS_STAFF_BOTTOM_Y - offset_from_bottom_line)
    else:
        raise ValueError("Invalid clef specified")

def get_clef_for_note(note_name_octave):
    """Simple rule to assign clef based on pitch."""
    try:
        midi = librosa.note_to_midi(note_name_octave)
        # Middle C (C4) is MIDI 60. Notes around/above C4 usually Treble, below Bass.
        # Let's put C4 and above on Treble, B3 and below on Bass.
        if midi >= 60: # C4 and above
            return 'treble'
        else: # B3 and below
            return 'bass'
    except Exception as e:
        print(f"Warning: Could not convert note {note_name_octave} to MIDI for clef selection: {e}")
        return 'treble' # Default to treble if conversion fails


def get_note_color(note_name, magnitude, max_magnitude):
    """Gets color based on note name and adjusts intensity based on magnitude."""
    match = re.match(r'^([A-Ga-g]{1}[#b]?)(-?\d+)$', note_name)
    if not match:
        base_name = 'C' # Default color if note name is weird
    else:
        base_name = match.group(1).capitalize().replace('#', '').replace('b', '')

    color = list(NOTE_COLORS.get(base_name, (150, 150, 150))) # Default to gray

    # Adjust brightness/saturation based on normalized magnitude
    norm_magnitude = magnitude / max_magnitude if max_magnitude > 0 else 0
    # Scale brightness from a base level (e.g., 0.4) to full brightness (1.0)
    brightness_factor = 0.4 + norm_magnitude * 0.6

    return tuple(int(c * brightness_factor) for c in color)


def extract_note_events(y, sr, min_duration_sec, magnitude_threshold_ratio):
    """
    Extracts discrete note events (pitch, start time, duration, magnitude)
    from frame-based pitch tracking data.
    """
    # Use piptrack for frame-level pitch and magnitude
    # Limit frequency range to typical musical notes
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=librosa.note_to_hz('C1'),
                                                fmax=librosa.note_to_hz('C8'))
    times = librosa.times_like(pitches, sr=sr)

    detected_notes = []
    current_note = None
    max_overall_magnitude = magnitudes.max() if magnitudes.size > 0 else 1.0

    # Threshold magnitude
    magnitude_threshold = max_overall_magnitude * magnitude_threshold_ratio

    for i in range(pitches.shape[1]):
        frame_time = times[i]
        # Find the dominant pitch in this frame
        frame_magnitudes = magnitudes[:, i]
        peak_idx = frame_magnitudes.argmax()
        peak_pitch = pitches[peak_idx, i]
        peak_magnitude = frame_magnitudes[peak_idx]

        # Consider a pitch detected only if its magnitude is above a threshold
        if peak_pitch > 0 and peak_magnitude >= magnitude_threshold:
            # Get note name, handling potential errors
            try:
                note_name = librosa.hz_to_note(peak_pitch, cents=False) # Get note name without cents
            except Exception as e:
                print(f"Warning: Could not convert frequency {peak_pitch:.2f} Hz to note: {e}")
                note_name = None # Skip this frame

            if note_name:
                if current_note and current_note['name'] == note_name:
                    # Extend current note
                    current_note['end_time'] = frame_time
                    current_note['duration'] = current_note['end_time'] - current_note['start_time']
                    current_note['magnitudes'].append(peak_magnitude)
                else:
                    # End previous note if it exists and meets min duration
                    if current_note and current_note['duration'] >= min_duration_sec:
                        current_note['avg_magnitude'] = np.mean(current_note['magnitudes'])
                        detected_notes.append(current_note)

                    # Start a new note
                    current_note = {
                        'name': note_name,
                        'start_time': frame_time,
                        'end_time': frame_time, # Will be updated
                        'duration': 0, # Will be updated
                        'magnitudes': [peak_magnitude]
                    }
        else:
            # No significant pitch detected, end current note
            if current_note and current_note['duration'] >= min_duration_sec:
                current_note['avg_magnitude'] = np.mean(current_note['magnitudes'])
                detected_notes.append(current_note)
            current_note = None # Reset

    # Add the last note if it exists and meets min duration
    if current_note and current_note['duration'] >= min_duration_sec:
        current_note['avg_magnitude'] = np.mean(current_note['magnitudes'])
        detected_notes.append(current_note)

    # Calculate max magnitude among detected notes for normalization
    max_note_magnitude = max((n['avg_magnitude'] for n in detected_notes), default=1.0)

    # Add normalized magnitude and clef to notes
    for note in detected_notes:
        note['norm_magnitude'] = note['avg_magnitude'] / max_note_magnitude
        note['clef'] = get_clef_for_note(note['name'])

    return detected_notes, max_note_magnitude

def quantize_note_durations(note_events, tempo, time_signature, tolerance=QUANTIZATION_TOLERANCE):
    """
    Quantizes note durations to standard musical note values based on tempo and time signature.
    Assumes the denominator of the time signature indicates the beat unit (e.g., 4 in 4/4 means quarter note is the beat).
    """
    if tempo <= 0:
        print("Warning: Tempo is zero or negative, cannot quantize durations.")
        for note in note_events:
            note['quantized_duration_sec'] = note['duration'] # Keep original duration
            note['note_value'] = 'unknown'
            note['symbol'] = '?' # Or a default symbol
        return note_events

    # Parse time signature
    try:
        numerator, denominator = map(int, time_signature.split('/'))
        if denominator <= 0 or not (denominator & (denominator - 1) == 0): # Denominator must be power of 2
            print(f"Warning: Invalid time signature denominator {denominator}. Using 4/4 logic.")
            denominator = 4
            numerator = 4
    except ValueError:
        print(f"Warning: Invalid time signature format '{time_signature}'. Using 4/4 logic.")
        numerator, denominator = 4, 4


    # Calculate duration of the beat unit (e.g., quarter note in 4/4)
    beat_duration_sec = 60.0 / tempo

    # Calculate durations of standard notes relative to the beat unit
    # Assuming denominator is the beat unit (e.g., 4 in 4/4 means quarter is the beat)
    unit_duration_sec = beat_duration_sec * (4 / denominator) # Duration of a quarter note

    standard_durations = {
        'whole': unit_duration_sec * 4,
        'half': unit_duration_sec * 2,
        'quarter': unit_duration_sec,
        'eighth': unit_duration_sec * 0.5,
        'sixteenth': unit_duration_sec * 0.25,
        '32nd': unit_duration_sec * 0.125,
        '64th': unit_duration_sec * 0.0625,
    }

    # Map standard note values to symbols
    duration_to_symbol = DURATION_SYMBOLS # Use the defined mapping

    quantized_notes = []
    for note in note_events:
        raw_duration = note['duration']
        best_match = None
        min_diff = float('inf')

        # Find the closest standard duration within tolerance
        for value, std_dur in standard_durations.items():
            # Calculate difference relative to the standard duration
            if std_dur > 0:
                diff = abs(raw_duration - std_dur) / std_dur
                if diff <= tolerance and diff < min_diff:
                    min_diff = diff
                    best_match = value
            # Also check against dotted notes? (Optional, adds complexity)
            # e.g., dotted quarter = 1.5 * unit_duration_sec
            # dotted eighth = 0.75 * unit_duration_sec
            # For simplicity, let's stick to simple durations first.

        if best_match:
            note['note_value'] = best_match
            note['quantized_duration_sec'] = standard_durations[best_match]
            note['symbol'] = duration_to_symbol.get(best_match, '?') # Get symbol, default to '?'
        else:
            # If no close match, maybe assign the closest one anyway or mark as unknown
            # Let's find the absolute closest for visualization, even if outside tolerance
            abs_best_match = None
            abs_min_diff = float('inf')
            for value, std_dur in standard_durations.items():
                if std_dur > 0:
                    diff = abs(raw_duration - std_dur)
                    if diff < abs_min_diff:
                        abs_min_diff = diff
                        abs_best_match = value

            if abs_best_match:
                note['note_value'] = abs_best_match
                note['quantized_duration_sec'] = standard_durations[abs_best_match]
                note['symbol'] = duration_to_symbol.get(abs_best_match, '?')
                # print(f"Note at {note['start_time']:.2f}s (duration {raw_duration:.3f}s) quantized to {best_match} ({standard_durations[best_match]:.3f}s) - outside tolerance.")
            else:
                note['note_value'] = 'unknown'
                note['quantized_duration_sec'] = raw_duration # Keep original if no match at all
                note['symbol'] = '?'
                # print(f"Note at {note['start_time']:.2f}s (duration {raw_duration:.3f}s) could not be quantized.")


        quantized_notes.append(note)

    return quantized_notes


# --- Drawing Functions ---

def draw_staff(draw, staff_bottom_y, line_spacing, num_lines=5):
    """Draws a single 5-line staff."""
    for i in range(num_lines):
        y = staff_bottom_y - i * line_spacing
        # Draw staff lines extending across the frame width where notation appears
        draw.line([(CLEF_WIDTH + TIME_SIG_WIDTH, y), (FRAME_WIDTH, y)], fill="black", width=2)

def draw_clef(draw, clef_type, staff_bottom_y, line_spacing, font):
    """Draws a clef symbol."""
    if clef_type == 'treble':
        # G clef position: loop around the G line (2nd line from bottom)
        clef_y = staff_bottom_y - 2 * line_spacing
        # Adjust vertical position for the symbol font
        # This might need empirical tuning based on the font size and symbol
        text_offset_y = 40 # Adjust this value as needed
        draw.text((5, clef_y - text_offset_y), "𝄞", fill="black",
font=font)
    elif clef_type == 'bass':
        # F clef position: dots around the F line (4th line from bottom)
        clef_y = staff_bottom_y - 4 * line_spacing
        # Adjust vertical position
        text_offset_y = 20 # Adjust this value as needed
        draw.text((5, clef_y - text_offset_y), "𝄢", fill="black",
font=font)

def draw_time_signature(draw, time_signature_str, staff_bottom_y, line_spacing, font):
    """Draws a time signature."""
    try:
        numerator, denominator = time_signature_str.split('/')
    except ValueError:
        numerator, denominator = '?', '?' # Handle invalid format

    # Position between clef and first bar line
    x_pos = CLEF_WIDTH + 5
    # Position numbers vertically centered around the staff
    # Top number usually on the 3rd space from bottom (between lines 3 and 4)
    # Bottom number usually on the 2nd space from bottom (between lines 1 and 2)
    y_pos_num = staff_bottom_y - 3.5 * line_spacing # Between lines 3 and 4
    y_pos_den = staff_bottom_y - 1.5 * line_spacing # Between lines 1 and 2

    draw.text((x_pos, y_pos_num), str(numerator), fill="black", font=font)
    draw.text((x_pos, y_pos_den), str(denominator), fill="black", font=font)


def draw_note(draw, note_event, current_time_sec, scroll_offset_x,
beat_duration_sec, font, max_magnitude):
    """Draws a single note symbol on the staff."""
    note_name_octave = note_event['name']
    start_time = note_event['start_time']
    norm_magnitude = note_event['norm_magnitude']
    clef = note_event['clef']
    symbol = note_event.get('symbol', '?') # Get the assigned symbol
    note_value = note_event.get('note_value', 'unknown')

    # Only draw if the note is visible within the frame's time window
    # A note is visible if its start time is within the window, plus some lookahead
    # Let's make the visible window from scroll_x to scroll_x + FRAME_WIDTH
    frame_start_time = current_time_sec - (scroll_offset_x / PIXELS_PER_BEAT) * beat_duration_sec
    frame_end_time = frame_start_time + (FRAME_WIDTH / PIXELS_PER_BEAT) * beat_duration_sec + beat_duration_sec # Add extra beat duration to see notes entering

    if not (start_time >= frame_start_time and start_time <= frame_end_time):
        return # Note is not visible in this frame

    # Calculate horizontal position based on time and scroll
    # Position relative to the very start of the music (time 0)
    # Add initial staff/clef/time sig width offset
    abs_x_position = (start_time / beat_duration_sec) * PIXELS_PER_BEAT + CLEF_WIDTH + TIME_SIG_WIDTH

    # Position relative to the left edge of the current frame
    frame_x_position = abs_x_position - scroll_offset_x

    # Calculate vertical position for the note head's center
    staff_y = get_staff_y_position(note_name_octave, clef)

    # Don't draw if position is invalid (e.g., outside defined staff range)
    if staff_y == -9999:
        return

    # --- Drawing the Symbol ---
    # The staff_y position is the *center* of where the note head would be.
    # Text drawing needs an anchor point (usually top-left). We need to adjust.
    # Get text bounding box to help with positioning
    try:
        # Use getbbox for more accurate positioning if available
        bbox = font.getbbox(symbol)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Adjust y position: staff_y is center, bbox[1] is top relative to baseline/origin
        # Vertical adjustment: move up by half the symbol height, plus the font's internal vertical offset (bbox[1])
        symbol_y = staff_y - text_height / 2 - bbox[1]
        # Adjust x position: center the symbol horizontally around frame_x_position
        symbol_x = frame_x_position - text_width / 2

    except AttributeError:
        # Fallback if getbbox is not available (older Pillow versions)
        # Use getsize (deprecated) or estimate size
        # size = font.getsize(symbol) # Deprecated
        # text_width, text_height = size
        # Estimate height based on staff spacing and font size
        estimated_height = STAFF_LINE_SPACING * 2 # Rough estimate
        symbol_y = staff_y - estimated_height / 2 # Center vertically based on estimate
        symbol_x = frame_x_position - font.getlength(symbol) / 2 # Center horizontally based on length

        print("Warning: Using fallback font positioning. Upgrade Pillow for better accuracy.")


    # Get color based on magnitude
    note_color = get_note_color(note_name_octave, note_event['avg_magnitude'], max_magnitude)

    # Draw the note symbol
    draw.text((symbol_x, symbol_y), symbol, fill=note_color, font=font)

    # --- Drawing Accidentals ---
    accidental = None
    if '#' in note_name_octave:
        accidental = '♯'
    elif 'b' in note_name_octave:
        accidental = '♭' # Use flat symbol

    if accidental:
        # Position accidental slightly to the left of the symbol
        accidental_x = symbol_x - 30 # Adjust this offset as needed
        accidental_y = symbol_y # Align vertically with the symbol
        draw.text((accidental_x, accidental_y), accidental, fill="black", font=font)

    # --- Drawing Ledger Lines ---
    # Ledger lines are drawn horizontally centered around the note symbol's position
    ledger_line_length = 30 # Length of the ledger line
    ledger_line_x_start = frame_x_position - ledger_line_length / 2
    ledger_line_x_end = frame_x_position + ledger_line_length / 2

    # Define the y coordinates of the standard staff lines for the relevant clef
    if clef == 'treble':
        staff_lines_y = [TREBLE_STAFF_BOTTOM_Y - i * STAFF_LINE_SPACING for i in range(5)]
    elif clef == 'bass':
        staff_lines_y = [BASS_STAFF_BOTTOM_Y - i * STAFF_LINE_SPACING for i in range(5)]
    else:
        staff_lines_y = [] # Should not happen if clef is valid

    # Check if the note is above the top staff line
    top_staff_y = staff_lines_y[-1] if staff_lines_y else -float('inf')
    if staff_y < top_staff_y - STAFF_LINE_SPACING/2: # Check if above the top space
        # Draw ledger lines above the staff
        # Start from the line just above the staff and go up until the note's vertical position
        # Ledger lines are drawn *on* the line Y coordinate
        for y_line_index in range(5, 20): # Check lines above the staff (index 5, 6, 7...)
            y_line = staff_lines_y[-1] - (y_line_index - 4) * STAFF_LINE_SPACING # Calculate Y for ledger line
            # Check if this ledger line is below or at the note's vertical position
            if y_line >= staff_y - STAFF_LINE_SPACING/2: # Draw if note is at or above this line/space
                draw.line([(ledger_line_x_start, y_line), (ledger_line_x_end, y_line)],
fill="black", width=2)
            else:
                break # Stop drawing ledger lines once we are below the note

    # Check if the note is below the bottom staff line
    bottom_staff_y = staff_lines_y[0] if staff_lines_y else float('inf')
    if staff_y > bottom_staff_y + STAFF_LINE_SPACING/2: # Check if below the bottom space
        # Draw ledger lines below the staff
        # Start from the line just below the staff and go down until the note's vertical position
        # Ledger lines are drawn *on* the line Y coordinate
        for y_line_index in range(1, 20): # Check lines below the staff (index -1, -2, ...)
            y_line = staff_lines_y[0] + y_line_index * STAFF_LINE_SPACING # Calculate Y for ledger line
            # Check if this ledger line is above or at the note's vertical position
            if y_line <= staff_y + STAFF_LINE_SPACING/2: # Draw if note is at or below this line/space
                draw.line([(ledger_line_x_start, y_line), (ledger_line_x_end, y_line)],
fill="black", width=2)
            else:
                break # Stop drawing ledger lines once we are above the note


def create_gif(image_folder, output_filename, duration_per_frame):
    """
    Create a GIF from images in the specified folder.

    Args:
        image_folder (str): Path to the folder containing images.
        output_filename (str): Path to save the output GIF.
        duration_per_frame (float): Duration of each frame in seconds.
    """
    # Collect image file paths
    filenames = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not filenames:
        print(f"No images found in {image_folder}")
        return

    # Read images in parallel
    print(f"Reading {len(filenames)} images...")
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(imageio.imread, filenames))

    # Save images as GIF
    print(f"Creating GIF: {output_filename}")
    # Use 'duration' argument which is in milliseconds for older imageio, or seconds for newer
    # Check imageio version or use a value that works for both (e.g., milliseconds)
    # Let's use duration in seconds, as per common imageio examples
    imageio.mimsave(output_filename, images, duration=duration_per_frame, loop=0)
    print(f"GIF saved as {output_filename}")

    # Optional: Clean up temporary files
    # import shutil
    # shutil.rmtree(image_folder)
    # print(f"Removed temporary folder: {image_folder}")


# --- Main Processing Function ---

def process_audio_to_gif(audio_path, output_gif_path, time_signature='4/4',
frame_rate=FRAME_RATE):
    """
    Processes an audio file to create a synchronized music notation GIF.
    Includes time signature handling and note duration quantization.
    """
    print(f"Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    print("Detecting beats and tempo...")
    try:
        # Use hop_length that's appropriate for beat tracking (e.g., 512 or 1024)
        # Default is 512, which is usually fine.
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Estimated tempo: {tempo:.2f} BPM")
        print(f"Detected beats at: {beats}")
    except Exception as e:
        print(f"Error during beat tracking: {e}")
        # Fallback: Assume a default tempo if detection fails
        tempo = 120.0
        # Simple beat approximation based on fallback tempo
        duration_sec = librosa.get_duration(y=y, sr=sr)
        beat_interval = 60 / tempo
        beats = np.arange(0, duration_sec, beat_interval)
        print(f"Beat tracking failed, using default tempo: {tempo} BPM and approximating beats.")


    beat_duration_sec = 60.0 / tempo if tempo > 0 else 0.5 # Duration of one beat in seconds
    if beat_duration_sec == 0:
        print("Tempo is zero, cannot calculate beat duration.")
        return

    # Extract raw note events (pitch, time, duration)
    print("Extracting raw note events...")
    detected_notes_raw, max_magnitude = extract_note_events(y, sr, MIN_NOTE_DURATION_SEC, MAGNITUDE_THRESHOLD_RATIO)
    print(f"Detected {len(detected_notes_raw)} raw note events.")

    # Quantize note durations
    print("Quantizing note durations...")
    quantized_notes = quantize_note_durations(detected_notes_raw, tempo, time_signature)
    print(f"Quantized {len(quantized_notes)} note events.")

    # Create a temporary directory for frames
    image_folder = "temp_frames"
    os.makedirs(image_folder, exist_ok=True)
    # Clear previous frames
    for file in os.listdir(image_folder):
        os.remove(os.path.join(image_folder, file))

    # Calculate GIF parameters
    total_duration_sec = librosa.get_duration(y=y, sr=sr)
    num_frames = int(total_duration_sec * frame_rate)
    time_per_frame = 1 / frame_rate

    # Calculate scrolling offset: position the current time (cursor) in the middle
    # The pixel position corresponding to time 't' is calculated based on beats and PIXELS_PER_BEAT
    pixels_at_current_time = lambda t: (t / beat_duration_sec) * PIXELS_PER_BEAT + CLEF_WIDTH + TIME_SIG_WIDTH
    # The scroll offset is the amount needed to shift the music left so 't' is at FRAME_WIDTH / 2
    scroll_offset_at_time = lambda t: pixels_at_current_time(t) - FRAME_WIDTH / 2

    # Load font for musical symbols
    try:
        # Assuming Bravura font is in a static or known location
        font_path = "static/Bravura.otf"
        if not os.path.exists(font_path):
            print(f"Warning: Bravura font not found at {font_path}. Using default PIL font. Musical symbols may not display correctly.")
            # Fallback to default font, symbols will likely fail
            font = ImageFont.load_default()
            symbol_font = ImageFont.load_default()
            print("Warning: Musical symbols will likely not display correctly.")
        else:
            # Font size for note symbols
            font_size_notes = int(STAFF_LINE_SPACING * 2.5) # Adjust size relative to staff spacing
            font = ImageFont.truetype(font_path, size=font_size_notes)
            # Font size for clefs/time sig
            font_size_symbols = int(STAFF_LINE_SPACING * 4) # Adjust size
            symbol_font = ImageFont.truetype(font_path, size=font_size_symbols)

    except Exception as e:
        print(f"Error loading font: {e}. Using default PIL font.")
        font = ImageFont.load_default()
        symbol_font = ImageFont.load_default() # Symbols will likely fail
        print("Warning: Musical symbols will likely not display correctly.")


    print("Generating GIF frames...")
    for i in range(num_frames):
        current_time_sec = i * time_per_frame
        artwork = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), "white")
        draw = ImageDraw.Draw(artwork)

        # Calculate the current horizontal scroll offset
        scroll_x = scroll_offset_at_time(current_time_sec)

        # Draw Grand Staff lines
        draw_staff(draw, TREBLE_STAFF_BOTTOM_Y, STAFF_LINE_SPACING)
        draw_staff(draw, BASS_STAFF_BOTTOM_Y, STAFF_LINE_SPACING)

        # Draw connecting line for Grand Staff
        grand_staff_top_y = TREBLE_STAFF_BOTTOM_Y - 4 * STAFF_LINE_SPACING
        grand_staff_bottom_y = BASS_STAFF_BOTTOM_Y
        draw.line([(CLEF_WIDTH + TIME_SIG_WIDTH / 2, grand_staff_top_y),
                         (CLEF_WIDTH + TIME_SIG_WIDTH / 2, grand_staff_bottom_y)], fill="black", width=4)

        # Draw clefs (fixed position on the frame)
        draw_clef(draw, 'treble', TREBLE_STAFF_BOTTOM_Y, STAFF_LINE_SPACING, symbol_font)
        draw_clef(draw, 'bass', BASS_STAFF_BOTTOM_Y, STAFF_LINE_SPACING, symbol_font)

        # Draw time signature (fixed position on the frame)
        draw_time_signature(draw, time_signature, TREBLE_STAFF_BOTTOM_Y, STAFF_LINE_SPACING, symbol_font) # Draw on Treble Staff height

        # Draw bar lines that are currently visible
        # Bar lines occur at the start of each measure.
        try:
            ts_numerator, ts_denominator = map(int, time_signature.split('/'))
            # Duration of a measure in seconds
            measure_duration_sec = (ts_numerator / ts_denominator) * 4 * beat_duration_sec # Assuming quarter note beat unit
        except ValueError:
            measure_duration_sec = 4 * beat_duration_sec # Fallback to 4/4 measure duration

        if measure_duration_sec > 0:
            num_measures = int(total_duration_sec // measure_duration_sec) + 3 # Draw a couple extra measures
            for m in range(num_measures):
                bar_time = m * measure_duration_sec
                bar_x_abs = pixels_at_current_time(bar_time) # Absolute pixel position from start of music
                bar_x_frame = bar_x_abs - scroll_x # Position on the current frame

                # Draw if the bar line is within the frame width (leaving space for clef/sig)
                if bar_x_frame >= CLEF_WIDTH + TIME_SIG_WIDTH and bar_x_frame <= FRAME_WIDTH:
                    draw.line([(bar_x_frame, grand_staff_top_y), (bar_x_frame, grand_staff_bottom_y)],
fill="black", width=2)

        # Draw notes that are currently visible
        for note in quantized_notes:
            draw_note(draw, note, current_time_sec, scroll_x, beat_duration_sec, font, max_magnitude)


        # Draw the playback cursor in the middle of the frame
        cursor_x = FRAME_WIDTH // 2
        draw.line([(cursor_x, 0), (cursor_x, FRAME_HEIGHT)], fill="red", width=2)


        # Save the frame
        frame_filename = os.path.join(image_folder, f"frame_{i:05d}.png")
        artwork.save(frame_filename)

    # Create the GIF
    create_gif(image_folder, output_gif_path, duration_per_frame=time_per_frame)

    # Optional: Clean up the temp_frames folder after GIF creation
    # import shutil
    # shutil.rmtree(image_folder)
    # print(f"Removed temporary folder: {image_folder}")


# Example Usage:
# Ensure you have an audio file named 'your_audio.wav' or change the path
# Ensure you have the Bravura.otf font file in a directory named 'static'
# process_audio_to_gif('your_audio.wav', 'music_visualization.gif', time_signature='4/4')
# process_audio_to_gif('waltz.wav', 'waltz_visualization.gif', time_signature='3/4') # Example for 3/4

# Previous code
def get_notes_from_audio(y,sr):
    
    # Perform pitch detection
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    
    # Get the dominant pitch (the one with the highest magnitude)
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()  # get index of max magnitude
        pitch.append(pitches[index, t])
    
    pitch = np.array(pitch)
    
    # Now convert the pitch to musical notes
    notes = []
    for p in pitch:
        if p > 0:
            note = librosa.hz_to_note(p)
            notes.append(note)
        
    
    return notes
# Frequency dictionary (from music_visualizer.py)
frequency_colors_update = {
    "A0": {"frequency": 27.50, "color": (139, 0, 0), "range": (27.50, 29.14)},
    "A#0/Bb0": {"frequency": 29.14, "color": (255, 69, 0), "range": (29.14, 30.87)},
    "B0": {"frequency": 30.87, "color": (204, 204, 0), "range": (30.87, 32.70)},
    "C1": {"frequency": 32.70, "color": (102, 152, 0), "range": (32.70, 34.65)},
    "C#1/Db1": {"frequency": 34.65, "color": (0, 100, 0), "range": (34.65, 36.71)},
    "D1": {"frequency": 36.71, "color": (0, 50, 69), "range": (36.71, 38.89)},
    "D#1/Eb1": {"frequency": 38.89, "color": (0, 0, 139), "range": (38.89, 41.20)},
    "E1": {"frequency": 41.20, "color": (75, 0, 130), "range": (41.20, 43.65)},
    "F1": {"frequency": 43.65, "color": (112, 0, 171), "range": (43.65, 46.25)},
    "F#1/Gb1": {"frequency": 46.25, "color": (148, 0, 211), "range": (46.25, 49.00)},
    "G1": {"frequency": 49.00, "color": (157, 0, 106), "range": (49.00, 51.91)},
    "G#1/Ab1": {"frequency": 51.91, "color": (165, 0, 0), "range": (51.91, 55.00)},
    "A1": {"frequency": 55.00, "color": (210, 0, 128), "range": (55.00, 58.27)},
    "A#1/Bb1": {"frequency": 58.27, "color": (255, 94, 0), "range": (58.27, 61.74)},
    "B1": {"frequency": 61.74, "color": (221, 221, 0), "range": (61.74, 65.41)},
    "C2": {"frequency": 65.41, "color": (111, 175, 0), "range": (65.41, 69.30)},
    "C#2/Db2": {"frequency": 69.30, "color": (0, 128, 0), "range": (69.30, 73.42)},
    "D2": {"frequency": 73.42, "color": (0, 64, 85), "range": (73.42, 77.78)},
    "D#2/Eb2": {"frequency": 77.78, "color": (0, 0, 170), "range": (77.78, 82.41)},
    "E2": {"frequency": 82.41, "color": (92, 0, 159), "range": (82.41, 87.31)},
    "F2": {"frequency": 87.31, "color": (119, 0, 96), "range": (87.31, 92.50)},
    "F#2/Gb2": {"frequency": 92.50, "color": (159, 0, 226), "range": (92.50, 98.00)},
    "G2": {"frequency": 98.00, "color": (175, 0, 113), "range": (98.00, 103.83)},
    "G#2/Ab2": {"frequency": 103.83, "color": (191, 0, 0), "range": (103.83, 110.00)},
    "A2": {"frequency": 110.00, "color": (223, 59, 128), "range": (110.00, 116.54)},
    "A#2/Bb2": {"frequency": 116.54, "color": (255, 119, 0), "range": (116.54, 123.47)},
    "B2": {"frequency": 123.47, "color": (238, 238, 0), "range": (123.47, 130.81)},
    "C3": {"frequency": 130.81, "color": (119, 159, 0), "range": (130.81, 138.59)},
    "C#3/Db3": {"frequency": 138.59, "color": (0, 160, 0), "range": (138.59, 146.83)},
    "D3": {"frequency": 146.83, "color": (0, 80, 100), "range": (146.83, 155.56)},
    "D#3/Eb3": {"frequency": 155.56, "color": (0, 0, 200), "range": (155.56, 164.81)},
    "E3": {"frequency": 164.81, "color": (109, 0, 188), "range": (164.81, 174.61)},
    "F3": {"frequency": 174.61, "color": (140, 0, 215), "range": (174.61, 185.00)},
    "F#3/Gb3": {"frequency": 185.00, "color": (170, 0, 241), "range": (185.00, 196.00)},
    "G3": {"frequency": 196.00, "color": (194, 0, 121), "range": (196.00, 207.65)},
    "G#3/Ab3": {"frequency": 207.65, "color": (217, 0, 0), "range": (207.65, 220.00)},
    "A3": {"frequency": 220.00, "color": (236, 72, 0), "range": (220.00, 233.08)},
    "A#3/Bb3": {"frequency": 233.08, "color": (255, 144, 0), "range": (233.08, 246.94)},
    "B3": {"frequency": 246.94, "color": (255, 255, 0), "range": (246.94, 261.63)},
    "C4": {"frequency": 261.63, "color": (128, 224, 0), "range": (261.63, 277.18)},
    "C#4/Db4": {"frequency": 277.18, "color": (0, 192, 0), "range": (277.18, 293.66)},
    "D4": {"frequency": 293.66, "color": (0, 96, 115), "range": (293.66, 311.13)},
    "D#4/Eb4": {"frequency": 311.13, "color": (0, 0, 230), "range": (311.13, 329.63)},
    "E4": {"frequency": 329.63, "color": (126, 0, 217), "range": (329.63, 349.23)},
    "F4": {"frequency": 349.23, "color": (159, 26, 236), "range": (349.23, 369.99)},
    "F#4/Gb4": {"frequency": 369.99, "color": (191, 51, 255), "range": (369.99, 392.00)},
    "G4": {"frequency": 392.00, "color": (217, 26, 128), "range": (392.00, 415.30)},
    "G#4/Ab4": {"frequency": 415.30, "color": (243, 0, 0), "range": (415.30, 440.00)},
    "A4": {"frequency": 440.00, "color": (249, 85, 0), "range": (440.00, 466.16)},
    "A#4/Bb4": {"frequency": 466.16, "color": (255, 169, 0), "range": (466.16, 493.88)},
    "B4": {"frequency": 493.88, "color": (255, 255, 51), "range": (493.88, 523.25)},
    "C5": {"frequency": 523.25, "color": (153, 255, 51), "range": (523.25, 554.37)},
    "C#5/Db5": {"frequency": 554.37, "color": (51, 255, 51), "range": (554.37, 587.33)},
    "D5": {"frequency": 587.33, "color": (51, 204, 204), "range": (587.33, 622.25)},
    "D#5/Eb5": {"frequency": 622.25, "color": (51, 51, 255), "range": (622.25, 659.25)},
    "E5": {"frequency": 659.25, "color": (128, 51, 255), "range": (659.25, 698.46)},
    "F5": {"frequency": 698.46, "color": (159, 87, 255), "range": (698.46, 739.99)},
    "F#5/Gb5": {"frequency": 739.99, "color": (190, 123, 255), "range": (739.99, 783.99)},
    "G5": {"frequency": 783.99, "color": (204, 87, 128), "range": (783.99, 830.61)},
    "G#5/Ab5": {"frequency": 830.61, "color": (255, 51, 51), "range": (830.61, 880.00)},
    "A5": {"frequency": 880.00, "color": (255, 128, 102), "range": (880.00, 932.33)},
    "A#5/Bb5": {"frequency": 932.33, "color": (255, 204, 102), "range": (932.33, 987.77)},
    "B5": {"frequency": 987.77, "color": (255, 255, 102), "range": (987.77, 1046.50)},
    "C6": {"frequency": 1046.50, "color": (179, 255, 102), "range": (1046.50, 1108.73)},
    "C#6/Db6": {"frequency": 1108.73, "color": (102, 255, 102), "range": (1108.73, 1174.66)},
    "D6": {"frequency": 1174.66, "color": (102, 204, 204), "range": (1174.66, 1244.51)},
    "D#6/Eb6": {"frequency": 1244.51, "color": (102, 102, 255), "range": (1244.51, 1318.51)},
    "E6": {"frequency": 1318.51, "color": (153, 102, 255), "range": (1318.51, 1396.91)},
    "F6": {"frequency": 1396.91, "color": (177, 128, 255), "range": (1396.91, 1479.98)},
    "F#6/Gb6": {"frequency": 1479.98, "color": (201, 153, 255), "range": (1479.98, 1567.98)},
    "G6": {"frequency": 1567.98, "color": (209, 128, 153), "range": (1567.98, 1661.22)},
    "G#6/Ab6": {"frequency": 1661.22, "color": (255, 102, 102), "range": (1661.22, 1760.00)},
    "A6": {"frequency": 1760.00, "color": (255, 153, 128), "range": (1760.00, 1864.66)},
    "A#6/Bb6": {"frequency": 1864.66, "color": (255, 204, 153), "range": (1864.66, 1975.53)},
    "B6": {"frequency": 1975.53, "color": (255, 255, 153), "range": (1975.53, 2093.00)},
    "C7": {"frequency": 2093.00, "color": (204, 255, 153), "range": (2093.00, 2217.46)},
    "C#7/Db7": {"frequency": 2217.46, "color": (153, 255, 153), "range": (2217.46, 2349.32)},
    "D7": {"frequency": 2349.32, "color": (153, 204, 204), "range": (2349.32, 2489.02)},
    "D#7/Eb7": {"frequency": 2489.02, "color": (153, 153, 255), "range": (2489.02, 2637.02)},
    "E7": {"frequency": 2637.02, "color": (197, 153, 255), "range": (2637.02, 2793.83)},   
    "F7": {"frequency": 2793.83, "color": (222, 176, 255), "range": (2793.83, 2959.96)},
    "F#7/Gb7": {"frequency": 2959.96, "color": (246, 198, 255), "range": (2959.96, 3135.96)},
    "G7": {"frequency": 3135.96, "color": (251, 176, 204), "range": (3135.96, 3322.44)},
    "G#7/Ab7": {"frequency": 3322.44, "color": (255, 153, 153), "range": (3322.44, 3520.00)},
    "A7": {"frequency": 3520.00, "color": (255, 194, 176), "range": (3520.00, 3729.31)},
    "A#7/Bb7": {"frequency": 3729.31, "color": (255, 234, 198), "range": (3729.31, 3951.07)},
    "B7": {"frequency": 3951.07, "color": (255, 255, 204), "range": (3951.07, 4186.01)},
    "C8": {"frequency": 4186.01, "color": (144, 238, 144), "range": (4186.01, 4434.92)}
}

freq_symbols={
    "A0": {"frequency": 27.50, "color": [139, 0, 0], "range": [27.50, 29.14], "symbol": "♩"},
    "A#0/Bb0": {"frequency": 29.14, "color": [255, 69, 0], "range": [29.14, 30.87], "symbol": "♯"},
    "B0": {"frequency": 30.87, "color": [204, 204, 0], "range": [30.87, 32.70], "symbol": "♩"},
    "C1": {"frequency": 32.70, "color": [102, 152, 0], "range": [32.70, 34.65], "symbol": "♩"},
    "C#1/Db1": {"frequency": 34.65, "color": [0, 100, 0], "range": [34.65, 36.71], "symbol": "♯"},
    "D1": {"frequency": 36.71, "color": [0, 50, 69], "range": [36.71, 38.89], "symbol": "♩"},
    "D#1/Eb1": {"frequency": 38.89, "color": [0, 0, 139], "range": [38.89, 41.20], "symbol": "♯"},
    "E1": {"frequency": 41.20, "color": [75, 0, 130], "range": [41.20, 43.65], "symbol": "♩"},
    "F1": {"frequency": 43.65, "color": [112, 0, 171], "range": [43.65, 46.25], "symbol": "♩"},
    "F#1/Gb1": {"frequency": 46.25, "color": [148, 0, 211], "range": [46.25, 49.00], "symbol": "♯"},
    "G1": {"frequency": 49.00, "color": [157, 0, 106], "range": [49.00, 51.91], "symbol": "♩"},
    "G#1/Ab1": {"frequency": 51.91, "color": [165, 0, 0], "range": [51.91, 55.00], "symbol": "♯"},
    "A1": {"frequency": 55.00, "color": [210, 0, 128], "range": [55.00, 58.27], "symbol": "♩"},
    "A#1/Bb1": {"frequency": 58.27, "color": [255, 94, 0], "range": [58.27, 61.74], "symbol": "♯"},
    "B1": {"frequency": 61.74, "color": [221, 221, 0], "range": [61.74, 65.41], "symbol": "♩"},
    "C2": {"frequency": 65.41, "color": [111, 175, 0], "range": [65.41, 69.30], "symbol": "♩"},
    "C#2/Db2": {"frequency": 69.30, "color": [0, 128, 0], "range": [69.30, 73.42], "symbol": "♯"},
    "D2": {"frequency": 73.42, "color": [0, 64, 85], "range": [73.42, 77.78], "symbol": "♩"},
    "D#2/Eb2": {"frequency": 77.78, "color": [0, 0, 170], "range": [77.78, 82.41], "symbol": "♯"},
    "E2": {"frequency": 82.41, "color": [92, 0, 159], "range": [82.41, 87.31], "symbol": "♩"},
    "F2": {"frequency": 87.31, "color": [119, 0, 96], "range": [87.31, 92.50], "symbol": "♩"},
    "F#2/Gb2": {"frequency": 92.50, "color": [159, 0, 226], "range": [92.50, 98.00], "symbol": "♯"},
    "G2": {"frequency": 98.00, "color": [175, 0, 113], "range": [98.00, 103.83], "symbol": "♩"},
    "G#2/Ab2": {"frequency": 103.83, "color": [191, 0, 0], "range": [103.83, 110.00], "symbol": "♯"},
    "A2": {"frequency": 110.00, "color": [223, 59, 128], "range": [110.00, 116.54], "symbol": "♩"},
    "A#2/Bb2": {"frequency": 116.54, "color": [255, 119, 0], "range": [116.54, 123.47], "symbol": "♯"},
    "B2": {"frequency": 123.47, "color": [238, 238, 0], "range": [123.47, 130.81], "symbol": "♩"},
    
        "C3": {"frequency": 130.81, "color": [119, 159, 0], "range": [130.81, 138.59], "symbol": "♩"},
    "C#3/Db3": {"frequency": 138.59, "color": [0, 160, 0], "range": [138.59, 146.83], "symbol": "♯"}, 
    "D3": {"frequency": 146.83, "color": [0, 80, 100], "range": [146.83, 155.56], "symbol": "♩"},
    "D#3/Eb3": {"frequency": 155.56, "color": [0, 0, 200], "range": [155.56, 164.81], "symbol": "♯"},
    "E3": {"frequency": 164.81, "color": [109, 0, 188], "range": [164.81, 174.61], "symbol": "♩"},
    "F3": {"frequency": 174.61, "color": [140, 0, 215], "range": [174.61, 185.00], "symbol": "♩"},
    "F#3/Gb3": {"frequency": 185.00, "color": [170, 0, 241], "range": [185.00, 196.00], "symbol": "♯"},
    "G3": {"frequency": 196.00, "color": [194, 0, 121], "range": [196.00, 207.65], "symbol": "♩"},
    "G#3/Ab3": {"frequency": 207.65, "color": [217, 0, 0], "range": [207.65, 220.00], "symbol": "♯"},
    "A3": {"frequency": 220.00, "color": [236, 72, 0], "range": [220.00, 233.08], "symbol": "♩"},
    "A#3/Bb3": {"frequency": 233.08, "color": [255, 144, 0], "range": [233.08, 246.94], "symbol": "♯"},
    "B3": {"frequency": 246.94, "color": [250, 250, 0], "range": [246.94, 261.63], "symbol": "♩"},

    "C4": {"frequency": 261.63, "color": [128, 224, 0], "range": [261.63, 277.18], "symbol": "♩"},
    "C#4/Db4": {"frequency": 277.18, "color": [0, 192, 0], "range": [277.18, 293.66], "symbol": "♯"},
    "D4": {"frequency": 293.66, "color": [0, 96, 115], "range": [293.66, 311.13], "symbol": "♩"},
    "D#4/Eb4": {"frequency": 311.13, "color": [0, 0, 230], "range": [311.13, 329.63], "symbol": "♯"},
    "D#4": {"frequency": 311.13, "color": [0, 0, 230], "range": [311.13, 329.63], "symbol": "♯"},
    "E4": {"frequency": 329.63, "color": [126, 0, 217], "range": [329.63, 349.23], "symbol": "♩"},
    "F4": {"frequency": 349.23, "color": [159, 26, 236], "range": [349.23, 369.99], "symbol": "♩"},
    "F#4/Gb4": {"frequency": 369.99, "color": [191, 51, 255], "range": [369.99, 392.00], "symbol": "♯"},
    "G4": {"frequency": 392.00, "color": [217, 26, 128], "range": [392.00, 415.30], "symbol": "♩"},
    "G#4/Ab4": {"frequency": 415.30, "color": [243, 0, 0], "range": [415.30, 440.00], "symbol": "♯"},
    "A4": {"frequency": 440.00, "color": [249, 85, 0], "range": [440.00, 466.16], "symbol": "♩"},
    "A#4/Bb4": {"frequency": 466.16, "color": [255, 169, 0], "range": [466.16, 493.88], "symbol": "♯"},
    "B4": {"frequency": 493.88, "color": [255, 255, 51], "range": [493.88, 523.25], "symbol": "♩"},
    
     "C5": {"frequency": 523.25, "color": [153, 255, 51], "range": [523.25, 554.37], "symbol": "♩"},
    "C#5/Db5": {"frequency": 554.37, "color": [51, 255, 51], "range": [554.37, 587.33], "symbol": "♯"},
    "D5": {"frequency": 587.33, "color": [51, 204, 204], "range": [587.33, 622.25], "symbol": "♪"},
    "D#5/Eb5": {"frequency": 622.25, "color": [51, 51, 255], "range": [622.25, 659.26], "symbol": "♭"},
    "E5": {"frequency": 659.26, "color": [128, 51, 255], "range": [659.26, 698.46], "symbol": "𝅘𝅥𝅮"},
    "F5": {"frequency": 698.46, "color": [159, 87, 255], "range": [698.46, 739.99], "symbol": "♩"},
    "F#5/Gb5": {"frequency": 739.99, "color": [190, 123, 255], "range": [739.99, 783.99], "symbol": "♯"},
   "G5": {"frequency": 783.99, "color": [204, 87, 128], "range": [783.99, 830.61], "symbol": "♫"},
    "G#5/Ab5": {"frequency": 830.61, "color": [255, 51, 51], "range": [830.61, 880.00], "symbol": "♭"},
   "A5": {"frequency": 880.00, "color": [255, 128, 102], "range": [880.00, 932.33], "symbol": "𝅗𝅥"},
    "A#5/Bb5": {"frequency": 932.33, "color": [255, 204, 102], "range": [932.33, 987.77], "symbol": "♯"},
   "B5": {"frequency": 987.77, "color": [255, 255, 102], "range": [987.77, 1046.50], "symbol": "𝅘𝅥"},
    
        "C6": {"frequency": 1046.50, "color": [179, 255, 102], "range": [1046.50, 1108.73], "symbol": "♩"},
    "C#6/Db6": {"frequency": 1108.73, "color": [102, 255, 102], "range": [1108.73, 1174.66], "symbol": "♯"},
    "D6": {"frequency": 1174.66, "color": [102, 204, 204], "range": [1174.66, 1244.51], "symbol": "♪"},
    "D#6/Eb6": {"frequency": 1244.51, "color": [102, 102, 255], "range": [1244.51, 1318.51], "symbol": "♭"},
    "E6": {"frequency": 1318.51, "color": [153, 102, 255], "range": [1318.51, 1396.91], "symbol": "𝅘𝅥𝅮"},
    "F6": {"frequency": 1396.91, "color": [171, 128, 255], "range": [1396.91, 1479.98], "symbol": "♩"},
    "F#6/Gb6": {"frequency": 1479.98, "color": [201, 153, 255], "range": [1479.98, 1567.98], "symbol": "♯"},
    "G6": {"frequency": 1567.98, "color": [209, 128, 153], "range": [1567.98, 1661.22], "symbol": "♫"},
    "G#6/Ab6": {"frequency": 1661.22, "color": [255, 102, 102], "range": [1661.22, 1760.00], "symbol": "♭"},
    "A6": {"frequency": 1760.00, "color": [255, 153, 128], "range": [1760.00, 1864.66], "symbol": "𝅗𝅥"},
    "A#6/Bb6": {"frequency": 1864.66, "color": [255, 204, 153], "range": [1864.66, 1975.53], "symbol": "♯"},
   "B6": {"frequency": 1975.53, "color": [255, 255, 153], "range": [1975.53, 2093.00], "symbol": "𝅘𝅥"},

    "C7": {"frequency": 2093.00, "color": [204, 255, 153], "range": [2093.00, 2217.46], "symbol": "♩"},
    "C#7/Db7": {"frequency": 2217.46, "color": [153, 255, 153], "range": [2217.46, 2349.32], "symbol": "♯"},
    "D7": {"frequency": 2349.32, "color": [153, 204, 204], "range": [2349.32, 2489.02], "symbol": "♪"},
    "D#7/Eb7": {"frequency": 2489.02, "color": [153, 153, 255], "range": [2489.02, 2637.02], "symbol": "♭"},
    "E7": {"frequency": 2637.02, "color": [197, 153, 255], "range": [2637.02, 2793.83], "symbol": "𝅘𝅥𝅮"},
    "F7": {"frequency": 2793.83, "color": [222, 176, 255], "range": [2793.83, 2959.96], "symbol": "♩"},
    "F#7/Gb7": {"frequency": 2959.96, "color": [246, 198, 255], "range": [2959.96, 3135.96], "symbol": "♯"},
    "G7": {"frequency": 3135.96, "color": [255, 176, 204], "range": [3135.96, 3322.44], "symbol": "♫"},
    "G#7/Ab7": {"frequency": 3322.44, "color": [255, 153, 153], "range": [3322.44, 3520.00], "symbol": "♭"},
    "A7": {"frequency": 3520.00, "color": [255, 194, 176], "range": [3520.00, 3729.32], "symbol": "𝅗𝅥"},
    "A#7/Bb7": {"frequency": 3729.32, "color": [255, 234, 198], "range": [3729.32, 3951.07], "symbol": "♯"},
   "B7": {"frequency": 3951.07, "color": [255, 255, 204], "range": [3951.07, 4186.01], "symbol": "𝅘𝅥"},
    "C8": {"frequency": 4186.01, "color": [144, 238, 144], "range": [4186.01, 4434.92], "symbol": "♩"},
   
}


note_number_dict = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11
}


# Function to combine images into a GIF
def create_gif(image_folder, output_filename, duration):
    # Get list of all image files in the specified folder
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            images.append(imageio.imread(os.path.join(image_folder, filename)))

    # Create the GIF and save it to the specified output file
    iio.mimsave(output_filename, images, duration=duration,loop=0)
    print(f"GIF saved as {output_filename}")



from concurrent.futures import ThreadPoolExecutor

def create_gif(image_folder, output_filename, duration=1.0):
    """
    Create a GIF from images in the specified folder.
    
    Args:
        image_folder (str): Path to the folder containing images.
        output_filename (str): Path to save the output GIF.
        duration (float): Duration per frame in seconds.
    """
    # Collect image file paths
    filenames = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # Read images in parallel
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(imageio.imread, filenames))

    # Save images as GIF
    imageio.mimsave(output_filename, images, duration=duration, loop=0)
    print(f"GIF saved as {output_filename}")

import re

# Map note names to semitone offsets
NOTE_OFFSETS = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11
}

def note_to_midi(note: str) -> int:
    """Convert a note like 'A4' or 'C#5' to a MIDI note number."""
    match = re.match(r'^([A-Ga-g]{1}[#b]?)(-?\d+)$', note)
    if not match:
        raise ValueError(f"Invalid note format: {note}")
    name = match.group(1).capitalize()
    octave = int(match.group(2))
    if name not in NOTE_OFFSETS:
        raise ValueError(f"Invalid note name: {name}")
    return 12 * (octave + 1) + NOTE_OFFSETS[name]

def get_string_number_mod12(note: str, reference_note: str = 'A4', reference_string: int = 0) -> int:
    """Map a note to a 12-string system (returns 0-11)."""
    midi_note = note_to_midi(note)
    ref_midi = note_to_midi(reference_note)
    diff = midi_note - ref_midi
    return (reference_string + diff) % 12


def process_audio_to_gif(audio_path,gif_path):
    audio_data_org, sr = librosa.load(audio_path)      
    counter=0

    # Canvas setup
    width=1000
    height=220
        
    t=len(audio_data_org)//sr
    
    files = os.listdir('music')
    
    
    for f in files:
        os.remove('music/'+f)

    nn=int(sr*0.7);

    
    d=len(list(range(0,sr*t-nn,nn)))
    canvas_size = (width, height*d)
    artwork = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(artwork)
    for tt in range(0,sr*t-nn,nn):
        
        
        audio_data=audio_data_org[tt:tt+nn]
        notes=get_notes_from_audio(audio_data,sr)
        lower_freq_lines = [120, 130, 140, 150, 160,170]  
        lower_freq_lines = [counter*height+ii for ii in lower_freq_lines]
        
        higher_freq_lines = [30, 40, 50, 60, 70,80]  
        higher_freq_lines = [counter*height+ii for ii in higher_freq_lines]

        
        Num_lines=6
        for i in range(Num_lines):           
            y=lower_freq_lines[i]
            draw.line([0, y, width, y], fill="black", width=2)
            
        for i in range(Num_lines):           
            y=higher_freq_lines[i]
            draw.line([0, y,width, y], fill="black", width=2)
            
        draw.line([0, 0, 0, height*d], fill="black", width=7)
        draw.line([width, 0, width, height*d], fill="black", width=7)

        C=(np.random.randint(255),np.random.randint(255),np.random.randint(255))

        font = ImageFont.truetype("static/Bravura.otf", size=50)
        y_position=lower_freq_lines[0]-80
        draw.text((5,y_position),"𝄢", fill=C, font=font)  

        y_position=higher_freq_lines[0]-60
        draw.text((5,y_position),"𝄞", fill=C, font=font)
        counter+=1
        
        for i,note in enumerate(notes):
                 
            n=note.replace("♯","#")
            
            if n not in freq_symbols:
                continue
            string_number = get_string_number_mod12(n[:3])
            
                
            symbol=freq_symbols[n]["symbol"]
            symbol=symbol.replace("#","♯")
            color=tuple(freq_symbols[n]["color"])
            freq_value=freq_symbols[n]["frequency"]
           
            total_notes = len(notes)
            spacing = width / total_notes
            note_index = list(freq_symbols.keys()).index(n)
            x_position = 5+int(width*1/len(notes)) + int(width*i/len(notes))
            
            print(n,string_number,symbol)
            
            
            if string_number<6:
                 y_position = lower_freq_lines[0]-60-2*string_number
                 draw.text((x_position,y_position),symbol, fill=color, font=font)                
            else:
                y_position = higher_freq_lines[0]-60-(2*(string_number-6))
                draw.text((x_position,y_position),symbol, fill=color, font=font)

            
                
    # Step 4: Save the artwork
    artwork.save('static/output/music.png', "PNG")

    # Example usage
    image_folder = 'music'  # Replace with your folder path
    output_filename = gif_path       # Desired output file name
    #len(os.listdir(image_folder))
    #create_gif(image_folder, output_filename,500)

