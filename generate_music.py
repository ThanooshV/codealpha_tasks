import os
import pickle
import numpy as np
from music21 import instrument, note, chord, stream
from keras.models import load_model
import subprocess  # to open the file automatically

# === Folders ===
base_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_dir, "data")
notes_path = os.path.join(data_folder, "notes.pkl")
note_to_int_path = os.path.join(base_dir, "note_to_int.pkl")
model_path = os.path.join(base_dir, "music_model.h5")

# === Load notes ===
with open(notes_path, "rb") as f:
    notes = pickle.load(f)

print("✅ Notes loaded successfully.")
print("Total number of notes in file:", len(notes))
print("Example notes:", notes[:20])

if len(notes) == 0:
    raise ValueError("Your notes dataset is empty. Add more MIDI files and regenerate notes.pkl.")

# === Safe note_to_int mapping ===
pitchnames = sorted(set(notes))

if os.path.exists(note_to_int_path):
    with open(note_to_int_path, "rb") as f:
        note_to_int = pickle.load(f)
        for n in pitchnames:
            if n not in note_to_int:
                note_to_int[n] = len(note_to_int)
else:
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    with open(note_to_int_path, "wb") as f:
        pickle.dump(note_to_int, f)

print(f"✅ note_to_int mapping contains {len(note_to_int)} items")

# === Load trained model ===
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found! Place your trained model here.")

model = load_model(model_path)
print("✅ Model loaded successfully")

# === Prepare input sequences ===
n_vocab = len(set(notes))
sequence_length = min(20, max(5, len(notes)//2))
print(f"Using sequence_length = {sequence_length}")

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)
if n_patterns == 0:
    raise ValueError("No input sequences created. Dataset too small.")

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)

print(f"✅ Prepared {n_patterns} input sequences for generation.")

# === Generate new music ===
start = np.random.randint(0, len(network_input) - 1)
pattern = network_input[start]
int_to_note = {number: note for note, number in note_to_int.items()}

prediction_output = []

for note_index in range(300):  # generate 300 notes
    prediction_input = np.reshape(pattern, (1, sequence_length, 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)

    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]

# === Convert output to MIDI ===
offset = 0
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_objs = [note.Note(int(n)) for n in notes_in_chord]
        for n in notes_objs:
            n.storedInstrument = instrument.Piano()
        new_chord = chord.Chord(notes_objs)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

# === Save MIDI in Documents folder and open automatically ===
output_path = os.path.join(os.path.expanduser("~"), "Documents", "generated_music.mid")
midi_stream.write('midi', fp=output_path)
print(f"✅ Music generated and saved as: {output_path}")

# Open MIDI automatically
try:
    if os.name == 'nt':  # Windows
        os.startfile(output_path)
    else:
        subprocess.call(["open", output_path])
except Exception as e:
    print("Could not open automatically:", e)
