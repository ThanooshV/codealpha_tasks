import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from music21 import stream, note, chord
import pickle
import random

# Load notes
with open('notes.pkl', 'rb') as f:
    notes = pickle.load(f)

pitch_names = sorted(set(notes))
note_to_int = {note:i for i, note in enumerate(pitch_names)}
int_to_note = {i:note for i, note in enumerate(pitch_names)}

# Load model
model = load_model('music_model.h5')

sequence_length = 100
start = np.random.randint(0, len(notes) - sequence_length - 1)
pattern = notes[start:start + sequence_length]
pattern_int = [note_to_int[n] for n in pattern]

# Generate notes
prediction_output = []

for note_index in range(200):
    prediction_input = np.reshape(pattern_int, (1, len(pattern_int), 1)) / float(len(pitch_names))
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern_int.append(index)
    pattern_int = pattern_int[1:]

# Convert to MIDI
offset = 0
output_notes = []

for pattern in prediction_output:
    if '.' in pattern or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_in_chord = [note.Note(int(n)) for n in notes_in_chord]
        new_chord = chord.Chord(notes_in_chord)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output/generated_song.mid')
print("Generated MIDI saved in output/generated_song.mid")
