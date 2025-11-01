# train_model.py
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization as BatchNorm

# -----------------------------
# Step 1: Load MIDI files
# -----------------------------
midi_files = glob.glob("midi_songs/*.mid")
notes = []

if len(midi_files) == 0:
    raise Exception("No MIDI files found in 'midi_songs' folder!")

for file in midi_files:
    midi = converter.parse(file)
    parts = instrument.partitionByInstrument(midi)
    if parts:  # file has instruments
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print(f"Total notes found: {len(notes)}")

# -----------------------------
# Step 2: Prepare sequences
# -----------------------------
sequence_length = min(20, len(notes) - 1)  # safe for small datasets
n_vocab = len(set(notes))

note_to_int = {note: i for i, note in enumerate(sorted(set(notes)))}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

print(f"Total sequences created: {len(network_input)}")

# -----------------------------
# Step 3: Reshape input & normalize
# -----------------------------
network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
network_input = network_input / float(n_vocab)
network_output = to_categorical(network_output, num_classes=n_vocab)

# Save mapping
with open("note_to_int.pkl", "wb") as f:
    pickle.dump(note_to_int, f)

# -----------------------------
# Step 4: Build LSTM model
# -----------------------------
model = Sequential()
model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(BatchNorm())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# -----------------------------
# Step 5: Train the model
# -----------------------------
model.fit(network_input, network_output, epochs=50, batch_size=64)

# Save the model
model.save("music_model.h5")
print("Model saved as music_model.h5")
