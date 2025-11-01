# save as preprocess_midi.py or run in notebook cell
import os
from music21 import converter, note, chord, stream, duration
import numpy as np
import pickle
from tqdm import tqdm

MIDI_DIR = "data/midi"     # put your .mid files here
SEQ_LEN = 64              # sequence length for training

def midi_to_tokens(midi_path):
    """
    Parse one MIDI file and produce a sequence of tokens:
    NOTE_<pitch> (e.g. NOTE_C4), CHORD_C4_E4_G4, REST, DUR_<quarterLength>
    We'll interleave duration tokens, e.g. NOTE_C4 DUR_0.25 NOTE_D4 DUR_0.25 ...
    """
    parsed = converter.parse(midi_path)
    tokens = []
    # flatten to parts and get events in time order
    flat = parsed.flat.notesAndRests
    for e in flat:
        if isinstance(e, note.Note):
            tokens.append(f"NOTE_{e.nameWithOctave}")
            tokens.append(f"DUR_{e.quarterLength}")
        elif isinstance(e, note.Rest):
            tokens.append("REST")
            tokens.append(f"DUR_{e.quarterLength}")
        elif isinstance(e, chord.Chord):
            chord_name = "_".join(n.nameWithOctave for n in e.pitches)
            tokens.append(f"CHORD_{chord_name}")
            tokens.append(f"DUR_{e.quarterLength}")
    return tokens

# collect all tokens from dataset
all_tokens = []
files = [os.path.join(MIDI_DIR, f) for f in os.listdir(MIDI_DIR) if f.lower().endswith(('.mid', '.midi'))]
print(f"Found {len(files)} MIDI files")
for f in tqdm(files):
    try:
        toks = midi_to_tokens(f)
        if len(toks) > 10:
            all_tokens.extend(toks)
    except Exception as e:
        print("Error parsing", f, e)

# build vocabulary
unique_tokens = sorted(list(set(all_tokens)))
token2idx = {t:i for i,t in enumerate(unique_tokens)}
idx2token = {i:t for t,i in token2idx.items()}

# save vocab for later
os.makedirs("model_data", exist_ok=True)
with open("model_data/token2idx.pkl", "wb") as fh:
    pickle.dump(token2idx, fh)
with open("model_data/idx2token.pkl", "wb") as fh:
    pickle.dump(idx2token, fh)

print("Vocab size:", len(unique_tokens))

# Convert each file into integer sequences and create training examples
X = []
Y = []

for f in tqdm(files):
    try:
        toks = midi_to_tokens(f)
        idxs = [token2idx[t] for t in toks if t in token2idx]
        # create sliding windows
        for i in range(0, len(idxs) - SEQ_LEN):
            seq_in = idxs[i:i+SEQ_LEN]
            seq_out = idxs[i+SEQ_LEN]
            X.append(seq_in)
            Y.append(seq_out)
    except Exception as e:
        print("error", f, e)

X = np.array(X, dtype=np.int32)
Y = np.array(Y, dtype=np.int32)
print("Training samples:", X.shape[0])

# save training data (you may prefer to use generators for large datasets)
np.save("model_data/X.npy", X)
np.save("model_data/Y.npy", Y)
