import os
import pickle
from music21 import converter, instrument, note, chord

# === Folders ===
base_dir = os.path.dirname(os.path.abspath(__file__))
midi_folder = os.path.join(base_dir, "midi_songs")
data_folder = os.path.join(base_dir, "data")

# Create data folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

notes = []

# === Read all MIDI files safely ===
for file in os.listdir(midi_folder):
    if file.endswith(".mid") or file.endswith(".midi"):  # only MIDI files
        file_path = os.path.join(midi_folder, file)
        try:
            midi = converter.parse(file_path)
            print(f"✅ Parsing {file}...")

            parts = instrument.partitionByInstrument(midi)
            if parts:  # if there are instrument parts
                for part in parts.parts:
                    notes_to_parse = part.recurse()
                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            notes.append('.'.join(str(n) for n in element.normalOrder))
            else:  # if no instruments, parse flat
                notes_to_parse = midi.flat.notes
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))

        except Exception as e:
            print(f"❌ Could not parse {file}: {e}")

# === Save notes to pickle ===
notes_path = os.path.join(data_folder, "notes.pkl")
with open(notes_path, "wb") as f:
    pickle.dump(notes, f)

print(f"✅ Done! Total notes extracted: {len(notes)}")
print(f"Saved notes.pkl to {notes_path}")
