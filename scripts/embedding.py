import os
import numpy as np
import torch
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForPreTraining,
    AutoProcessor,
    UniSpeechSatModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
    WhisperModel,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Wav2Vec2
'''
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wavtwovectwo = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

def extract_w2v2(path):
    # import pdb;pdb.set_trace()
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    array = np.array(array)
    array = np.mean(array, axis=0)
    input = processor(array.squeeze(), sampling_rate=sample_rate, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = wavtwovectwo(**input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states

# WavLM
processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
wavlm.to(device)

def extract_wavlm(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    array = np.array(array)
    array = np.mean(array, axis=0)
    input = processor(array.squeeze(), sampling_rate=sample_rate, return_tensors="pt").to(device)

    input = input.to(device)
    with torch.no_grad():
        outputs = wavlm(**input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states

# unispeech-SAT
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/unispeech-sat-base")
unispeechsat = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base")
unispeechsat.to(device)

def extract_unispeech(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    array = np.array(array)
    array = np.mean(array, axis=0)
    input = processor(array.squeeze(), sampling_rate=sample_rate, return_tensors="pt")
    input = input.to(device)

    with torch.no_grad():
        outputs = unispeechsat(**input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states

# Whisper-Base
model = WhisperModel.from_pretrained("openai/whisper-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
model.to(device)

def extract_features_whisper(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    input = feature_extractor(array.squeeze(), sampling_rate=sample_rate, return_tensors="pt")
    input = input.to(device)
    input = input.input_features

    with torch.no_grad():
        outputs = model.encoder(input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states
'''
# MMS
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b")
model = Wav2Vec2Model.from_pretrained("facebook/mms-1b")
model.to(device)

def extract_features_mms(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    input = processor(array.squeeze(), sampling_rate=sample_rate, return_tensors="pt")
    input = input.to(device)

    with torch.no_grad():
        outputs = model(**input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states

# Specify the input and output folder paths
input_folder = 'ASVP-ESD-Update/Audio'
output_folder = 'embedding/mms/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of subdirectories in the input folder
sub_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

# Process each sub-folder
for folder in sub_folders:
    # Construct the full path for the sub-folder
    sub_folder_path = os.path.join(input_folder, folder)

    # Get a list of .wav files in the sub-folder
    wav_files = [file for file in os.listdir(sub_folder_path) if file.endswith('.wav')]

    # Process each .wav file
    for wav_file in wav_files:
        # Extract the modality and vocal channel from the filename
        modality = wav_file[0:2]
        vocal_channel = wav_file[3:5]

        # Check if the file is meant for non-speech (modality = 03 and vocal channel = 02)
        if modality == '03' and vocal_channel == '02':
            # Construct the full file paths
            input_path = os.path.join(sub_folder_path, wav_file)
            output_path = os.path.join(output_folder, f"{os.path.splitext(wav_file)[0]}.npy")
            # Pass the .wav file through the embedding extractor function
            embedding = extract_features_mms(input_path)

            # Save the embedding as a .npy file
            np.save(output_path, embedding)

print("Conversion completed successfully.")