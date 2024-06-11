#Wav2Vec2
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wavtwovectwo = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)



# audio file is decoded on the fly
def extract_w2v2(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    array  = np.array(array)
    array = np.mean(array, axis = 0)
    input = processor(array.squeeze(), sampling_rate= sample_rate, return_tensors="pt").to(device)
    # apply the model to the input array from wav
    with torch.no_grad():
        outputs = wavtwovectwo(**input)
    # extract last hidden state, compute average, convert to numpy
    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#WavLM

from transformers import AutoProcessor, WavLMModel, Wav2Vec2Processor
import torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")# audio file is decoded on the fly
wavlm.to(device)

def extract_wavlm(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    array  = np.array(array)
    array = np.mean(array, axis = 0)
    input = processor(array.squeeze(), sampling_rate= sample_rate, return_tensors="pt").to(device)
    # apply the model to the input array from wav
    input = input.to(device)
    with torch.no_grad():
       outputs = wavlm(**input)
    # extract last hidden state, compute average, convert to numpy
    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#unispeech-SAT

from transformers import AutoProcessor, UniSpeechSatModel, Wav2Vec2FeatureExtractor
import torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/unispeech-sat-base")
unispeechsat = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base")
unispeechsat.to(device)

def extract_unispeech(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    array = np.array(array)
    array = np.mean(array, axis = 0)
    input = processor(array.squeeze(), sampling_rate=sample_rate, return_tensors="pt")
    input = input.to(device)
    with torch.no_grad():
       outputs = unispeechsat(**input)
    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    #embeddings = torch.nn.functional.normalize(features, dim=-1).cpu()
    return last_hidden_states
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Whisper- Base
from transformers import AutoFeatureExtractor, WhisperModel
from datasets import load_dataset

model = WhisperModel.from_pretrained("openai/whisper-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
model.to(device)
import torchaudio
def extract_whisper(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    input = feature_extractor(array.squeeze(), sampling_rate = sample_rate, return_tensors = 'pt')
    input = input.to(device)
    input = input.input_features
    with torch.no_grad():
        outputs = model.encoder(input)
    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis = 0).to("cpu").numpy()
    return last_hidden_states

#MMS

from transformers import AutoProcessor, AutoModelForPreTraining, Wav2Vec2FeatureExtractor
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b")
model = Wav2Vec2Model.from_pretrained("facebook/mms-1b", load_in_4bit=True, device_map="auto")
model.to(device)


def extract_mms(path):
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    input = processor(array.squeeze(), sampling_rate=sample_rate, return_tensors="pt")
    input = input.to(device)
    with torch.no_grad():
       outputs = model(**input)
    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    #embeddings = torch.nn.functional.normalize(features, dim=-1).cpu()
    return last_hidden_states