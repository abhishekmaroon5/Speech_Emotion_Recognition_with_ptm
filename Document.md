# Paper Summaries

## Paper 1: Transforming the Embeddings: A Lightweight Technique for Speech Emotion Recognition Tasks

### Objective
The objective of this paper is to explore the effectiveness of using Pre-trained Models (PTMs) initially trained for speaker recognition for Speech Emotion Recognition (SER) tasks. It argues that knowledge gained for speaker recognition, such as learning tone, pitch, etc., can be beneficial for SER.

### Summary of Sections 2 and 3
1. Comparison among five model embeddings: x-vector, ECAPA, wav2vec 2.0, wavLM, and Unispeech-SAT.
2. All models are trained with a sampling rate of 16 kHz.
3. The final hidden states are retrieved from wavLM, UniSpeech-SAT, and wav2vec 2.0, transformed into a 768-dimensional vector for each audio file, and utilized as input features for the downstream classifier using pooling average.
4. Among SSL-based methods, wavLM and UniSpeech-SAT perform better than wav2vec 2.0.
5. x-vector and ECAPA are supervised models, with ECAPA being a modified version of x-vector.
6. Speech processing Universal PERformance Benchmark (SUPERB) is used for evaluating features from SSL PTMs (wav2vec 2.0, wavLM, Unispeech-SAT) across various tasks such as speaker identification, SER, speech recognition, voice separation, etc.

### Additional Notes
- SUPERB evaluates features across a wide range of tasks, providing a comprehensive assessment of SSL PTMs.


x-vector: deep neural network (DNN) that processes speech segments to produce fixed-dimensional embeddings known as x-vectors. 

ECAPA: It a robust speaker identification method,  It combines elements of Time-Delay Neural Networks (TDNNs) with attention mechanisms, allowing it to effectively model long-range dependencies in speech data. 

Wav2Vec 2.0: SSL training for speech recognition. It employs a multi-layer convolutional neural network (CNN) for feature extraction followed by a Transformer-based architecture for context aggregation. 

WavLM: WavLM is a waveform-based language model that directly operates on raw audio waveforms for tasks such as speech recognition. Architectures = CNNs and RNNs to model temporal dependencies in the audio signal.

UniSpeech-SAT: Unified Speech Synthesis with Self-Attentive Tacotron is a method for speech synthesis model that combines elements of Tacotron and Transformer architectures. It leverages self-attention mechanisms to capture long-range dependencies in text and generates high-quality speech waveforms.
    Tacotron: Tacotron is an end-to-end generative text-to-speech model that takes a character sequence as input and outputs the corresponding spectrogram.

## paper 2: Heterogeneity over Homogeneity: Investigating Multilingual Speech Pre-Trained Models for Detecting Audio Deepfake

They compared eight PTM's:

1) For multilingual PTMs we choose, XLS-R (Babu et al., 2022), Whisper (Radford et al., 2023), and Massively Multilingual Speech
2) For Monolingual PTMs (WavLM, Unispeech-SAT, Wav2vec2) based on the SUPERB.
3) As speaker recognition PTM, they consider, x-vectorand as emotion recognition PTM, XLSR_emo.

Objective: 
1) Multilingual PTM are the best performers for ADD task.
2) By combining representations from different PTMs as it has been seen in other speech processing tasks such as speech recognition that certain representations act as complementary to each other and we propose a framework, Merge into One (MiO) for the same.