# Paper Summaries

## Paper 1: Transforming the Embeddings: A Lightweight Technique for Speech Emotion Recognition Tasks

### Objective
This paper aims to investigate the efficacy of repurposing Pre-trained Models (PTMs) originally designed for speaker recognition in Speech Emotion Recognition (SER) tasks. It posits that the insights gained from speaker recognition, such as nuances in tone and pitch, could enhance SER performance.

### Summary of Sections 2 and 3
1. Comparative analysis of five model embeddings: x-vector, ECAPA, wav2vec 2.0, wavLM, and Unispeech-SAT.
2. All models trained with a sampling rate of 16 kHz.
3. Final hidden states extracted from wavLM, UniSpeech-SAT, and wav2vec 2.0, transformed into 768-dimensional vectors per audio file, serving as input features for downstream classifiers using average pooling.
4. Among SSL-based methods, wavLM and UniSpeech-SAT outperform wav2vec 2.0.
5. x-vector and ECAPA are supervised models, with ECAPA being a modified version of x-vector.
6. Speech processing Universal PERformance Benchmark (SUPERB) employed to evaluate SSL PTM features across various tasks including speaker identification, SER, speech recognition, and voice separation.

### Additional Notes
- SUPERB offers a comprehensive evaluation of SSL PTM features across diverse tasks.

#### Models:
- **x-vector**: A DNN generating fixed-dimensional embeddings from speech segments.
- **ECAPA**: A robust speaker identification method combining TDNNs with attention mechanisms.
- **Wav2Vec 2.0**: SSL model for speech recognition utilizing CNNs and Transformer architecture.
- **WavLM**: A waveform-based language model operating directly on raw audio for tasks like speech recognition, using CNNs and RNNs.
- **UniSpeech-SAT**: A method for speech synthesis leveraging elements of Tacotron and Transformer architectures.

## Paper 2: Heterogeneity over Homogeneity: Investigating Multilingual Speech Pre-Trained Models for Detecting Audio Deepfake

This paper compares eight PTMs:

1) Multilingual PTMs: XLS-R, Whisper, and Massively Multilingual Speech
2) Monolingual PTMs: WavLM, Unispeech-SAT, Wav2vec2 (based on SUPERB)
3) Speaker recognition PTM: x-vector; Emotion recognition PTM: XLSR_emo.

### Objective
1) Multilingual PTMs excel in Audio Deepfake Detection (ADD) tasks.
2) Proposing a framework, Merge into One (MiO), to combine representations from different PTMs, inspired by the complementarity observed in other speech processing tasks.

### Equal Error Rate (EER)
EER represents the point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).

### Modeling Approaches
1(a): Fully Connected Network (FCN)
1(b): Convolutional Neural Network (CNN)
![OpenAI Logo](fig1a_and1b.png "OpenAI Logo")

### MiO:
MiO follows a consistent modeling pattern for each incoming representation, applying linear projection to a 120-dimensional space followed by bilinear pooling (BP) to facilitate effective feature interaction. BP involves the outer product of two vectors p and q of dimension (D,1), resulting in a matrix of dimension (D, D), represented as:
BPD,
D = pD,1 ⊗ qD,1 = pqT
![OpenAI Logo](fig2.png "OpenAI Logo")
In MiO, multiple PTMs are integrated, with performance explanations provided in the appendix.

Appendix:

1) Dataset: 4 dataset, ASV, ITW and DC-C, DC-E

2) Detailed Information of the Pre-Trained Models: XLS-R(1 billion parameters), Whisper(74 million), MMS(1 billion)[PTM's multiligual.] Unispeech-SAT(94 Million), WavLM(base(94 Million) and large(316 Million)), Wav2vec2(95 Million), X-vector(supervised)(4.2 Million), XLSR-emo.

3) Cross-Corpus Evaluation:

First Experiment(PTMs): Cross corpus experiment with 120 and 240 dimentions with PCA.[best is XLS-R]

Second Experiment(MIO): Cross corpus experiment with 120 and 240 dimentions with PCA.[only two combinations(1.XLS-R + x-vector) (2.Whisper + Unispeech-SAT)]

