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

---

This README.md provides a brief summary of Paper 1, focusing on the comparison of different model embeddings and their application in Speech Emotion Recognition tasks. For more detailed information, please refer to the original paper.
