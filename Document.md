# Document for lituration summarries:

Paper1: Transforming the Embeddings: A Lightweight Technique for Speech Emotion Recognition Tasks

Objective: PTM initially trained for speaker recognition can be more effective for SER as knowledge gained for speaker recognition such as learning tone, pitch, etc. from speech can be beneficial.

Summary of section 2 and 3:
    1) Comparision among five model embedding x-vector, ECAPA, wav2vec 2.0, wavLM, and Unispeech-SAT. x-vector.
    2) sampling rate of 16 kHz.
    3) The final hidden states are retrieved from wavLM, UniSpeech-SAT, and wav2vec 2.0 and transformed to a vector of 768-dimension for each audio file to be utilized as input features for the downstream classifier using pooling av- erage.

    4) Among SSL based methods wavLM and UniSpeech-SAT, works better then wav2vec 2.0.

    5) x-vector and ECAPA are supervised. and ECAPA(Emphasized Channel Attention, Propagation, and Aggregation) is modified x-vector.

    6) They used Speech processing Universal PERformance Benchmark (SUPERB) during consideration of speech SSL PTMs (wav2vec 2.0, wavLM, Unispeech-SAT). SUPERB evaluates features from SSL PTMs across a wide range of tasks, such as speaker identification, SER, speech recognition, voice separation, and so on.