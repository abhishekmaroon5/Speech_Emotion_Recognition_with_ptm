# Schedule and Upcoming Todos

## Things Completed:

1. Go through dimensionality reduction methods: (DONE)
   - SVD, PCA, and LDA.
   - GRP, KPCA, and Autoencoders (Single-hidden layer).

2. Create an environment for the pipeline. (DONE)

3. Go through the pipeline and run it end-to-end on GPU. (DONE))

4. Apply dimensionality reduction methods learned above in the SER pipeline:
   - PCA, KPCA, SVD, LDA, GRP.

   **Experiments:**
   - Embedding size of 120.
   - Embedding size of 240.

   *All results are updated in* `results_document.md` *document.*

5. **MLP Experiments:**
   - 240 Embedding size.
   - 120 Embedding size.

   **SVM Experiment:**

6. Experiment with random selection from embedding vector with sizes 384, 240, and 120. (DONE))

 
# Upcoming targets 
Modify the pipeline for SER and use multiple PTMs from Hugging Face. (TODO)

Code walkthrough for paper 2. (TODO)

# New Plan post 24 May: 

1. Work on two new datasets for speech emotion recognization similar to CREMA-D.
2. Work on comparision of diffused redundancy and dimensionality reduction on accuray.
3. Create relavant graphs with embedding vector size varing from 10% to 100% with an interval of 10 and plot the accuracy.
4. Repeat the above experiment three times.

# New Plan post 30 May:

1) Given two dataset from kaggle, using embedding extraction code we have extract the features.
2) Using above features perform downstreaming task.
3) Compare dim-reduction technique and diffused redundance present in the embedding.


Dataset 1:
https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances


Dataset 2:
https://www.kaggle.com/datasets/johaangeorge/vivae-non-speech


# Plan post 4 June:

1) Using scripts for extracting embedding from 6 different PTM's (Xvector, Wav2Vec2, WavLM, Unispeechsat, whisper, MMS)
2) Use above dataset for training, evaluation and testing using above embeddings.
3) Apply dimensionality reduction on above embeddings and compare the result, also apply diffused redundancy and compare.