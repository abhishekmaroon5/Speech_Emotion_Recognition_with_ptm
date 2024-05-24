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
