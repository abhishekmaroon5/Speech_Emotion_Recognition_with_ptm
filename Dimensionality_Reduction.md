
## Essence of Dimensionality Reduction:
It’s not feasible to analyze each and every dimension at a microscopic level in high-dimensional data. It might take us days or months to perform any meaningful analysis which requires a lot of time, money, and manpower in our business, which is not often encouraged. Training data with high dimensions will lead to problems like:

- Space required to store the data gets increased with increasing dimensions.
- Less dimensions will take low time complexity in training a model.
- As dimensions increase, the possibility of overfitting the model also increases.
- We cannot visualize high-dimensional data. By dimensionality reduction, we will reduce the data to 2D or 3D for better visualization.
- It will remove all the correlated features in our data.

### Components of Dimensionality Reduction:
There are two major components of dimensionality reduction which will be discussed in detail here.

#### I) Feature Selection:

Feature selection involves finding a subset of original data so that there will be a minimum loss of information. It has the following three strategies:

- Filter Strategy: Strategy to gain more information on the data.
- Wrapper Strategy: Basing on the model accuracy we will select features.
- Embedded Strategy: Basing on model prediction errors, we will take a decision whether to keep or remove the selected features.

#### II) Feature Projection:

Feature Projection, also known as Feature Extraction, is used to transform the data in high-dimensional space to low-dimensional space. The data transformation can be done in both linear and non-linear ways.

- For linear transformation, we have Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA).

### PCA (Done)
PCA is mostly used as a tool in exploratory data analysis (EDA) and for making predictive models. PCA can be done by eigenvalue decomposition of a data covariance (or correlation) matrix or singular value decomposition of a data matrix.

#### Advantages:
- It removes correlated features.
- Improves model efficiency.
- Reduces overfitting.
- Improves Visualization.

#### Disadvantages:
- PCA is a linear algorithm and it won’t work very well for polynomial or other complex functions. We can somehow use kernel PCA for such data.
- After PCA, we may lose a lot of information if we won’t choose the right number of dimensions to get eliminated.
- Less interpretability


## LDA
PCA tries to find the components that maximizes the variance, while on the other hand LDA tries to find the new axes that:

1) Maximizes the separability of the categories and
![OpenAI Logo](images/lda_1.png "OpenAI Logo")
2)  Minimizes the variance among categories.
![OpenAI Logo](images/lda_2.png "OpenAI Logo")

By minimizing the variance, we can well separate the clusters of individual groups. Hence it is as important as maximizing the mean values of groups.

![OpenAI Logo](images/lda_3.png "OpenAI Logo")

Comparision between PCA and LDA:

To know the difference between the working of PCA and LDA, let’s look at the following plot. Where PCA tries to maximizes the variance unlike LDA which tries to maximizes the separability of three categories.

![OpenAI Logo](images/lda_vs_pca.png "OpenAI Logo")

We can see the difference between the both plots. In PCA, their is some overlapping in the data and it is difficult to find a line separating the two groups. LDA can help us to separate the three groups since their is less overlapping in the data.

## SVD:

Singular Value Decomposition is a matrix factorization technique widely used in various applications, including linear algebra, signal processing, and machine learning. It decomposes a matrix into three other matrices, allowing for the representation of the original matrix in a reduced form.

🔍 Title: "Singular Value Decomposition (SVD): Overview"

Singular Value Decomposition (SVD) is a 🚀 powerful tool in numerical linear algebra for data processing, particularly for data reduction and dimensionality reduction.
It is likened to a 🎨 data-driven generalization of the Fourier transform, allowing the creation of tailored coordinate systems or transformations based on specific data.
SVD is utilized in various applications, including solving linear systems of equations, principal component analysis (PCA), and building models for linear regression.
Widely used across industries, SVD is integral to algorithms such as Google's PageRank, facial recognition systems, and recommender systems like those used by Amazon and Netflix.
SVD is prized for its simplicity, interpretability, scalability, and applicability to diverse data sets, making it a valuable tool for leveraging linear algebra in practical contexts. 🛠️📊🌐

![OpenAI Logo](images/svd.png "OpenAI Logo")

Comparision:
Conclusion
The choice between Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Singular Value Decomposition (SVD) depends on the specific objectives and characteristics of the data. Here are general guidelines on when to use each technique:






1. PCA (Principal Component Analysis)

Use Cases:
1. When the goal is to reduce the dimensionality of the dataset.
2. In scenarios where capturing global patterns and relationships within the data is crucial.
3. For exploratory data analysis and visualization.

2. LDA (Linear Discriminant Analysis)

Use Cases:
1. In classification problems where enhancing the separation between classes is important.
2. When there is a labeled dataset, and the goal is to find a projection that maximizes class discrimination.
3. LDA is particularly effective when the assumption of normally distributed classes and equal covariance matrices holds.

3. SVD (Singular Value Decomposition)

Use Cases:
1. When dealing with sparse data or missing values.
2. In collaborative filtering for recommendation systems.
3. SVD is also applicable in data compression and denoising.


## KPCA:

Kernel PCA (KPCA)
- What is it?
Kernel PCA is an extension of PCA that allows for nonlinear dimensionality reduction by using kernel functions.
- How does it work?
It first maps the input data into a higher-dimensional space using a kernel function (like Gaussian, polynomial, etc.).
- In this higher-dimensional space, it then performs PCA.
This allows KPCA to find nonlinear relationships in the data that PCA would miss.
- Benefits:
Can capture complex, nonlinear relationships in the data.
Useful when linear methods like PCA may not be sufficient.


Autoencoders (Single-hidden layer)
- **What are they?**

- Autoencoders are a type of neural network used for unsupervised learning, particularly for data compression and dimensionality reduction.
- How do they work?

- An autoencoder consists of an encoder and a decoder.
The encoder compresses the input data into a lower-dimensional representation (the bottleneck layer).
The decoder then tries to reconstruct the original input from this compressed representation.
The model is trained to minimize the reconstruction error, which forces it to learn a compact representation of the data.
- Single-hidden layer autoencoder:
In this version, there's a single hidden layer between the encoder and decoder.
The number of neurons in this hidden layer determines the dimensionality of the compressed representation.
Training involves adjusting the weights to minimize the difference between input and output.
- Benefits:
Can learn complex patterns in data.
Allows for nonlinear dimensionality reduction.
Can handle high-dimensional data effectively.


Comparison:
- PCA is a linear method, while KPCA and Autoencoders (with nonlinear activation functions) can capture nonlinear relationships.
- PCA is simple and interpretable but might not capture complex patterns.
- KPCA extends PCA to capture nonlinear relationships using kernels.
- Autoencoders are powerful for learning representations and can be used for nonlinear dimensionality reduction.

## Gaussian Random Projection (GRP)
What is it?

- Gaussian Random Projection (GRP) is a technique for dimensionality reduction that uses random matrices with elements drawn from a Gaussian distribution.
- It's based on the Johnson-Lindenstrauss lemma, which states that a set of high-dimensional data points can be projected into a lower-dimensional space while approximately preserving pairwise distances between the points.

How does it work?
- GRP generates a random Gaussian matrix.
Each element of this matrix is sampled independently from a Gaussian distribution.
- The data is then multiplied by this random matrix to project it onto a lower-dimensional space.
Despite being random, the projection is designed such that it approximately preserves the distances between the data points.

Benefits:
- Efficiency: It can be computationally faster than methods like PCA for very high-dimensional data.
Approximate preservation of distances: While not exact, it aims to preserve the pairwise distances between points.
- Simple: It's relatively straightforward to implement.
Limitations:
- Lack of interpretability: Unlike PCA, the new dimensions may not have clear interpretations.
Loss of information: Similar to other dimensionality reduction techniques, there is a loss of information when projecting to a lower-dimensional space.

- Comparison with PCA:
PCA is based on finding the principal components that maximize variance, while GRP uses random projections.
- PCA has an interpretation in terms of the original features, while GRP's new dimensions might not have such clear interpretations.
- ¯GRP can be faster for very high-dimensional data, but PCA might be more accurate in preserving variance.

## Truncated SVD
🔍 **Eckart-Young Truncated Singular Value Decomposition (eYSVD)**

Eckart-Young truncated singular value decomposition (eYSVD) is a truncated singular value decomposition (TSVD) algorithm that is used to approximate a matrix A by a product of a matrix U, a diagonal matrix Σ, and a matrix V, where U and V are orthogonal matrices. The algorithm is used to reduce the dimensionality of a matrix while preserving its important features. The truncation is done by selecting the k largest singular values and keeping only the corresponding columns of U and rows of V. The eYSVD algorithm is particularly useful when the matrix A is sparse or has a large number of singular values.

💡 **Key Steps:**

1. Compute the singular value decomposition A = U Σ V^T, where U and V are orthogonal matrices and Σ is a diagonal matrix containing the singular values of A.

2. Truncate the singular values and corresponding columns of U and rows of V to retain only the k largest singular values.

3. Form the truncated matrices U_k and V_k using the selected columns of U and rows of V.

4. Compute the low-rank approximation of A as A_k = U_k Σ_k V_k^T, where Σ_k contains only the retained singular values on its diagonal.

By selecting an appropriate value of k, eYSVD allows users to balance the trade-off between approximation accuracy and computational efficiency. This technique is particularly beneficial in scenarios where memory or computational resources are limited, or when dealing with large-scale datasets.

