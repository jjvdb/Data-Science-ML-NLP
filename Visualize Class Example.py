# This file shows an example of how we would use the Visualize class

# create a new Visualize object to treat and visualize our data set
visualizer = Visualize()

# generate a sparse matrix of dimensions 100x100 to use in the dimensionality reduction function, Reduce_Dim_SVD
from sklearn.random_projection import sparse_random_matrix
X = sparse_random_matrix(100, 100, density = 0.01, random_state = 42)

# reduce the sparse matrix X to 20 dimensions using SVD, and further reduce it to 2 dimensions using T-SNE - output both 
# of these reduced-dimensionality arrays
test = visualizer.Reduce_Dim_SVD(X, 20, 2)

# grab the 2D, T-SNE reduced-dimensionality array
graph = test[1]

# plot the 2D, T-SNE reduced-dimensionality array using the plot_2D function
visualizer.plot_2D(graph)
