# Implementation and experiements of skeletonized graph construction and skeletonized label propagation

## Folder structure
|-Articles : A couple of the articles and tech reports read

|-Data : Folder for the different experiements and models created

|- GSSL: Code folder for different stages of SGSSL

|-- DistMatrix : Functions for creating similarity matrix, and for saving and loading.

|-- Graph : Functions for creating graph, and propagating through it. Contains code for creating embedded and unembedded graphs.

|-- MNIST : Short function for extracting MNIST numbers from zipfile. Depricated in favor of ```pymde.datasets.MNIST```

|-- Skeletonized : Functions for converting from matrix to pygel and vice versa, and for remapping skeleton nodes.

|- Rest of folders : Should be consideret misc. 

## Short introduction
The implementation in this repo merges skeletonization from graph-theory and graph-based semi-supervised learning methods, to experiment and see what the results would be.
