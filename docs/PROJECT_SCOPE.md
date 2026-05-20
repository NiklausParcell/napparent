# Project Scope

The current problem in modern machine learning is that it was meant to be predictive, rather than physical. As the world transitions to more of a physical AI paradigm, new machine learning is needed to fit that paradigm. Therefore, what we are building in this scope is a mapping from an image to word-vector-space using an AI transparency algorithm built on knowledge graph embeddings.

Ideally, we want to use nabled -- a Rust crate a partner and I have been working on. nabled is a pure linear algebra library written in Rust. We can use this for the backbone of the algorithm.

## Concepts to use and build upon

- This is a unified multimodal representation
- Knowledge graph representations are built by sparse matrices
- Vectors are similarity matched to vectors in embedding space
- Instead of word to word similarity, it will be "parts of an image" compared to word embedding space
- Later, sensors or any other data type can be similarity compared based on vector mapping to different spaces
- Once similarity mapped and represented from the knowledge graph, the next step is to map to an action space
