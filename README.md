# Low-Latency Semantic Cache & Prompt-Hash Reverse Proxy for Local LLMs

llm-semantic-cache is a lightweight reverse proxy designed to sit in front of local inference engines (such as `llama.cpp`). It implements a two-tier caching hierarchy combining deterministic SHA-256 prompt hashing with sub-5ms Approximate Nearest Neighbor (ANN) vector search.

By short-circuiting redundant queries before they reach the language model, it slashes a significant amount of latency on consumer hardware while eliminating unnecessary CPU/memory bandwidth thrashing.
