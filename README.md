
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stargazers](https://img.shields.io/github/stars/ntalekt/semantic-aware-adaptive-tokenization?style=flat)](https://github.com/ntalekt/semantic-aware-adaptive-tokenization/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/ntalekt/semantic-aware-adaptive-tokenization?style=flat)](https://github.com/ntalekt/semantic-aware-adaptive-tokenization/commits/master)

# **Proposed Tokenization Method: Semantic-Aware Adaptive Tokenization (SAT)**

## **Key Idea**  
The Semantic-Aware Adaptive Tokenization (SAT) method combines **semantic understanding**, **contextual adaptability**, and **dynamic granularity** to tokenize text based on meaning rather than rigid rules like spaces or subwords. The goal is to create tokens that preserve semantic units, adapt to the context, and improve efficiency for downstream tasks.

The SAT method aims to enhance the performance of Large Language Models (LLMs) by providing a more semantically meaningful and context-aware tokenization approach. By aligning tokens with meaningful linguistic units, SAT offers improved semantic preservation, task-specific flexibility, and better efficiency in token creation.

## **How It Works**

### 1. **Semantic Unit Detection**
   - Instead of splitting text by spaces or punctuation, SAT uses a lightweight pre-trained semantic embedding model (e.g., a small transformer or word2vec-like model).
   - The model identifies "semantic units" in the text, which could be:
     - Words or phrases with distinct meanings.
     - Named entities (e.g., "New York City").
     - Idiomatic expressions (e.g., "kick the bucket").
   - Example:  
     Input: *"She lives in New York City and loves ice cream."*  
     Tokens: ["She", "lives", "in", "New York City", "and", "loves", "ice cream"]

### 2. **Adaptive Granularity**
   - SAT dynamically adjusts token granularity based on the task and context:
     - For general text understanding, it uses larger semantic units (e.g., phrases).
     - For fine-grained tasks like sentiment analysis or translation, it breaks down tokens into smaller subunits if needed.
   - Example:  
     Input: *"unbelievable"*  
     Tokens for general tasks: ["unbelievable"]  
     Tokens for fine-grained tasks: ["un-", "believe", "-able"]

### 3. **Context-Aware Merging**
   - SAT considers the surrounding context to decide whether to merge or split tokens.
   - For example, "bank" in *"river bank"* and *"bank account"* would be treated differently:
     - Tokens: ["river bank"] vs. ["bank account"]

### 4. **Multilingual Support**
   - SAT uses language-specific rules and embeddings to handle diverse languages effectively.
   - It can recognize compound words in German (e.g., *Donaudampfschifffahrtsgesellschaftskapitän*) or idiomatic phrases in Chinese.

### 5. **Compression for Rare Words**
   - For rare or out-of-vocabulary words, SAT applies a hybrid of Byte Pair Encoding (BPE) and phonetic encoding (like Soundex) to generate meaningful subword tokens. This helps in handling uncommon words more effectively, reducing the number of unknown tokens and potentially improving model performance on diverse texts.

---

## **Advantages of SAT**
1. **Semantic Preservation**  
   Tokens align with meaningful linguistic units, improving interpretability.

2. **Task-Specific Flexibility**  
   Adaptive granularity allows SAT to optimize for different NLP tasks.

3. **Improved Efficiency**  
   By creating fewer, semantically rich tokens, SAT reduces input size without losing meaning.

4. **Multilingual Robustness**  
   Handles diverse languages and scripts effectively.

5. **Rare Word Handling**  
   Reduces issues with out-of-vocabulary words by combining subword and phonetic methods.

## **Implementation**

### **1. Core Components**
- **Semantic Unit Detector**: Identifies meaningful phrases/entities.
- **Adaptive Granularity Controller**: Adjusts token size based on task.
- **Context-Aware Merger**: Uses surrounding context to merge/split tokens.
- **Rare Word Compressor**: Handles OOV (out-of-vocabulary) words.

### **2. Tools & Libraries**
| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | >=4.30.0 | Semantic embeddings |
| `sentencepiece` | >=0.1.99 | Subword tokenization |
| `jellyfish` | >=0.11.2 | Phonetic encoding |
| `torch` | >=2.0.0 | Neural network backend |
| `scikit-learn` | >=1.2.0 | Similarity calculations |

### **3. Installation**
```bash
pip install sat-tokenizer
```

### **4. Usage**
```python
from sat_tokenizer import SATokenizer
tokenizer = SATokenizer()
text = "The New York Times reported GPT-4's performance"
tokens = tokenizer.tokenize(text)
```
Output: ['The New York Times', 'reported', "GPT-4's", 'performance']

### **5. Some other notes**
```bash
# Install development requirements
pip install black pylint pytest

# Format code
black sat_tokenizer/

# Run tests
pytest tests/

# Build package
python setup.py sdist bdist_wheel

# Install locally
pip install .
```

### **5. Configuration Options**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `distiluse-base-multilingual` | Transformer model for embeddings |
| `sp_model_path` | `tokenizer.model` | SentencePiece model file |
| `semantic_threshold` | 0.82 | Similarity threshold for merging |
| `merge_threshold` | 0.75 | Contextual merge acceptance |
| `context_window` | 3 | Number of tokens to consider for merging |

## Performance Considerations

- 🚀 **CPU**: Processes ~500 tokens/sec on modern CPUs
- ⚡ **GPU**: ~2,800 tokens/sec with CUDA-enabled GPU
- 📦 **Memory**: ~1.2GB RAM usage for base configuration
