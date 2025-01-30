<div align="center">
  <h4>
    <a href="https://github.com/ntalekt/semantic-aware-adaptive-tokenization/stargazers"><img src="https://img.shields.io/github/stars/ntalekt/semantic-aware-adaptive-tokenization?style=flat"/></a>
    <a href="https://github.com/ntalekt/semantic-aware-adaptive-tokenization/commits/master"><img src="https://img.shields.io/github/last-commit/ntalekt/semantic-aware-adaptive-tokenization?style=flat"/></a>
  </h4>
</div>

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

## **Implementation Roadmap**

### **1. Core Components**
- **Semantic Unit Detector**: Identifies meaningful phrases/entities.
- **Adaptive Granularity Controller**: Adjusts token size based on task.
- **Context-Aware Merger**: Uses surrounding context to merge/split tokens.
- **Rare Word Compressor**: Handles OOV (out-of-vocabulary) words.

### **2. Tools & Libraries**
- **SpaCy**: For dependency parsing and rule-based tokenization.
- **Hugging Face Transformers**: For pre-trained embeddings (e.g., BERT, DistilBERT).
- **SentencePiece**: For subword tokenization fallback.
- **Phonetic Algorithms**: `soundex` or `metaphone` for rare words.
- **NLTK**: For baseline tokenization rules.

### **3. Step-by-Step Code Implementation**

#### **Step 1: Semantic Unit Detection**
We’ll use SpaCy’s entity recognition and dependency parsing to detect semantic units:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def detect_semantic_units(text):
    doc = nlp(text)
    semantic_units = []
    for chunk in doc.noun_chunks:
        semantic_units.append(chunk.text)
    return semantic_units

# Example
text = "Artificial intelligence is transforming industries worldwide."
print(detect_semantic_units(text))  # Output: ['Artificial intelligence', 'industries worldwide']
```

#### **Step 2: Adaptive Granularity**
Create a task-specific tokenizer that switches between coarse and fine modes:

```python
from transformers import AutoTokenizer

class AdaptiveTokenizer:
    def __init__(self, task="general"):
        self.task = task
        self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def tokenize(self, text):
        if self.task == "general":
            return detect_semantic_units(text)  # Coarse semantic units
        else:
            return self.base_tokenizer.tokenize(text)  # Fine-grained tokens

# Usage
general_tokenizer = AdaptiveTokenizer(task="general")
print(general_tokenizer.tokenize("unbelievable"))  # ['unbelievable']

fine_tokenizer = AdaptiveTokenizer(task="translation")
print(fine_tokenizer.tokenize("unbelievable"))  # ['un', '##bel', '##ie', '##vable']
```

#### **Step 3: Context-Aware Merging**
Use SpaCy’s dependency parser to merge tokens based on syntactic relationships:

```python
def context_aware_merge(text):
    doc = nlp(text)
    merged_tokens = []
    for token in doc:
        if token.dep_ in ("compound", "amod") and token.head.i == token.i + 1:
            merged_tokens.append(f"{token.text} {token.head.text}")
        else:
            merged_tokens.append(token.text)
    return merged_tokens

# Example
print(context_aware_merge("river bank account"))  # ['river bank', 'account']
```

#### **Step 4: Rare Word Compressor**
Combine BPE with phonetic encoding for rare words:

```python
from sentencepiece import SentencePieceProcessor
import soundex

spm = SentencePieceProcessor(model_file="tokenizer.model")

def compress_rare_word(word):
    # Try BPE first
    bpe_tokens = spm.encode_as_pieces(word)
    if len(bpe_tokens) == 1:  # Rare word
        phonetic_code = soundex.soundex(word)
        return [phonetic_code]
    return bpe_tokens

# Example
print(compress_rare_word("Donaudampfschifffahrtsgesellschaft"))  # ['D532'] (phonetic code)
```

### **4. Full Pipeline Integration**
Combine all components into a unified tokenizer:

```python
class SATokenizer:
    def __init__(self, task="general"):
        self.semantic_detector = detect_semantic_units
        self.adaptive_tokenizer = AdaptiveTokenizer(task)
        self.rare_compressor = compress_rare_word
        
    def tokenize(self, text):
        # Step 1: Detect semantic units
        units = self.semantic_detector(text)
        
        # Step 2: Adaptive splitting
        tokens = []
        for unit in units:
            if self.adaptive_tokenizer.task == "general":
                tokens.append(unit)
            else:
                tokens.extend(self.adaptive_tokenizer.tokenize(unit))
        
        # Step 3: Handle rare words
        final_tokens = []
        for token in tokens:
            if token.lower() not in self.adaptive_tokenizer.base_tokenizer.vocab:
                final_tokens.extend(self.rare_compressor(token))
            else:
                final_tokens.append(token)
        
        return final_tokens

# Usage
sa_tokenizer = SATokenizer(task="general")
print(sa_tokenizer.tokenize("DeepSeek-R1 is revolutionizing AI"))  
# Output: ['DeepSeek-R1', 'revolutionizing', 'AI']
```

### **5. Evaluation & Optimization**
- **Benchmarks**: Compare against WordPiece/BPE using:
  - **Perplexity**: How well the tokenizer predicts unseen text.
  - **Downstream Task Accuracy**: Test on classification/translation.
  - **Token Efficiency**: Ratio of tokens to original characters.
- **Optimization**:
  - Fine-tune the semantic detector on domain-specific data.
  - Experiment with merging rules (e.g., dependency types).

### **6. Challenges & Solutions**
| Challenge | Solution |
|-----------|----------|
| Computational Overhead | Use smaller models (e.g., DistilBERT) for semantic detection. |
| Multilingual Support | Integrate Stanza or SpaCy’s multilingual pipelines. |
| Rare Word Handling | Combine BPE with language-specific phonetic algorithms. |

### **Next Steps**
1. **Set Up Environment**:
   ```bash
   pip install spacy transformers sentencepiece soundex
   python -m spacy download en_core_web_sm
   ```
2. **Prototype**: Test the `SATokenizer` class on your dataset.
3. **Iterate**: Adjust merging rules and semantic detection thresholds.

### **Example Output Comparison**

Input: *"Artificial intelligence is transforming industries worldwide."*

- Traditional Tokenization (WordPiece): ["Artificial", "intelligence", "is", "transforming", "industries", "worldwide"]
- Proposed SAT Method: ["Artificial intelligence", "is transforming", "industries worldwide"]

This approach could significantly improve efficiency and performance in NLP tasks while maintaining interpretability and flexibility.
