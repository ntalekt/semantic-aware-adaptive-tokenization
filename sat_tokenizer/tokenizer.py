import torch
import re
from typing import List
from transformers import AutoTokenizer, AutoModel
from sentencepiece import SentencePieceProcessor
import jellyfish
from sklearn.metrics.pairwise import cosine_similarity


class SATokenizer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
        sp_model_path: str = None,
        semantic_threshold: float = 0.82,
        merge_threshold: float = 0.75,
        context_window: int = 3,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.sp_processor = None
        if sp_model_path:
            self.sp_processor = SentencePieceProcessor(model_file=sp_model_path)
        self.soundex = jellyfish.soundex
        self.semantic_threshold = semantic_threshold
        self.merge_threshold = merge_threshold
        self.context_window = context_window
        self.special_char_pattern = re.compile(r"[\W_]+", re.UNICODE)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def _semantic_segmentation(self, text: str) -> List[str]:
        words = [w for w in re.findall(r"\w+(?:-\w+)*|\S", text) if w.strip()]

        if len(words) < 2:
            return words

        embeddings = self._get_embeddings(words)
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:])

        units = []
        current_unit = [words[0]]

        for i, sim in enumerate(similarities.diagonal()):
            if sim > self.semantic_threshold:
                current_unit.append(words[i + 1])
            else:
                units.append("".join(current_unit).strip())
                current_unit = [words[i + 1]]

        units.append("".join(current_unit).strip())
        return [u for u in units if u]

    def _context_aware_merge(self, units: List[str]) -> List[str]:
        if len(units) < 2:
            return units

        merged = []
        i = 0
        while i < len(units):
            best_score = -1
            best_length = 1
            current_context = units[i : i + self.context_window]

            for j in range(1, len(current_context) + 1):
                candidate = " ".join(current_context[:j])
                candidate_emb = self._get_embeddings([candidate])
                original_embs = self._get_embeddings(current_context[:j])
                score = cosine_similarity(
                    candidate_emb, original_embs.mean(axis=0, keepdims=True)
                )[0][0]

                if score > best_score and score > self.merge_threshold:
                    best_score = score
                    best_length = j

            merged.append(" ".join(current_context[:best_length]))
            i += best_length

        return merged

    def _handle_rare_words(self, token: str) -> List[str]:
        if self.special_char_pattern.sub("", token) == "":
            return [token]

        if token.lower() in self.tokenizer.vocab:
            return [token]

        if self.sp_processor:
            bpe_tokens = self.sp_processor.encode_as_pieces(token)
            if len(bpe_tokens) > 1:
                return bpe_tokens

        return [jellyfish.soundex(token)]

    def tokenize(self, text: str, granularity: str = "auto") -> List[str]:
        units = self._semantic_segmentation(text)
        merged_units = self._context_aware_merge(units)

        final_tokens = []
        for unit in merged_units:
            if granularity == "fine" or (
                granularity == "auto" and len(unit.split()) > 1
            ):
                sub_tokens = self._handle_rare_words(unit)
                final_tokens.extend(sub_tokens)
            else:
                final_tokens.append(unit)

        return final_tokens
