"""Perform a two-fold segmentation :
    1] Segment the texts using a classification model and returns the k most
    probable cuts using beam-search.
    2] Select the most relevant cut by measuring the perplexity of a
    transformer on the cut sentence.
"""
from typing import Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sc_segmenter.segmenters.segmenter import Segmenter


class HierarchicalSegmenter(Segmenter):
    """Perform hierarchical segmentation.
    First performs a prediction at the character level, and use
    beam-search to select the $k$ most likely cut.

    Then retrieve the perplexity associated with a cut using a word
    embedding and choose the cut with the lowest perplexity.
    """

    def __init__(self,
                 autoregressive_model_path: str,
                 character_model_path: str,
                 beam_width: int = 5,
                 max_length: int = 128
                 ) -> None:
        """Create an object of the class HierarchicalSegmenter.

        Args:
            embedding_model_path (str): Path to the embedding model and its
                tokenizer.
            chracter_model_path (str): Path to the character model and its
                tokenizer.
        """
        super().__init__()
        self.autoregressive_tokenizer = AutoTokenizer.from_pretrained(
            autoregressive_model_path)
        self.autoregressive_model = AutoModelForCausalLM.from_pretrained(
            autoregressive_model_path)
        self.autoregressive_model.eval()
        self.character_classification_model, \
            self.character_classification_tokenizer = \
            self.load_segmentation_model(
                character_model_path)
        self.beam_width = beam_width
        self.max_length = max_length

    def compute_perplexity(self, sentence: str) -> float:
        """
        Compute perplexity of a sentence using the language model.
        Lower perplexity = more likely sequence according to the model.
        """
        inputs = self.autoregressive_tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.autoregressive_model(
                **inputs,
                labels=inputs["input_ids"])
            # loss = negative log-likelihood per token
            perplexity = torch.exp(outputs.loss).item()
        return perplexity

    def beam_search_decode(self,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor)\
            -> list[dict[str, Any]]:
        """
        Perform beam search to find the top-k most likely sequences.

        Args:
            model: The trained model
            tokenizer: The tokenizer
            input_ids: Input token IDs
            attention_mask: Attention mask
            beam_width: Number of beams to keep at each step
            max_length: Maximum sequence length

        Returns:
            List of dictionaries containing sequences, scores, and
            probabilities
        """
        model = self.character_classification_model.eval()

        with torch.no_grad():
            # Get model outputs
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, seq_len, num_labels)

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Remove batch dimension for simplicity
            logits = logits[0]  # Shape: (seq_len, num_labels)
            probabilities = probabilities[0]  # Shape: (seq_len, num_labels)
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]

            # Get sequence length (excluding special tokens)
            seq_len = torch.sum(attention_mask).item()
            # Initialize beams: each beam is (score, sequence, completed)
            # Start with empty sequence and score 0
            beams = [(0.0, [], False)]  # (log_prob, sequence, completed)

            # Iterate through each position in the sequence
            # Start after [CLS]
            for pos in range(1, min(int(seq_len + 1),
                                    self.max_length - 1)):
                if pos >= probabilities.shape[0]:
                    break

                new_beams = []

                # Get probabilities for current position
                pos_probs = probabilities[pos]
                num_labels = pos_probs.shape[0]
                k = min(self.beam_width,
                        num_labels)
                top_k_probs, top_k_indices = torch.topk(pos_probs, k)

                for beam_score, beam_sequence, beam_completed in beams:
                    if beam_completed:
                        # If beam is completed, carry it forward
                        new_beams.append((beam_score, beam_sequence, True))
                    else:
                        # Expand current beam with top-k possibilities
                        for prob, label_idx in zip(top_k_probs, top_k_indices):
                            new_sequence = beam_sequence + [label_idx.item()]
                            new_score = beam_score + torch.log(prob).item()
                            new_beams.append((new_score, new_sequence, False))

                # Keep only the top-k beams
                # Higher scores first
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:self.beam_width]

            # Convert to final results
            results = []
            for score, sequence, completed in beams:
                # Convert sequence of label IDs to actual labels
                labels = [model.config.id2label[label_id]
                          for label_id in sequence]
                # Convert log probability back to probability
                probability = np.exp(score)

                results.append({
                    'sequence': sequence,
                    'labels': labels,
                    'score': score,
                    'probability': probability
                })

            return results

    def beam_search_predict(self,
                            sentence: str) -> list[dict[str, Any]]:
        """Predict k possible label sequences using beam search"""
        tokenizer = self.character_classification_tokenizer
        # Tokenize
        inputs = tokenizer(sentence,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=self.max_length)

        # Perform beam search
        beam_results = self.beam_search_decode(
            inputs["input_ids"],
            inputs["attention_mask"],
        )

        return beam_results

    def select_best_beam(self,
                         text: str):
        """Select the best beam in terms of perplexity of
        the considered model."""
        beam_results = self.beam_search_predict(text)
        scores = {}
        for i, result in enumerate(beam_results):
            reconstructed = self.reconstruct_text(text,
                                                  result['labels'])

            scores[reconstructed] = self.compute_perplexity(
                reconstructed)
        return min(scores, key=scores.get)

    def segment(self, text: str):  # -> tuple[list[dict[str, Any]], list[Any]]:
        """Perform segmentation of an input text.
        """
        return self.select_best_beam(text=text).split()
