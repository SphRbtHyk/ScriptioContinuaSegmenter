"""Base class for the different segmenters."""
import difflib
import warnings
from typing import Union, Any, Set, Tuple, List

from transformers import AutoModelForTokenClassification, AutoTokenizer


class Segmenter:
    """Base class for all segmenters.
    """

    def segment(self, text: str) -> list[str]:
        """Given a string, segment it into several words.

        Args:
            text (str): The text to segment as a text
                without spaces.

        Returns:
            str: The text returned as a list of word.
        """
        raise NotImplementedError

    def reconstruct_text(self,
                         text: str,
                         labels: list[Union[str, int]]) -> str:
        """Given a text and a set of predicted white spaces, reconstruct
        the text by inserting the white spaces.

        Args:
            text (str): The considered text without white space.
            labels (list[str]): The list of labeled predictions.
                The label "space" should correspond to a white space.
        """
        result = []
        for char, label in zip(text, labels):
            result.append(char)
            if label == "space" or label == 1 or label == 3\
                    or label == "single" or label == "end":
                result.append(" ")
        return "".join(result).strip()

    @staticmethod
    def load_segmentation_model(model_path: str) -> tuple[Any, Any]:
        """Load a saved segmentation model"""
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        return model, tokenizer

    @staticmethod
    def _extract_boundaries(words: List[str]) -> Set[int]:
        """Extract word boundary positions from a list of words.

        A boundary is the position (0-indexed) after the last character
        of each word, excluding the final boundary (end of text).

        Args:
            words: List of words.

        Returns:
            Set of boundary positions.
        """
        boundaries = set()
        position = 0
        for i, word in enumerate(words):
            position += len(word)
            if i < len(words) - 1:
                boundaries.add(position)
        return boundaries

    @staticmethod
    def _align_texts(
        source: str,
        target: str
    ) -> List[Tuple[str, int, int, int, int]]:
        """Align two texts using difflib.SequenceMatcher.

        Args:
            source: Source text (ground truth continuous).
            target: Target text (noisy input).

        Returns:
            List of opcodes (op, src_start, src_end, tgt_start, tgt_end).
        """
        matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
        return matcher.get_opcodes()

    @staticmethod
    def _project_boundaries(
        gt_boundaries: Set[int],
        gt_continuous: str,
        noisy_text: str,
        opcodes: List[Tuple[str, int, int, int, int]]
    ) -> Set[int]:
        """Project boundaries from GT to noisy text via alignment.

        For each boundary in the GT, finds the corresponding position
        in the noisy text by following the character-by-character alignment.

        Args:
            gt_boundaries: Boundary positions in GT continuous text.
            gt_continuous: Ground truth without spaces.
            noisy_text: Noisy input text.
            opcodes: Alignment operations from difflib.

        Returns:
            Set of projected boundary positions in noisy text.
        """
        gt_to_noisy = {}
        gt_pos = 0
        noisy_pos = 0

        for op, src_start, src_end, tgt_start, tgt_end in opcodes:
            src_len = src_end - src_start
            tgt_len = tgt_end - tgt_start

            if op == 'equal':
                for i in range(src_len):
                    gt_to_noisy[gt_pos + i] = noisy_pos + i
                gt_pos += src_len
                noisy_pos += tgt_len

            elif op == 'replace':
                for i in range(src_len):
                    ratio = i / src_len if src_len > 0 else 0
                    mapped_pos = noisy_pos + int(ratio * tgt_len)
                    gt_to_noisy[gt_pos + i] = mapped_pos
                gt_pos += src_len
                noisy_pos += tgt_len

            elif op == 'delete':
                for i in range(src_len):
                    gt_to_noisy[gt_pos + i] = noisy_pos
                gt_pos += src_len

            elif op == 'insert':
                noisy_pos += tgt_len

        gt_to_noisy[len(gt_continuous)] = len(noisy_text)

        projected = set()
        for boundary in gt_boundaries:
            if boundary in gt_to_noisy:
                projected.add(gt_to_noisy[boundary])
            else:
                closest = min(
                    gt_to_noisy.keys(), key=lambda x: abs(x - boundary)
                )
                projected.add(gt_to_noisy[closest])

        return projected

    def compute_scores_boundaries(
        self,
        text: str,
        ground_truth: list[str],
        noisy: bool = False
    ) -> dict[str, float]:
        """Evaluate based on word boundary detection.

        Args:
            text: The text to segment (may contain OCR noise if noisy=True).
            ground_truth: The reference segmentation as a list of words.
            noisy: If True, use alignment-based projection for noisy OCR input.
                When the input text differs from the ground truth due to OCR
                errors (insertions, deletions, substitutions), this mode aligns
                the texts and projects reference boundaries accordingly.

        Returns:
            Dictionary with accuracy, precision, recall, f1, and counts.
        """
        segmented_words = self.segment(text)
        predicted_boundaries = self._extract_boundaries(segmented_words)

        if noisy:
            # Noisy mode: align GT with input and project boundaries
            gt_continuous = "".join(ground_truth)
            gt_boundaries = self._extract_boundaries(ground_truth)

            # Align and project boundaries
            opcodes = self._align_texts(gt_continuous, text)
            expected_boundaries = self._project_boundaries(
                gt_boundaries, gt_continuous, text, opcodes
            )
        else:
            # Clean mode: direct boundary comparison
            expected_boundaries = self._extract_boundaries(ground_truth)

        # Strict comparison
        true_positives = len(
            expected_boundaries.intersection(predicted_boundaries)
        )
        false_positives = len(predicted_boundaries - expected_boundaries)
        false_negatives = len(expected_boundaries - predicted_boundaries)

        precision = true_positives / \
            (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        accuracy = true_positives / len(expected_boundaries) \
            if expected_boundaries else 1.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "boundary_true_positives": true_positives,
            "boundary_false_positives": false_positives,
            "boundary_false_negatives": false_negatives
        }

    def compute_scores(
        self,
        text: str,
        ground_truth: list[str],
        noisy: bool = False
    ) -> dict[str, float]:
        """Given a properly segmented ground truth and a text
        to segment, perform the segmentation of the input text
        and compare it to the ground truth.
        Returns accuracy, precision, and recall scores.

        Args:
            text: The text to segment (may contain OCR noise if noisy=True).
            ground_truth: The reference segmentation as a list of words.
            noisy: If True, use alignment-based projection for noisy OCR input.
                When the input text differs from the ground truth due to OCR
                errors (insertions, deletions, substitutions), this mode aligns
                the texts and projects reference boundaries accordingly.

        Returns:
            dict[str, float]: Dictionary with accuracy, precision, and recall
                scores.
        """
        segmented_words = self.segment(text)

        # 1. Boundary-based evaluation (most robust to shifting)
        boundary_scores = self.compute_scores_boundaries(
            text, ground_truth, noisy=noisy
        )

        # 2. Word set evaluation (position-independent)
        # Note: In noisy mode, word matching is less meaningful due to OCR errors
        gt_set = set(ground_truth)
        pred_set = set(segmented_words)
        common_words = gt_set.intersection(pred_set)

        word_precision = len(common_words) / len(pred_set) if pred_set else 0.0
        word_recall = len(common_words) / len(gt_set) if gt_set else 0.0
        word_f1 = 2 * (word_precision * word_recall) / (word_precision +
                                                        word_recall) if \
            (word_precision + word_recall) > 0 else 0.0

        # 3. Traditional sequence evaluation (for reference)
        # Note: In noisy mode, sequence matching is less meaningful
        min_length = min(len(segmented_words), len(ground_truth))
        exact_matches = sum(1 for i in range(min_length)
                            if segmented_words[i] == ground_truth[i])
        sequence_accuracy = exact_matches / \
            len(ground_truth) if ground_truth else 0.0

        return {
            # Boundary metrics
            "boundary_accuracy": boundary_scores["accuracy"],
            "boundary_precision": boundary_scores["precision"],
            "boundary_recall": boundary_scores["recall"],
            "boundary_f1": boundary_scores["f1"],

            # Word-level metrics
            "word_precision": word_precision,
            "word_recall": word_recall,
            "word_f1": word_f1,

            # Traditional metrics (for comparison)
            "sequence_accuracy": sequence_accuracy,

            # Counts
            "expected_words": len(ground_truth),
            "predicted_words": len(segmented_words),
            "correct_words": len(common_words),
            "exact_position_matches": exact_matches
        }
