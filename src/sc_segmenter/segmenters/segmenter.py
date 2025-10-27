"""Base class for the different segmenters."""
from typing import Union
from typing import Any
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
            if label == "space" or label == 1:
                result.append(" ")
        return "".join(result).strip()

    @staticmethod
    def load_segmentation_model(model_path: str) -> tuple[Any, Any]:
        """Load a saved segmentation model"""
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        return model, tokenizer

    def compute_scores_boundaries(self,
                                  text: str,
                                  ground_truth: list[str]) -> dict[str, float]:
        """Evaluate based on word boundary detection."""
        segmented_words = self.segment(text)

        # Generate expected boundaries from ground truth
        expected_boundaries = set()
        current_pos = 0
        for word in ground_truth:
            current_pos += len(word)
            expected_boundaries.add(current_pos)

        # Generate predicted boundaries
        predicted_boundaries = set()
        current_pos = 0
        for word in segmented_words:
            current_pos += len(word)
            predicted_boundaries.add(current_pos)

        # Remove the final boundary (end of text)
        final_pos = len(text)
        expected_boundaries.discard(final_pos)
        predicted_boundaries.discard(final_pos)

        # Calculate boundary-based metrics
        true_positives = len(
            expected_boundaries.intersection(predicted_boundaries))
        false_positives = len(predicted_boundaries - expected_boundaries)
        false_negatives = len(expected_boundaries - predicted_boundaries)

        precision = true_positives / \
            (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / \
            (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / \
            (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy: proportion of correct boundaries
        accuracy = true_positives / \
            len(expected_boundaries) if expected_boundaries else 1.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "boundary_true_positives": true_positives,
            "boundary_false_positives": false_positives,
            "boundary_false_negatives": false_negatives
        }

    def compute_scores(self,
                       text: str,
                       ground_truth: list[str]) -> dict[str, float]:
        """Given a properly segmented ground truth and a text
        to segment, perform the segmentation of the input text
        and compare it to the ground truth.
        Returns accuracy, precision, and recall scores.

        Args:
            text (str): The text to segment.
            ground_truth (str): The corrected segmented text.

        Returns:
            dict[str, float]: Dictionary with accuracy, precision, and recall
                scores.
        """
        segmented_words = self.segment(text)

        # 1. Boundary-based evaluation (most robust to shifting)
        boundary_scores = self.compute_scores_boundaries(text, ground_truth)

        # 2. Word set evaluation (position-independent)
        gt_set = set(ground_truth)
        pred_set = set(segmented_words)
        common_words = gt_set.intersection(pred_set)

        word_precision = len(common_words) / len(pred_set) if pred_set else 0.0
        word_recall = len(common_words) / len(gt_set) if gt_set else 0.0
        word_f1 = 2 * (word_precision * word_recall) / (word_precision +
                                                        word_recall) if \
            (word_precision + word_recall) > 0 else 0.0

        # 3. Traditional sequence evaluation (for reference)
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
