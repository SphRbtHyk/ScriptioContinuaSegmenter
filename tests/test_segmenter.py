"""Performs unit-tests for the Segmenter containing base
classes for compute the scores.
"""
import unittest
# Replace with actual import path
from sc_segmenter.segmenters.segmenter import Segmenter


class TestSegmenter(unittest.TestCase):
    """Test Segmenter base class functionality."""

    def test_reconstruct_text_various_labels(self):
        """Test reconstruct_text method with different label types."""
        segmenter = Segmenter()

        test_cases = [
            # (text, labels, expected_result)
            ("hello", [0, 0, 0, 0, 0], "hello"),
            ("hello", [0, 0, 1, 0, 0], "hel lo"),  # space after 3rd char
            ("hello", [0, 1, 0, 1, 0], "he ll o"),
        ]

        for text, labels, expected in test_cases:
            result = segmenter.reconstruct_text(text,
                                                labels)
            self.assertEqual(result, expected)

    def test_compute_scores_boundaries_basic(self):
        """Test boundary-based scoring with simple cases."""
        segmenter = Segmenter()

        # Mock the segment method that is not implemented
        # in the base class
        def mock_segment(text):
            return ["hello", "world"]

        segmenter.segment = mock_segment

        # Test case: perfect match
        text = "helloworld"
        ground_truth = ["hello", "world"]

        scores = segmenter.compute_scores_boundaries(text, ground_truth)

        self.assertEqual(scores["accuracy"], 1.0)
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["f1"], 1.0)

    def test_compute_scores_boundaries_mismatch(self):
        """Test boundary-based scoring with incorrect segmentation."""
        segmenter = Segmenter()

        def mock_segment(text):
            return ["he", "llo", "world"]  # Boundaries at 2, 5

        segmenter.segment = mock_segment

        text = "helloworld"
        ground_truth = ["hello", "world"]  # Boundary only at 5

        scores = segmenter.compute_scores_boundaries(text, ground_truth)

        self.assertEqual(scores["boundary_true_positives"], 1)
        self.assertEqual(scores["boundary_false_positives"], 1)
        self.assertEqual(scores["boundary_false_negatives"], 0)

    def test_compute_scores_comprehensive(self):
        """Test the comprehensive scoring method."""
        segmenter = Segmenter()

        def mock_segment(text):
            # Same words as ground truth but different boundaries
            return ["hello", "world"]

        segmenter.segment = mock_segment

        text = "helloworld"
        ground_truth = ["hello", "world"]

        scores = segmenter.compute_scores(text, ground_truth)

        # Check all score types are present
        expected_keys = [
            "boundary_accuracy",
            "boundary_precision",
            "boundary_recall",
            "boundary_f1",
            "word_precision", "word_recall", "word_f1",
            "sequence_accuracy",
            "expected_words", "predicted_words", "correct_words",
            "exact_position_matches"
        ]

        for key in expected_keys:
            self.assertIn(key, scores)
            self.assertIsInstance(scores[key], (int, float))

    def test_compute_scores_empty_cases(self):
        """Test scoring with edge cases like empty inputs."""
        segmenter = Segmenter()

        def mock_segment(text):
            return [] if text == "" else ["a"]

        segmenter.segment = mock_segment

        # Test empty text
        scores_empty = segmenter.compute_scores("", [])
        self.assertEqual(scores_empty["expected_words"], 0)
        self.assertEqual(scores_empty["predicted_words"], 0)

        # Test single character
        scores_single = segmenter.compute_scores("a", ["a"])
        self.assertEqual(scores_single["expected_words"], 1)
        self.assertEqual(scores_single["predicted_words"], 1)

    def test_segment_method_not_implemented(self):
        """Test that segment method raises NotImplementedError."""
        segmenter = Segmenter()

        with self.assertRaises(NotImplementedError):
            segmenter.segment("test")


if __name__ == '__main__':
    unittest.main()
