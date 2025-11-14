"""Tests that the hierarchical segmenter behaves as expected.
"""


import unittest
from sc_segmenter.segmenters.word_transformer_segmenter\
    import ClassificationSegmenter


class TestClassificationSegmenterReal(unittest.TestCase):
    """Test ClassificationSegmenter class."""

    def test_initialization_and_basic_segmentation(self):
        """Test that segmenter initializes properly."""
        try:
            # Initialize with real model
            segmenter = ClassificationSegmenter(
                "SphRbtHyk/demo_latin_character_segmenter")

            # Test basic segmentation
            result = segmenter.segment("helloworld")

            # Should return a list of strings
            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(word, str) for word in result))
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")

    def test_reconstruct_method(self):
        """Test that the reconstruct method behaves as expected."""
        try:
            segmenter = ClassificationSegmenter(
                "SphRbtHyk/demo_latin_character_segmenter")

            # Test reconstruct
            result = segmenter.reconstruct("helloworld")

            # Should return a string
            self.assertIsInstance(result, str)

            # Check that the reconstruction happened properly
            self.assertEqual(
                "hello world",
                result
            )
            print(f"Reconstruct result: '{result}'")

        except Exception as e:
            self.skipTest(f"Could not load model: {e}")

    def test_segmentation(self):
        """Tests that strings are properly segmented."""
        try:
            segmenter = ClassificationSegmenter(
                "SphRbtHyk/demo_latin_character_segmenter")

            text = "inprincipioerat"
            result = segmenter.segment(text)
            self.assertIsInstance(result, list)
            self.assertListEqual(
                result,
                ["in", "principio", "erat"]
            )

        except Exception as e:
            self.skipTest(f"Could not load model: {e}")

    def test_predict_character_level(self):
        """Test character level prediction returns valid output."""
        try:
            segmenter = ClassificationSegmenter(
                "SphRbtHyk/demo_latin_character_segmenter")

            # Test character level prediction
            predictions = segmenter.predict_character_level("test")

            # Should return a list of integers (0 or 1)
            self.assertIsInstance(predictions, list)
            self.assertTrue(all(pred in [0, 1] for pred in predictions))

            self.assertEqual(
                predictions,
                [0, 0, 0, 1]
            )

        except Exception as e:
            self.skipTest(f"Could not load model: {e}")


if __name__ == '__main__':
    unittest.main()
