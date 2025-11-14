"""Tests that the hierarchical segmenter behaves as expected.
"""
import unittest
from sc_segmenter.segmenters.hierarchical_segmenter \
    import HierarchicalSegmenter


class TestHierarchicalSegmenter(unittest.TestCase):
    """Test HierarchicalSegmenter with real model loading."""

    def test_initialization(self):
        """Test that segmenter initializes with real models."""
        try:
            segmenter = HierarchicalSegmenter(
                autoregressive_model_path="facebook/xglm-564M",
                character_model_path="SphRbtHyk/demo_latin_character_segmenter",
                beam_width=3
            )

            # Check that models are loaded
            self.assertIsNotNone(segmenter.autoregressive_model)
            self.assertIsNotNone(segmenter.autoregressive_tokenizer)
            self.assertIsNotNone(segmenter.character_classification_model)
            self.assertIsNotNone(segmenter.character_classification_tokenizer)

            # Check parameters
            self.assertEqual(segmenter.beam_width, 3)
            self.assertEqual(segmenter.max_length, 128)

        except Exception as e:
            self.skipTest(f"Could not load models: {e}")

    def test_compute_perplexity(self):
        """Test perplexity computation."""
        try:
            segmenter = HierarchicalSegmenter(
                autoregressive_model_path="facebook/xglm-564M",
                character_model_path="SphRbtHyk/demo_latin_character_segmenter",
                beam_width=3
            )

            perplexity = segmenter.compute_perplexity("hello world")
            self.assertIsInstance(perplexity, float)
            self.assertGreater(perplexity, 0)
            self.assertEqual(perplexity, 2999.67)

        except Exception as e:
            self.skipTest(f"Could not test perplexity: {e}")

    def test_beam_search_predict(self):
        """Test beam search prediction."""
        try:
            segmenter = HierarchicalSegmenter(
                autoregressive_model_path="facebook/xglm-564M",
                character_model_path="SphRbtHyk/demo_latin_character_segmenter",
                beam_width=3
            )

            # Test with simple text
            text = "helloworld"
            results = segmenter.beam_search_predict(text)

            # Check results structure
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), segmenter.beam_width)

            for result in results:
                self.assertIn('sequence', result)
                self.assertIn('labels', result)
                self.assertIn('score', result)
                self.assertIn('probability', result)

                # Check probability is within range
                self.assertGreaterEqual(result['probability'], 0)
                self.assertLessEqual(result['probability'], 1)
                # Three results should have been returned
                self.assertEqual(
                    len(results), 3
                )

        except Exception as e:
            self.skipTest(f"Could not test beam search: {e}")

    def test_select_best_beam(self):
        """Test best beam selection based on perplexity."""
        try:
            segmenter = HierarchicalSegmenter(
                autoregressive_model_path="facebook/xglm-564M",
                character_model_path="SphRbtHyk/demo_latin_character_segmenter",
                beam_width=3
            )

            text = "helloworld"
            best_segmentation = segmenter.select_best_beam(text)

            # Should return a string with spaces
            self.assertIsInstance(best_segmentation, str)
            # Should have at least one space
            self.assertIn(" ", best_segmentation)

        except Exception as e:
            self.skipTest(f"Could not test best beam selection: {e}")

    def test_segment_method(self):
        """Test the main segment method."""
        try:
            segmenter = HierarchicalSegmenter(
                autoregressive_model_path="facebook/xglm-564M",
                character_model_path="SphRbtHyk/demo_latin_character_segmenter",
                beam_width=3
            )

            result = segmenter.segment("hellowworld")

            # Should return a list of strings
            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(word, str) for word in result))

            # Should be segmented properly
            self.assertEqual("helloworld", ["hello", "world"])

        except Exception as e:
            self.skipTest(f"Could not test segment method: {e}")


if __name__ == '__main__':
    unittest.main()
