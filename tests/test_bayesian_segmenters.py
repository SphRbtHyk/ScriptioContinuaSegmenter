"""Tests that the Bayesian segmenter behaves as expected.
"""

import unittest
from sc_segmenter.segmenters.bayesian_segmenter import BayesianSegmenter


class TestBayesianSegmenter(unittest.TestCase):
    """Test BayesianSegmenter basic functionality."""

    def test_initialization(self):
        """Test that segmenter initializes with default parameters."""
        segmenter = BayesianSegmenter()

        # Check that model is created
        self.assertIsNotNone(segmenter.model)

        # Test with custom parameters
        segmenter_custom = BayesianSegmenter(
            n_dim=5,
            init_d=0.3,
            init_theta=1.5
        )
        self.assertIsNotNone(segmenter_custom.model)

    def test_segment_method(self):
        """Tests that the segmentation behaves properly."""
        segmenter = BayesianSegmenter()

        # Train on simple input
        segmenter.train(["thisisatest", "thisis"],
                        ["hello"])
        # Test
        result = segmenter.segment("helloworld")

        # Should return a list of strings
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(word, str) for word in result))

        # Results of the segmentation cannot be
        # checked directly as results are stochastic


if __name__ == '__main__':
    unittest.main()
