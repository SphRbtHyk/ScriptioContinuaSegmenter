"""Unit tests for the ground truth annotator."""
import unittest
from sc_segmenter.training.ground_truth_annotator import binary_annotation, \
    quadrimodal_annotation, annotate_dict


class TestAnnotator(unittest.TestCase):
    """Test the ground annotating functions."""

    def test_binary_annotation(self):
        """Tests that the binary annotation behaves as expected.
        """
        ground_truth = "un test"
        annotations = binary_annotation(ground_truth)
        self.assertListEqual(
            annotations,
            ["I", "E", "I", "I", "I", "E"]
        )

    def test_quadrimodal_annotation(self):
        """Tests that the quadrimodal annotation behaves as expected.
        """
        ground_truth = "un test t"
        annotations = quadrimodal_annotation(ground_truth)
        self.assertListEqual(
            annotations,
            ["B", "E", "B", "I", "I", "E", "S"]
        )

    def test_annotate_dict_binary(self):
        """Tests that annotating a dictionary behaves as expected."""
        test_data = {
            "livre1": {
                "1": {
                    "1": "ceci est un test",
                    "2": "test aussi"
                },
                "2": {
                    "1": "ceci est un test",
                }
            }}
        self.assertEqual(
            annotate_dict(test_data, annotation_scheme="binary"),
            [('ceciestuntest',
              ['I', 'I', 'I', 'E', 'I', 'I', 'E',
               'I', 'E', 'I', 'I', 'I', 'E'],
              "ceci est un test"),
             ('testaussi',
              ['I', 'I', 'I', 'E', 'I',
               'I', 'I', 'I', 'E'],
              "test aussi"),
             ('ceciestuntest',
              ['I', 'I', 'I', 'E', 'I', 'I', 'E',
               'I', 'E', 'I', 'I', 'I', 'E'],
              "ceci est un test")]
        )

    def test_annotate_dict_quadri(self):
        """Tests that annotating a dictionary behaves as expected."""
        test_data = {
            "livre1": {
                "1": {
                    "1": "ceci est un test",
                    "2": "test aussi"
                },
                "2": {
                    "1": "ceci est un test",
                }
            }}
        self.assertEqual(
            annotate_dict(test_data,
                          annotation_scheme="quadrimodal"),
            [('ceciestuntest',
              ['B', 'I', 'I', 'E', 'B', 'I', 'E',
               'B', 'E', 'B', 'I', 'I', 'E'],
              "ceci est un test"),
             ('testaussi',
              ['B', 'I', 'I', 'E', 'B',
               'I', 'I', 'I', 'E'],
              "test aussi"),
             ('ceciestuntest',
              ['B', 'I', 'I', 'E', 'B', 'I',
               'E', 'B', 'E', 'B', 'I', 'I', 'E'],
              "ceci est un test")]
        )

    def test_annotate_dict_unknown(self):
        """Tests that annotating a dictionary behaves as expected."""
        test_data = {
            "livre1": {
                "1": {
                    "1": "ceci est un test",
                    "2": "test aussi"
                },
                "2": {
                    "1": "ceci est un test",
                }
            }}
        with self.assertRaises(ValueError):
            annotate_dict(
                test_data,
                annotation_scheme="unknown"
            )


if __name__ == "__main__":
    unittest.main()
