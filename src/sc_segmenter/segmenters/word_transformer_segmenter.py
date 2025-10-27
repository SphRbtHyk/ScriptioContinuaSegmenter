"""Perform segmentation using a one-level character-based segmenter.
A transformer is trained on an input corpora.
"""

import torch
from sc_segmenter.segmenters.segmenter import Segmenter


class ClassificationSegmenter(Segmenter):
    """Implements the basic classification segmenter, that
    uses a trained classification model (quadrimodal or bimodal) to
    predict if there is a space between words or no.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize a classification segmenter object.

        Args:
            model_path (str): The path to the tokenizer and the classifier
                model.
        """
        super().__init__()
        try:
            self.model, self.tokenizer = \
                self.load_segmentation_model(model_path)
        except Exception:
            raise ValueError(f"Could not retrieve models in {model_path}")

    def reconstruct(self, text: str) -> str:
        """Perform the segmentation of an input text.
        """
        return self.reconstruct_text(
            text=text,
            labels=self.predict_character_level(text=text)
        )

    def segment(self, text: str) -> list[str]:
        """Perform the segmentation of an input text.
        """
        return self.reconstruct(text).split()

    def predict_character_level(self,
                                text: str):
        """Given a text, perform a prediction at the character
        level."""
        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=128)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Get tokens and predictions
        input_ids = inputs["input_ids"][0]
        preds = predictions[0]

        results = []

        for i, (token_id, pred) in enumerate(zip(input_ids, preds)):

            # Skip special tokens
            if token_id not in [self.tokenizer.cls_token_id,
                                self.tokenizer.sep_token_id,
                                self.tokenizer.pad_token_id]:
                results.append(pred.item())
        return results


if __name__ == "__main__":
    # SANITY CHECK FOR NOW
    MODEL_PATH = "../../../notebooks/character-classifier-final"

    try:
        # Initialize segmenter
        print("Loading model...")
        segmenter = ClassificationSegmenter(MODEL_PATH)
        print("Model loaded successfully!")

        # Test cases
        test_cases = [
            "οιησουειπεν"
        ]

        print("\nTesting segmentation:")
        print("-" * 40)

        for test_text in test_cases:
            print(f"Input:    '{test_text}'")
            segmented = segmenter.segment(test_text)
            print(f"Output:   '{segmented}'")

            # Show character-level predictions
            predictions = segmenter.predict_character_level(test_text)
            print(f"Labels:   {predictions}")
            print("-" * 40)

        # Sanity check regarding evaluation
        print("=== SCORES")
        print(
            segmenter.compute_scores(
                "οιησουειπεν",
                ["ο", "ιησου", "ειπεν"]
            ))

    except Exception as e:
        print(f"Error: {e}")
