"""Generate synthetic data from noise learned from OCR transcriptions."""

import json
from scrambledtext import CorruptionEngine, ProbabilityDistributions


class LearntNoiseGenerator:
    """Generate text from a learnt noise distribution using
    the `scrambledtext` package."""

    def __init__(
        self,
        distributions_path: str,
        target_cer: float,
        target_wer: int,
    ) -> None:
        """Constructor for the learnt noise generator.

        Args:
            distributions_path (str): Path to JSON file containing learned
                probability distributions from aligned OCR text pairs.
            target_cer (float): Target Character Error Rate.
            target_wer (int): Target Word Error Rate.
        """
        self.distributions = ProbabilityDistributions.load_from_json(
            distributions_path
        )
        self.engine = CorruptionEngine(
            conditional_probs=self.distributions.conditional,
            substitution_table=self.distributions.substitutions,
            insertion_table=self.distributions.insertions,
            target_cer=target_cer,
            target_wer=target_wer,
        )

    def generate(self, text: str) -> str:
        """Generate a new text presenting synthetic noise.

        Applies OCR-like corruptions based on learned error patterns.

        Args:
            text (str): The input text to add noise to.

        Returns:
            str: The text with learned noise applied.
        """
        if not text:
            return text

        corrupted_text, _, _, _ = self.engine.corrupt_text(text)
        return corrupted_text

    @classmethod
    def from_aligned_texts(
        cls,
        ground_truth: list[str],
        noisy_texts: list[str],
        target_cer: float,
        target_wer: int,
    ) -> "LearntNoiseGenerator":
        """Create a generator by learning from aligned text pairs.

        Args:
            ground_truth (list[str]): List of ground truth texts.
            noisy_texts (list[str]): List of corresponding noisy OCR outputs.
            target_cer (Optional[float]): Target Character Error Rate.
            target_wer (Optional[float]): Target Word Error Rate.

        Returns:
            LearntNoiseGenerator: A new generator with learned distributions.
        """
        instance = cls.__new__(cls)
        instance.distributions = ProbabilityDistributions(
            (truth, noise) for truth, noise in zip(ground_truth, noisy_texts)
        )
        instance.engine = CorruptionEngine(
            conditional_probs=instance.distributions.conditional,
            substitution_table=instance.distributions.substitutions,
            insertion_table=instance.distributions.insertions,
            target_cer=target_cer,
            target_wer=target_wer,
        )
        return instance

    def save_distributions(self, path: str, exclude_chars: set = None) -> None:
        """Save the learned distributions to a JSON file.

        Args:
            path (str): Path where to save the distributions.
            exclude_chars (set): Characters to exclude (default: gap char).
        """
        if exclude_chars is None:
            exclude_chars = {"⌀"}

        # Filter out excluded characters
        char_dist = {
            k: v for k, v in self.distributions.character_distribution.items()
            if k not in exclude_chars
        }
        conditional = {
            k: dict(v) for k, v in self.distributions.conditional.items()
            if k not in exclude_chars
        }
        substitutions = {
            k: dict(v) for k, v in self.distributions.substitutions.items()
            if k not in exclude_chars
        }
        insertions = {
            k: dict(v) for k, v in self.distributions.insertions.items()
            if k not in exclude_chars
        }

        data = {
            "character_distribution": char_dist,
            "conditional": conditional,
            "substitutions": substitutions,
            "insertions": insertions,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
