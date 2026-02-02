"""Generate noisy datasets in the noisy_datasets folder using learned noise
(scrambledtext package), separately for Greek, Latin, and seals.

Output structure:
    noisy_datasets/
        greek/
            greek_mild.json, greek_sota.json, greek_severe.json
            (+ corresponding _distributions.json files)
        lat/
            lat_mild.json, lat_sota.json, lat_severe.json
            (+ corresponding _distributions.json files)
        seals/
            seals_mild.json, seals_sota.json
            (+ corresponding _distributions.json files)
"""

import json
import os
from pathlib import Path

from generate_learned_noise import LearntNoiseGenerator

# Learned noise configurations per language type
# Based on different statistical distributions of error per artefact type
LEARNED_CONFIGS = {
    # Codices (Greek): SOTA=5%, mild=2%, severe=10%
    "greek": [
        {"name": "mild", "cer": 0.02},
        {"name": "sota", "cer": 0.05},
        {"name": "severe", "cer": 0.10},
    ],
    # Codices (Latin): SOTA=4%, mild=2%, severe=10%
    "lat": [
        {"name": "mild", "cer": 0.02},
        {"name": "sota", "cer": 0.04},
        {"name": "severe", "cer": 0.10},
    ],
    # Seals: SOTA=30%, mild=15%, no severe (current SOTA already challenging)
    "seals": [
        {"name": "mild", "cer": 0.15},
        {"name": "sota", "cer": 0.30},
    ],
}

# Special character for insertions/deletions in alignment
GAP_CHAR = "⌀"

# Standard alphabets (only these characters will be kept)
GREEK_ALPHABET = set("αβγδεζηθικλμνξοπρστυφχψως ")
LATIN_ALPHABET = set("abcdefghijklmnopqrstuvwxyz ")

# Character normalization mappings (applied before alphabet filtering)
GREEK_NORMALIZATIONS = {
    "ϲ": "σ",  # lunate sigma -> standard sigma
    "ϛ": "στ",  # stigma -> sigma tau
    "ͻ": "σ",  # reversed lunate sigma
    "Ϲ": "Σ",  # capital lunate sigma
    # Polytonic Greek -> monotonic
    "ἀ": "α", "ἁ": "α", "ἂ": "α", "ἃ": "α", "ἄ": "α", "ἅ": "α", "ἆ": "α", "ἇ": "α",
    "ᾀ": "α", "ᾁ": "α", "ᾂ": "α", "ᾃ": "α", "ᾄ": "α", "ᾅ": "α", "ᾆ": "α", "ᾇ": "α",
    "ὰ": "α", "ά": "α", "ᾰ": "α", "ᾱ": "α", "ᾲ": "α", "ᾳ": "α", "ᾴ": "α", "ᾶ": "α", "ᾷ": "α",
    "ἐ": "ε", "ἑ": "ε", "ἒ": "ε", "ἓ": "ε", "ἔ": "ε", "ἕ": "ε",
    "ὲ": "ε", "έ": "ε",
    "ἠ": "η", "ἡ": "η", "ἢ": "η", "ἣ": "η", "ἤ": "η", "ἥ": "η", "ἦ": "η", "ἧ": "η",
    "ᾐ": "η", "ᾑ": "η", "ᾒ": "η", "ᾓ": "η", "ᾔ": "η", "ᾕ": "η", "ᾖ": "η", "ᾗ": "η",
    "ὴ": "η", "ή": "η", "ῂ": "η", "ῃ": "η", "ῄ": "η", "ῆ": "η", "ῇ": "η",
    "ἰ": "ι", "ἱ": "ι", "ἲ": "ι", "ἳ": "ι", "ἴ": "ι", "ἵ": "ι", "ἶ": "ι", "ἷ": "ι",
    "ὶ": "ι", "ί": "ι", "ῐ": "ι", "ῑ": "ι", "ῒ": "ι", "ΐ": "ι", "ῖ": "ι", "ῗ": "ι",
    "ὀ": "ο", "ὁ": "ο", "ὂ": "ο", "ὃ": "ο", "ὄ": "ο", "ὅ": "ο",
    "ὸ": "ο", "ό": "ο",
    "ὐ": "υ", "ὑ": "υ", "ὒ": "υ", "ὓ": "υ", "ὔ": "υ", "ὕ": "υ", "ὖ": "υ", "ὗ": "υ",
    "ὺ": "υ", "ύ": "υ", "ῠ": "υ", "ῡ": "υ", "ῢ": "υ", "ΰ": "υ", "ῦ": "υ", "ῧ": "υ",
    "ὠ": "ω", "ὡ": "ω", "ὢ": "ω", "ὣ": "ω", "ὤ": "ω", "ὥ": "ω", "ὦ": "ω", "ὧ": "ω",
    "ᾠ": "ω", "ᾡ": "ω", "ᾢ": "ω", "ᾣ": "ω", "ᾤ": "ω", "ᾥ": "ω", "ᾦ": "ω", "ᾧ": "ω",
    "ὼ": "ω", "ώ": "ω", "ῲ": "ω", "ῳ": "ω", "ῴ": "ω", "ῶ": "ω", "ῷ": "ω",
    "ῤ": "ρ", "ῥ": "ρ",
    GAP_CHAR: "",  # Remove gap character
}

LATIN_NORMALIZATIONS = {
    GAP_CHAR: "",  # Remove gap character
}


def normalize_text(text: str, normalizations: dict) -> str:
    """Normalize text by replacing variant characters with standard ones."""
    for old, new in normalizations.items():
        text = text.replace(old, new)
    return text


def filter_to_alphabet(text: str, alphabet: set) -> str:
    """Keep only characters that are in the allowed alphabet."""
    return "".join(c for c in text if c in alphabet)


def align_texts(gt: str, noisy: str) -> tuple[str, str]:
    """Align two strings using edit distance, inserting gap characters.

    Returns aligned strings of equal length where:
    - Matching characters are preserved
    - Substitutions show the different characters at same position
    - Deletions in noisy text show GAP_CHAR in noisy string
    - Insertions in noisy text show GAP_CHAR in gt string

    Args:
        gt: Ground truth string.
        noisy: Noisy OCR string.

    Returns:
        Tuple of (aligned_gt, aligned_noisy) with equal lengths.
    """
    m, n = len(gt), len(noisy)

    # Build DP table for edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == noisy[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    # Backtrack to find alignment
    aligned_gt = []
    aligned_noisy = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i - 1] == noisy[j - 1]:
            aligned_gt.append(gt[i - 1])
            aligned_noisy.append(noisy[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            aligned_gt.append(gt[i - 1])
            aligned_noisy.append(noisy[j - 1])
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # Insertion in noisy
            aligned_gt.append(GAP_CHAR)
            aligned_noisy.append(noisy[j - 1])
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion in noisy
            aligned_gt.append(gt[i - 1])
            aligned_noisy.append(GAP_CHAR)
            i -= 1
        else:
            # Fallback (shouldn't happen)
            if i > 0:
                aligned_gt.append(gt[i - 1])
                aligned_noisy.append(GAP_CHAR)
                i -= 1
            elif j > 0:
                aligned_gt.append(GAP_CHAR)
                aligned_noisy.append(noisy[j - 1])
                j -= 1

    return "".join(reversed(aligned_gt)), "".join(reversed(aligned_noisy))


def load_transcription_pairs(
    corrected_dir: Path,
    suffix: str,
    alphabet: set,
    normalizations: dict,
) -> tuple[list[str], list[str]]:
    """Load ground truth and OCR transcription pairs, aligned for scrambledtext.

    Args:
        corrected_dir: Path to the corrected transcriptions directory.
        suffix: 'v' for Greek or 'r' for Latin.
        alphabet: Set of allowed characters.
        normalizations: Dict of character replacements to normalize.

    Returns:
        Tuple of (ground_truth_texts, noisy_texts) with aligned character lengths.
    """
    ground_truths = []
    noisy_texts = []

    # Alphabet without space for alignment
    alphabet_no_space = alphabet - {" "}

    for filename in sorted(os.listdir(corrected_dir)):
        if filename.endswith("_gt.txt"):
            continue
        if not filename.endswith(".txt"):
            continue

        base_name = filename.replace(".txt", "")
        if not base_name.endswith(suffix):
            continue

        gt_file = corrected_dir / f"{base_name}_gt.txt"
        trans_file = corrected_dir / filename

        if not gt_file.exists():
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_text = f.read().strip().replace(" ", "").lower()

        with open(trans_file, "r", encoding="utf-8") as f:
            noisy_text = f.read().strip().replace(" ", "").lower()

        # Normalize and filter to allowed alphabet
        gt_text = normalize_text(gt_text, normalizations)
        gt_text = filter_to_alphabet(gt_text, alphabet_no_space)
        noisy_text = normalize_text(noisy_text, normalizations)
        noisy_text = filter_to_alphabet(noisy_text, alphabet_no_space)

        # Align the texts to have equal length
        aligned_gt, aligned_noisy = align_texts(gt_text, noisy_text)
        ground_truths.append(aligned_gt)
        noisy_texts.append(aligned_noisy)

    return ground_truths, noisy_texts


def load_dataset(dataset_path: Path) -> dict:
    """Load a manuscript dataset.

    Args:
        dataset_path: Path to ms.json file.

    Returns:
        The dataset dictionary with structure {MS: {CHAPTER: {VERSE: text}}}.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def corrupt_dataset(
    dataset: dict,
    generator,
    alphabet: set,
    normalizations: dict,
) -> dict:
    """Apply noise generator to all verses in the dataset.

    Args:
        dataset: Original dataset with structure {MS: {CHAPTER: {VERSE: text}}}.
        generator: Noise generator with generate() method.
        alphabet: Set of allowed characters to keep in output.
        normalizations: Dict of character replacements to normalize.

    Returns:
        New dataset with same structure but corrupted text.
    """
    corrupted = {}
    for ms_name, chapters in dataset.items():
        corrupted[ms_name] = {}
        for chapter, verses in chapters.items():
            corrupted[ms_name][chapter] = {}
            for verse_num, text in verses.items():
                # Remove spaces to treat as single word for noise generation
                text_no_spaces = text.replace(" ", "")
                # Generate corrupted text
                noisy = generator.generate(text_no_spaces)
                # Clean up: normalize and filter to alphabet
                noisy = normalize_text(noisy, normalizations)
                noisy = filter_to_alphabet(noisy, alphabet)
                # Remove any GAP_CHAR that might have leaked through
                noisy = noisy.replace(GAP_CHAR, "")
                corrupted[ms_name][chapter][verse_num] = noisy
    return corrupted


def save_dataset(dataset: dict, output_path: Path):
    """Save dataset to JSON file.

    Args:
        dataset: Dataset dictionary.
        output_path: Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


def generate_learned_noise_dataset(
    dataset: dict,
    ground_truths: list[str],
    noisy_texts: list[str],
    output_path: Path,
    distributions_path: Path,
    language: str,
    target_cer: float,
    alphabet: set,
    normalizations: dict,
) -> dict | None:
    """Generate dataset using learned noise patterns.

    Args:
        dataset: Original clean dataset.
        ground_truths: Ground truth texts for learning.
        noisy_texts: OCR outputs for learning.
        output_path: Path to save the output dataset.
        distributions_path: Path to save the learned distributions.
        language: 'greek' or 'latin' for logging.
        target_cer: Target Character Error Rate for this language.
        alphabet: Set of allowed characters.
        normalizations: Dict of character replacements.

    Returns:
        Corrupted dataset or None if no transcription pairs available.
    """
    if not ground_truths or not noisy_texts:
        print(f"  Warning: No transcription pairs for {language}, skipping.")
        return None

    print(f"  Learning from {len(ground_truths)} {language} transcription pairs...")
    print(f"  Target CER: {target_cer:.2f}")

    generator = LearntNoiseGenerator.from_aligned_texts(
        ground_truth=ground_truths,
        noisy_texts=noisy_texts,
        target_cer=target_cer,
        target_wer=1,  # Single word, so WER is always 1 if any error
    )

    # Save the learned distributions (Markovian analysis results)
    distributions_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_distributions(str(distributions_path))
    print(f"  Saved distributions to {distributions_path}")

    print("  Generating noisy versions...")
    corrupted = corrupt_dataset(dataset, generator, alphabet, normalizations)

    save_dataset(corrupted, output_path)
    print(f"  Saved dataset to {output_path}")

    return corrupted


def process_language(
    language: str,
    dataset_path: Path,
    corrected_dir: Path,
    output_base: Path,
    transcription_suffix: str,
    alphabet: set,
    normalizations: dict,
):
    """Process a single language: load data, generate all noise variants.

    Args:
        language: 'greek', 'lat', or 'seals'.
        dataset_path: Path to the clean ms.json file.
        corrected_dir: Path to corrected transcriptions directory.
        output_base: Base output directory for noisy datasets.
        transcription_suffix: 'v' for Greek, 'r' for Latin, 'seals' for seals.
        alphabet: Set of allowed characters for this language.
        normalizations: Dict of character replacements for this language.
    """
    print(f"\n{'='*60}")
    print(f"Processing {language.upper()}")
    print(f"{'='*60}")

    # Load clean dataset
    print(f"\nLoading {language} dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    verse_count = sum(
        len(verses)
        for chapters in dataset.values()
        for verses in chapters.values()
    )
    print(f"  Loaded {len(dataset)} manuscripts, {verse_count} verses.")

    # Load transcription pairs for learning
    print(f"\nLoading {language} transcription pairs (suffix: {transcription_suffix})...")
    ground_truths, noisy_texts = load_transcription_pairs(
        corrected_dir, transcription_suffix, alphabet, normalizations
    )
    print(f"  Found {len(ground_truths)} transcription pairs.")

    output_dir = output_base / language

    # Get configs for this language
    configs = LEARNED_CONFIGS[language]

    # Generate learned noise datasets with varying CER levels
    for config in configs:
        print(f"\n--- Learned Noise: {config['name']} ({language}) ---")
        print(f"  Target CER: {config['cer']}")
        generate_learned_noise_dataset(
            dataset=dataset,
            ground_truths=ground_truths,
            noisy_texts=noisy_texts,
            output_path=output_dir / f"{language}_{config['name']}.json",
            distributions_path=output_dir / f"{language}_{config['name']}_distributions.json",
            language=language,
            target_cer=config['cer'],
            alphabet=alphabet,
            normalizations=normalizations,
        )


def main():
    script_dir = Path(__file__).parent
    corrected_dir = script_dir / "transcriptions" / "corrected"
    output_base = script_dir / "noisy_datasets"

    # Dataset paths
    greek_dataset_path = script_dir / "benchmark" / "test_datasets" / "grc" / "ms.json"
    latin_dataset_path = script_dir / "benchmark" / "test_datasets" / "lat" / "ms.json"
    seals_dataset_path = script_dir / "benchmark" / "test_datasets" / "seals" / "ms.json"

    # Check if running from scripts/ocr or need to adjust paths
    if not greek_dataset_path.exists():
        # Try relative to project root
        greek_dataset_path = script_dir.parent / "benchmark" / "test_datasets" / "grc" / "ms.json"
        latin_dataset_path = script_dir.parent / "benchmark" / "test_datasets" / "lat" / "ms.json"
        seals_dataset_path = script_dir.parent / "benchmark" / "test_datasets" / "seals" / "ms.json"

    # Process Greek (v transcriptions -> Greek noise patterns)
    if greek_dataset_path.exists():
        process_language(
            language="greek",
            dataset_path=greek_dataset_path,
            corrected_dir=corrected_dir,
            output_base=output_base,
            transcription_suffix="v",  # Greek transcriptions are verso pages
            alphabet=GREEK_ALPHABET,
            normalizations=GREEK_NORMALIZATIONS,
        )
    else:
        print(f"Warning: Greek dataset not found at {greek_dataset_path}")

    # Process Latin (r transcriptions -> Latin noise patterns)
    if latin_dataset_path.exists():
        process_language(
            language="lat",
            dataset_path=latin_dataset_path,
            corrected_dir=corrected_dir,
            output_base=output_base,
            transcription_suffix="r",  # Latin transcriptions are recto pages
            alphabet=LATIN_ALPHABET,
            normalizations=LATIN_NORMALIZATIONS,
        )
    else:
        print(f"Warning: Latin dataset not found at {latin_dataset_path}")

    # Process Seals (uses Greek alphabet, v transcriptions for noise learning)
    if seals_dataset_path.exists():
        process_language(
            language="seals",
            dataset_path=seals_dataset_path,
            corrected_dir=corrected_dir,
            output_base=output_base,
            transcription_suffix="v",  # Seals use Greek transcriptions for noise
            alphabet=GREEK_ALPHABET,
            normalizations=GREEK_NORMALIZATIONS,
        )
    else:
        print(f"Warning: Seals dataset not found at {seals_dataset_path}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
