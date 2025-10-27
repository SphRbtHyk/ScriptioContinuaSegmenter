"""Run the benchmarking process.

For each of the verse of the dataset, compute the accuracy, precision and
recall of the approach.
"""
import time
import random
import pathlib
import json
import pandas as pd
from sc_segmenter.segmenters.word_transformer_segmenter\
    import ClassificationSegmenter
from sc_segmenter.segmenters.hierarchical_segmenter\
    import HierarchicalSegmenter
from sc_segmenter.segmenters.bayesian_segmenter\
    import BayesianSegmenter

ALGORITHM_TESTED = "BAYESIAN"
# ALGORITHM_TESTED = "HIERARCHICAL"
# ALGORITHM_TESTED = "STANDARD"
# LANGUAGE = "lat"

for LANGUAGE in ["grc", "lat", "seals"]:
    "======== PARAMETERS OF BENCHMARKS ========"
    FILE_PATH = pathlib.Path(__file__).parent.resolve()
    MODEL_PATH = f"{FILE_PATH}/../train/model/"\
        f"{LANGUAGE}_character-classifier-final"
    AR_MODEL = f"{FILE_PATH}/../train/model/"\
        f"{LANGUAGE}_xglm-564M-finetuned"
    TRAINING_FILE = {
        "lat": f"{FILE_PATH}/../train/data/ground_truth/vulgata_binary.txt",
        "grc": f"{FILE_PATH}/../train/data/ground_truth/sblgnt_binary.txt"
    }
    "========"

    with open(f"{FILE_PATH}/test_datasets/{LANGUAGE}/ms.json") as f:
        benchmark_dataset = json.load(f)

    scores = []
    segmentations = []

    if ALGORITHM_TESTED == "STANDARD":
        segmenter = ClassificationSegmenter(MODEL_PATH)
    elif ALGORITHM_TESTED == "HIERARCHICAL":
        segmenter = HierarchicalSegmenter(
            autoregressive_model_path=AR_MODEL,
            character_model_path=MODEL_PATH,
            beam_width=10)
    elif ALGORITHM_TESTED == "BAYESIAN":
        segmenter = BayesianSegmenter()
        # Need to perform the training "online"
        with open(TRAINING_FILE[LANGUAGE]) as f:
            train_ds = [
                line.split("-")[0].strip() for line in
                f.readlines()
            ]

        random.shuffle(train_ds)
        split_idx = int(0.8 * len(train_ds))
        train_x = train_ds[:split_idx]
        dev_x = train_ds[split_idx:]
        print("---training bayesian segmenter")
        start = time.time()
        segmenter.train(train_x=train_x,
                        dev_x=dev_x)
        end = time.time()
        print(f"--- TRAINING TIME BAYESIAN {LANGUAGE}: {end - start}")

    ix = 0

    for ms, ms_content in benchmark_dataset.items():
        for chapter, chapter_content in ms_content.items():
            for verse, verse_content in chapter_content.items():
                # if ix < 2:
                try:
                    segmented = segmenter.compute_scores(
                        text=verse_content.replace(" ", ""),
                        ground_truth=verse_content.split()
                    )
                    scores.append(segmented)
                    segmentations.append((segmenter.segment(
                        text=verse_content.replace(" ", "")),
                        verse_content)
                    )
                    ix += 1
                    print(f"--- Benchmarking iteration {ix}")
                except Exception:
                    continue
                # else:
                #     break

    # Dump to csv
    scores_df = pd.DataFrame(scores).mean()
    scores_df.to_csv(f"results/{LANGUAGE}_{ALGORITHM_TESTED}.csv")
    print(f"Total scores: {scores_df}")

    # Write down segmentations
    with open(f"results/{LANGUAGE}_{ALGORITHM_TESTED}_segments.txt", "w") as f:
        for line in segmentations:
            f.write(f"{line}\n")
