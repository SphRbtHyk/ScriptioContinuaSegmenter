#!/usr/bin/env python3
"""Run the benchmarking process for specific parameters."""

import argparse
import random
import pathlib
import json
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sc_segmenter.segmenters.word_transformer_segmenter import ClassificationSegmenter
from sc_segmenter.segmenters.hierarchical_segmenter import HierarchicalSegmenter
from sc_segmenter.segmenters.bayesian_segmenter import BayesianSegmenter


def unpack_dict(input_dict):
    """Unpack nested dicts that contain the ground truth/
    test datasets."""
    content_training_file = []
    for _, book_content in input_dict.items():
        for _, chapter_content in book_content.items():
            for _, verse_content in chapter_content.items():
                if verse_content.replace(" ", ""):
                    content_training_file.append(verse_content.replace(" ", ""))

    # Debug info
    logger.info(f"Unpacked {len(content_training_file)} training examples")

    return content_training_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run segmentation benchmark")
    parser.add_argument("--language", type=str, required=True,
                        choices=["grc", "lat", "seals"])
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["STANDARD", "HIERARCHICAL", "BAYESIAN"])
    parser.add_argument("--annotation", type=str,
                        choices=["binary", "quadri"], default="binary")
    parser.add_argument("--beam_width", type=int, default=1)
    return parser.parse_args()


TRAIN_ON_REAL_DATA = True


def main():
    args = parse_arguments()

    logger.info(
        f"Starting benchmark: {args.language}, {args.algorithm}, {args.annotation}, beam_width={args.beam_width}")

    FILE_PATH = pathlib.Path(__file__).parent.resolve()
    MODEL_PATH = f"{FILE_PATH}/../train/model/{args.language}_character-classifier-{args.annotation}-final"
    AR_MODEL = f"{FILE_PATH}/../train/model/{args.language}_xglm-564M-finetuned"
    TRAINING_FILE = {
        "lat": f"{FILE_PATH}/../train/data/ground_truth/vulgata_binary.txt",
        "grc": f"{FILE_PATH}/../train/data/ground_truth/sblgnt_binary.txt",
        "seals": f"{FILE_PATH}/../train/data/ground_truth/seals_binary.txt",
    }

    with open(f"{FILE_PATH}/test_datasets/{args.language}/ms.json") as f:
        benchmark_dataset = json.load(f)

    scores = []
    segmentations = []

    # Initialize segmenter based on algorithm
    if args.algorithm == "STANDARD":
        segmenter = ClassificationSegmenter(MODEL_PATH)
    elif args.algorithm == "HIERARCHICAL":
        segmenter = HierarchicalSegmenter(
            autoregressive_model_path=AR_MODEL,
            character_model_path=MODEL_PATH,
            beam_width=args.beam_width
        )
    elif args.algorithm == "BAYESIAN":
        segmenter = BayesianSegmenter()
        if TRAIN_ON_REAL_DATA:
            train_ds = unpack_dict(benchmark_dataset)
        else:
            with open(TRAINING_FILE[args.language]) as f:
                train_ds = [line.split("-")[0].strip() for line in f.readlines()]
        random.shuffle(train_ds)
        split_idx = int(0.8 * len(train_ds))

        logger.info("Training Bayesian segmenter...")
        segmenter.train(
            train_x=train_ds[:split_idx],
            dev_x=train_ds[split_idx:])

    # Count total number of verses for progress bar
    total_verses = 0
    for ms, ms_content in benchmark_dataset.items():
        for chapter, chapter_content in ms_content.items():
            total_verses += len(chapter_content)

    logger.info(f"Processing {total_verses} verses...")

    # Process all verses with progress bar
    ix = 0
    errors = 0

    with tqdm(total=total_verses, desc=f"{args.algorithm} {args.language}", unit="verse") as pbar:
        for ms, ms_content in benchmark_dataset.items():
            for chapter, chapter_content in ms_content.items():
                for verse, verse_content in chapter_content.items():
                    try:
                        segmented = segmenter.compute_scores(
                            text=verse_content.replace(" ", ""),
                            ground_truth=verse_content.split()
                        )
                        scores.append(segmented)
                        segmentations.append((
                            segmenter.segment(
                                text=verse_content.replace(" ", "")),
                            verse_content
                        ))
                        ix += 1

                        # Update progress bar every verse
                        pbar.update(1)
                        pbar.set_postfix({
                            'processed': ix,
                            'errors': errors,
                            'current': f"{ms}/{chapter}/{verse}"
                        })

                    except Exception as e:
                        errors += 1
                        logger.error(f"Problem at {ms}/{chapter}/{verse}: {e}")
                        pbar.set_postfix({
                            'processed': ix,
                            'errors': errors,
                            'current': f"ERROR:{ms}/{chapter}/{verse}"
                        })
                        continue

    # Save results
    logger.info(f"Completed: {ix} verses processed, {errors} errors")

    if scores:  # Only save if we have results
        scores_df = pd.DataFrame(scores).mean()
        output_file = f"results/{args.language}_{args.annotation}_{args.algorithm}_{args.beam_width}.csv"
        scores_df.to_csv(output_file)
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Total scores: {scores_df}")

        # Save segmentations
        with open(f"results/{args.language}_{args.algorithm}_{args.annotation}_{args.beam_width}_segments.txt", "w") as f:
            for line in segmentations:
                f.write(f"{line}\n")
        logger.info(f"Segmentations saved for {len(segmentations)} verses")
    else:
        logger.error("No successful segmentations to save!")


if __name__ == "__main__":
    main()
