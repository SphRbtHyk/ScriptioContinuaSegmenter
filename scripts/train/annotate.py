"""Perform annotation of the ground truth data to perform the training.
"""
from pathlib import Path
import json
from sc_segmenter.training.ground_truth_annotator import annotate_dict

# Files to annotate
FILES_TO_ANNOTATE = ["data/raw_data/sblgnt.json",
                     "data/raw_data/vulgata.json",
                     "data/raw_data/seals.json"]

# Methods of annotation
ANNOTATION_METHOD = ["binary", "quadrimodal"]

# Iterate over the files to annotate + annotation methods
for file in FILES_TO_ANNOTATE:
    for method in ANNOTATION_METHOD:
        # Open file
        with open(file, "r") as f:
            content = json.load(f)
        annotated_data = annotate_dict(content,
                                       annotation_scheme=method)
        with open(f"data/ground_truth/{Path(file).stem}_{method}.txt", "w") as f:
            for line in annotated_data:
                f.write(f"{line[0]} - {line[1]} - {line[2]}\n")
