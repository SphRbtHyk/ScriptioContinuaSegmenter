"""Provide a description of every training dataset available in the study.
"""
import json

DATASETS = ["sblgnt", "vulgata", "seals"]


def unpack_dict(input_dict):
    """Unpack nested dicts that contain the ground truth/
    test datasets."""
    content_training_file = ""
    for book, book_content in input_dict.items():
        for chapter, chapter_content in book_content.items():
            for verse, verse_content in chapter_content.items():
                content_training_file += verse_content + " "
    return content_training_file.lower()


def count_seals(input_dict):
    """Count the number of total seals in the dataset."""
    seal_names = []
    for _, book_content in input_dict.items():
        for _, chapter_content in book_content.items():
            for verse, _ in chapter_content.items():
                seal_names.append(verse)
    names = []
    for name in seal_names:
        names.append(name.split("_")[1])
    return len(set(names))


for dataset in DATASETS:
    print(f"==== DATASET {dataset}")
    with open(f"data/raw_data/{dataset}.json") as f:
        training_file = json.load(f)

    if dataset == "sblgnt":
        lg = "grc"
    elif dataset == "vulgata":
        lg = "lat"
    elif dataset == "seals":
        lg = "seals"

    train_data = unpack_dict(training_file).split()

    with open(f"../benchmark/test_datasets/{lg}/ms.json") as f:
        test_file = json.load(f)

    test_data = unpack_dict(test_file).split()

    print(f"Number words train : {len(train_data)}")
    print(f"Number words test : {len(test_data)}")

    overlaps = set(train_data).intersection(set(test_data))
    print("Overlap words train/test:"
          f"{round(len(overlaps)/len(set(train_data))*100, 2)}")
    
    if lg == "seals":
        print(f"Number of instances of seals: {count_seals(test_file)}")

