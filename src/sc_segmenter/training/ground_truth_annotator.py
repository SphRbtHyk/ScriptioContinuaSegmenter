"""Utility functions for annotating the ground truth.
"""


def binary_annotation(input_string: str) -> list[str]:
    """Perform the binary annotation of a string, and returns
    the corresponding labels for the training of the segmentation
    model:
        - If a character is followed by a white space, then add the "E"
        (End of Word) tag.
        - Otherwise, add a "I" (Inside) tag.

    Args:
        input_string (str): The string to use as ground truth to
            build the labels.

    Returns:
        annotations (list[str]): The labels to use for training
            the model.
    """
    annotations = []
    for word in input_string.split():
        annotations.extend(["I"] *
                           (len(word) - 1) + ["E"])
    return annotations


def quadrimodal_annotation(input_string: str) -> list[str]:
    """Perform a quadrimodal annotation of a string, and returns
    the corresponding labels for the training of the segmentation
    model:
        - B if the character is the beginning of a word.
        - I if the character is inside the word.
        - E if the character is at the end of the word.
        - S if the character is a single one.

    Args:
        input_string (str): The string to use as ground truth to
            build the labels.

    Returns:
        annotations (list[str]): The labels to use for training
            the model.
    """
    annotations = []
    for word in input_string.split():
        # Check if the length of the word is 1
        if len(word) == 1:
            annotations.extend("S")
        else:
            annotations.extend(["B"] + ["I"] *
                               (len(word) - 2) + ["E"])
    return annotations


def annotate_dict(input_dict: dict[str, dict[str, dict[str, str]]],
                  annotation_scheme: str = "binary") -> \
        list[tuple[str, list[str], str]]:
    """Perform the annotation of an input dictionary.

    Args:
        input_dict (dict[str, dict[str, dict[str, str]]]): The input
            dictionary to annotate. It must be a nested dictionary,
            of the form book_name/chap/verse.

    Returns:
        list[tuple[str, list[str]]]: The list of the strings, their
            annotations as a list of tuple and the corresponding ground truth.
            The spaces will be removed as they are indicated in the label.
    """
    if annotation_scheme == "binary":
        annotation_fun = binary_annotation
    elif annotation_scheme == "quadrimodal":
        annotation_fun = quadrimodal_annotation
    else:
        raise ValueError("Unknown required annotation mode"
                         f"{annotation_scheme}")
    annotations = []
    for _, book_content in input_dict.items():
        for _, chapter_content in book_content.items():
            for _, verse_content in chapter_content.items():
                annotations.append((
                    verse_content.replace(" ", "").lower(),
                    annotation_fun(verse_content),
                    verse_content)
                )
    return annotations
