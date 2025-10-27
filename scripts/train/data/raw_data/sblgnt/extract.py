"""Extract in JSON format the SBLGNT dataset

!!  This current implementation strips the diacritics whenever
outputting the data."""
from pathlib import Path
import unicodedata as ud


def remove_diacritics(text: str) -> str:
    """Given a Greek text containing diacritics, remove them.

    Args:
        text (str): The text to strip the diacritics from.

    Returns:
        str: The cleaned up text;
    """
    return ''.join(c for c in ud.normalize('NFD', text)
                   if ud.category(c) != 'Mn')


def clean_up(text: str, symbols: list[str] = ["'", "’"]) -> str:
    """Clean up the input text by removing unwanted symbols.

    Args:
        text (str): Text to clean up.
        symbols (list[str], optional): The list of symbols to strip from
            the dataset. Defaults to [''].

    Returns:
        str: The cleaned up text.
    """
    return ''.join(c for c in text if c not in symbols)


def load_sbglnt_verses(input_folder: str = "sblgnt/"):
    """
    Load the SBLGNT JSON files into a list of Python dictionary on
    a per verse basis using encoding
    adapted to the gnt. The only loaded text is the lemmed and stemmed
    words, as only these will be considered whenever performing the
    clustering.

    Args:
        input_folder (str): Folder to find the data in
    """
    json_output = {}
    for file in Path(Path(__file__).resolve().parent / input_folder).glob("*.txt"):
        book = file.name.split("-")[1]
        split_text = file.read_text(encoding="utf8").split("\n")
        text = {}
        for texts in split_text:
            parsed_text = texts.split(" ")
            if parsed_text[0]:
                chapter_ix = str(int(parsed_text[0][2:4]))
                verse_ix = str(int(parsed_text[0][4:6]))
                # Check if chapter already exists
                try:
                    text[chapter_ix]
                # If it doesn't, fill it with an empty dictionary
                except KeyError:
                    text[chapter_ix] = {}
                # Else, fill it with content
                try:
                    text[chapter_ix][verse_ix] += clean_up(remove_diacritics(
                        parsed_text[4])) + " "
                except KeyError:
                    text[chapter_ix][verse_ix] = \
                        clean_up(remove_diacritics(parsed_text[4])) + " "
        json_output[book] = text
    return json_output


if __name__ == "__main__":
    import json
    with open("../sblgnt.json", "w") as f:
        json.dump(load_sbglnt_verses(), f, ensure_ascii=False)
