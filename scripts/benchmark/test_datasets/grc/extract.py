"""Download the data from the NTVMR and outputs them as a JSON file.
"""
import os
import glob
from pathlib import Path
import unicodedata
from xml.etree import ElementTree
import httpx
import time
from loguru import logger
import re
import json


DATASETS_FOLDER = os.path.dirname(__file__)


def parse_igntp_names(name: str):
    """
    Parse a name like 'NT_GRC_1_1Cor' to retrieve the name of the
    book and the name of the manuscript.
    """
    parts = name.split("_")
    manuscript = parts[2]
    book = parts[3]
    return manuscript, book


def parse_chapter(chap_str: str):
    """Parse the chapter number.
    """
    m1 = re.match(r"^(\d?[A-Za-z]+)\.(\d+)\.(\d+)$", chap_str)
    logger.info(chap_str)

    if "incip" in chap_str.lower():
        return "Incipit"
    if "subscrip" in chap_str.lower():
        return "Subscriptio"
    if "inscript" in chap_str.lower():
        return "Inscriptio"
    if m1:
        chapter = int(m1.group(2))
    else:
        chapter = chap_str.split("K")[1]
    return chapter


def parse_verse(verse_str: str):
    """Parse the verse number.
    """
    if "subscrip" in verse_str.lower():
        return "Subscriptio"
    if "inscript" in verse_str.lower():
        return "Inscriptio"
    # Check which format is the specification
    m1 = re.match(r"^(\d?[A-Za-z]+)\.(\d+)\.(\d+)$", verse_str)
    if m1:
        verse = int(m1.group(3))
    else:
        verse = verse_str.split("V")[1]
    return verse


def parse_manuscript(response_str: str,
                     book_id: str = "B20",
                     variant: str = "orig",
                     use_reconstructed: bool = True):
    """Given the a NTVMR request response, parse the manuscript into
    a Python dictioinary.
    """
    et = ElementTree.fromstring(response_str.replace(
        '<lb break="no"/>', "").replace('<lb/>', ''))
    title = et.find(".//{http://www.tei-c.org/ns/1.0}title").attrib["n"]
    flat_text = {}

    for elem in et.iter():
        if elem.tag == "{http://www.tei-c.org/ns/1.0}div":
            if elem.attrib["type"] == "book":
                book = elem.attrib["n"]
                if book != book_id:
                    break
            if elem.attrib["type"] == "chapter":
                chapter = parse_chapter(elem.attrib["n"])
                if chapter not in flat_text:
                    flat_text[chapter] = {}
            if elem.attrib["type"] == "incipit":
                chapter = parse_chapter(elem.attrib["n"])
                if chapter not in flat_text:
                    flat_text[chapter] = {}

        if elem.tag == "{http://www.tei-c.org/ns/1.0}ab":
            if elem.attrib.get("n"):
                verse = parse_verse(elem.attrib["n"])
            try:
                verse
            except UnboundLocalError:
                verse = "0"
            if verse not in flat_text[chapter]:
                flat_text[chapter][verse] = ""

            for subelem in list(elem):
                if subelem.tag == "{http://www.tei-c.org/ns/1.0}w":
                    # Nested words structure (nominem sacrum, abbreviation, etc)
                    if not subelem.text:
                        for subsubelem in subelem.iter():
                            if subsubelem.text:
                                flat_text[chapter][verse] += subsubelem.text
                            if subsubelem.tail:
                                flat_text[chapter][verse] += subsubelem.tail
                        flat_text[chapter][verse] += " "
                    else:
                        # Get all texts, including abbreviation and unclear texts if
                        # use reconstructed is enabled
                        if not use_reconstructed:
                            flat_text[chapter][verse] += subelem.text + " "
                        else:
                            flat_text[chapter][verse] += ''.join(
                                subelem.itertext()) + " "
                if subelem.tag == "{http://www.tei-c.org/ns/1.0}app":
                    for subsubelem in subelem.iter():
                        if subsubelem.tag == "{http://www.tei-c.org/ns/1.0}rdg":
                            try:
                                if subsubelem.attrib["type"] == variant:
                                    for subsubsubelem in subsubelem.iter():
                                        if subsubsubelem.tag == "{http://www.tei-c.org/ns/1.0}w":
                                            if subsubsubelem.text:
                                                flat_text[chapter][verse] += subsubsubelem.text + " "
                                            # Check if again a nested structure
                                            else:
                                                for subsubsubsubelem in subsubsubelem.iter():
                                                    if subsubsubsubelem.text:
                                                        flat_text[chapter][verse] += subsubsubsubelem.text + " "
                            except KeyError:
                                for subsubsubelem in subsubelem.iter():
                                    if subsubsubelem.tag == "{http://www.tei-c.org/ns/1.0}w":
                                        if subsubsubelem.text:
                                            flat_text[chapter][verse] += subsubsubelem.text + " "
                                        # Check if again a nested structure
                                        else:
                                            for subsubsubsubelem in subsubsubelem.iter():
                                                if subsubsubsubelem.text:
                                                    flat_text[chapter][verse] += subsubsubsubelem.text + " "
                            flat_text[chapter][verse] += " "
    flat_text = {chapter: {verse: flat_text[chapter][verse]
                           for verse in flat_text[chapter]} for chapter in flat_text}
    # Remove extra spaces
    flat_text = {chapter: {verse: " ".join(flat_text[chapter][verse].split(
    )) for verse in flat_text[chapter]} for chapter in flat_text}
    return title, flat_text


def extract_text(data):
    """
    Recursively extracts the '#text' and 'w' content from a nested dictionary or list structure, 
    including specific handling for 'app' and 'rdg' fields with type 'orig'.

    Parameters:
        data (dict or list): The input structure to extract text from.

    Returns:
        list: A list of extracted text strings.
    """
    texts = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "#text":
                texts.append(value)
            elif key == "w":
                if isinstance(value, dict) and "unclear" in value:
                    texts.append(value["unclear"])
                else:
                    texts.append(value)
            else:
                texts.extend(extract_text(value))
    elif isinstance(data, list):
        for item in data:
            texts.extend(extract_text(item))
    return texts


def extract_original_w(data):
    """
    Extracts the 'w' field from 'rdg' elements where '@type' is 'orig'.
    If 'w' is a dictionary containing '#text', extracts the '#text' value.

    Parameters:
        data (dict or list): The input structure containing 'rdg' elements or '#text' fields.

    Returns:
        list: A list of extracted text ('w' or '#text') values.
    """
    results = []

    if isinstance(data, dict):
        # Handle 'rdg' elements
        if "rdg" in data:
            for rdg_entry in data["rdg"]:
                if rdg_entry.get("@type") == "orig" and "w" in rdg_entry:
                    w_value = rdg_entry["w"]
                    # If 'w' is a dictionary, extract '#text'
                    if isinstance(w_value, dict) and "#text" in w_value:
                        results.append(w_value["#text"])
                    elif isinstance(w_value, str):
                        results.append(w_value)
        else:
            # Recursively process dictionary values
            for value in data.values():
                results.extend(extract_original_w(value))
    elif isinstance(data, list):
        # Recursively process list elements
        for item in data:
            results.extend(extract_original_w(item))

    return results


def extract_manuscript_content(parsed_xml):
    manuscript_content = {}
    for content in parsed_xml["TEI"]["text"]["body"]["div"]["div"]:
        if type(content) == str:
            continue
        if not "ab" in content:
            continue
        if not type(content["ab"]) == list:
            continue
        for verse_content in content["ab"]:
            if not verse_content:
                continue
            if not "@n" in verse_content:
                continue
            chapter, verse = parse_chapter(
                verse_content["@n"]), parse_verse(verse_content["@n"])
            if not chapter in manuscript_content:
                manuscript_content[chapter] = {}
            if not verse in manuscript_content[chapter]:
                manuscript_content[chapter][verse] = ""
            if "w" in verse_content:

                for value in verse_content["w"]:
                    if type(value) == dict:

                        # if not "supplied" in value:
                        if "supplied" in value:
                            manuscript_content[chapter][verse] += "SUPPLIED"
                        if not type(extract_text(value)) == dict and not type(extract_text(value)) == list:
                            manuscript_content[chapter][verse] += "".join(
                                extract_text(value)[::-1]).replace("\n", "") + " "
                    else:
                        manuscript_content[chapter][verse] += value.replace(
                            "\n", "") + " "
            if "app" in verse_content:
                manuscript_content[chapter][verse] += " ".join(
                    extract_original_w(verse_content["app"])).replace("\n", "") + " "
    return manuscript_content


def remove_control_characters(s: str):
    """
    Remove control characters from a string.
    """
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


def get_manuscripts_id(uncials_range=(1, 326),
                       papyri_range=(1, 135),
                       miniscules_range=(1, 500)):
    """Get the list of manuscripts to retrieve from the NTVMR.
    """
    if uncials_range:
        oncials = [f"2{str(i).zfill(4)}" for i in range(*uncials_range)]
    else:
        oncials = []
    if papyri_range:
        papyrus = [f"1{str(i).zfill(4)}" for i in range(*papyri_range)]
    else:
        papyrus = []
    if miniscules_range:
        miniscules = [f"3{str(i).zfill(4)}" for i in range(*miniscules_range)]
    else:
        miniscules = []
    return oncials + papyrus + miniscules


def retrieve_manuscript_content(manuscript_id: str,
                                tradition_name: str,
                                timer: int = 60):
    """Retrieve the content of a manuscript from the NTVMR.
    """
    base_url = "https://ntvmr.uni-muenster.de/community/vmr/api/transcript/get/"
    request = f"?docID={manuscript_id}&indexContent={tradition_name}&format=teiraw"
    logger.info(f"Submitting request to retrieve manuscript {manuscript_id}")
    try:
        response = httpx.get(base_url + request, timeout=350)
        print(base_url + request)
    except Exception:
        logger.info(f"Timeout when downloading manuscript {manuscript_id}")
        return None
    if response.status_code == 200:
        if "No Transcription Available" in response.text:
            logger.info(f"No transcription available for manuscript"
                        f"{manuscript_id}")
            return None
        logger.info(f"Downloaded manuscript {manuscript_id}")
        # Avoid reset by peer errors by sleeping at the end of the request
        time.sleep(timer)
        return remove_control_characters(response.text)
    else:
        logger.error(f"No data available for manuscript {manuscript_id}")
        # Avoid reset by peer errors by sleeping at the end of the request
        time.sleep(timer)
        return None


def get_manuscripts(tradition_name: str,
                    uncials_range=None,
                    papyri_range=None,
                    miniscules_range=None,
                    manuscripts_list=None,
                    timer: int = 30):
    """Retrieve the manuscripts from the NTVMR.
    """
    if not manuscripts_list:
        manuscripts_list = get_manuscripts_id(
            uncials_range=uncials_range,
            papyri_range=papyri_range,
            miniscules_range=miniscules_range)
    responses = []
    ix = 0
    for manuscript in manuscripts_list:
        response = retrieve_manuscript_content(
            manuscript, tradition_name, timer)
        if response:
            responses.append(response)
            ix += 1
    logger.info(f"Retrieved {ix} manuscripts")
    return responses


if __name__ == "__main__":
    MANUSCRIPT_LIST = [
        '20001',
        '20002',
        '20003',
        '20005']
    #    '30001',
    #    '30013',
    #    '30018',
    #    '30022',
    #    '30033',
    #    '30079',
    # #    "30069",
    #    '30343',
    #    '100P4',
    #    '10P42']
    BOOK = "B03"
    TRADITION_NAME = "Luke"
    manuscript_parsed = {}

    manuscript_content = get_manuscripts(
        tradition_name=TRADITION_NAME,
        manuscripts_list=MANUSCRIPT_LIST,
        timer=30
    )
    # Iterate over every manuscript and dump them
    # for manuscript_name, manuscript in zip(MANUSCRIPT_LIST, manuscript_content):
    #     logger.info(f"Dumping manuscript {manuscript_name}")
    #     with open(f"{DATASETS_FOLDER}/{manuscript_name}.xml", "w") as f:
    #         f.write(manuscript)

    for manuscript_name, manuscript in zip(MANUSCRIPT_LIST, manuscript_content):
        logger.info(f"Processing manuscript {manuscript_name}")
        try:
            title, flat_text = parse_manuscript(manuscript, book_id=BOOK)
        except ElementTree.ParseError:
            logger.warning(
                f"Could not process manuscript {manuscript_name} for book {BOOK}")
        # Store content of the manuscript and iteratively overwrite file in case of failure
        manuscript_parsed[manuscript_name] = flat_text
        with open(f"{DATASETS_FOLDER}/manuscripts.json", "w", encoding='utf8') as f:
            json.dump(
                manuscript_parsed,
                f,
                ensure_ascii=False
            )
    # logger.info("Done processing manuscripts")
    # for file in glob.glob(f'{DATASETS_FOLDER}/*.xml'):
    #     manuscript_name = Path(file).stem
    #     logger.info(f"Processing manuscript {manuscript_name}")
    #     with open(file, "r") as f:
    #         manuscript = f.read()
    #     # try:
    #     #
    #     title, flat_text = parse_manuscript(manuscript, book_id=BOOK)
    #     # except ElementTree.ParseError:
    #     #     logger.warning(
    #     #         f"Could not process manuscript {manuscript_name} for book {BOOK}")
    #     # Store content of the manuscript and iteratively overwrite file in case of failure
    #     manuscript_parsed[manuscript_name] = flat_text
    #     with open(f"{DATASETS_FOLDER}/all_manuscripts.json", "w", encoding='utf8') as f:
    #         json.dump(
    #             manuscript_parsed,
    #             f,
    #             ensure_ascii=False
    #         )
    logger.info("Done processing manuscripts")
