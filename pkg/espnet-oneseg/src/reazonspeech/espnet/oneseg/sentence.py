import re
import copy
import spacy
from .interface import Caption

__all__ = "build_sentences",

_SPECIALS = {ord(x): "" for x in "…〜＜＞♬:→　"}

def _cleanup(s):
    """Remove special characters to help sentence splitter"""
    s = re.sub(r"^.*≫", "", s)
    s = re.sub(r"^.*＞＞", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"（[^）]*）", "", s)
    s = re.sub(r"\s", "", s)
    return s.translate(_SPECIALS)

def _merge(start, end, sentence):
    caption = copy.copy(start)
    caption.text = sentence
    caption.end_seconds = end.end_seconds
    return caption

def build_sentences(captions):
    """Reformat captions into sentences

    This reorganizes a list of captions based on the sentence boundaries.
    For example, the following captions:

        Caption(start_seconds=10, end_seconds=12, text='輸送機は午前１０時に')
        Caption(start_seconds=12, end_seconds=15, text='離陸しました。')

    will be merged into:

        Caption(start_seconds=10, end_seconds=15, text='輸送機は午前１０時に離陸しました。')

    Args:
        captions (list of Captions): The captions to merge (or split) into sentences.

    Returns:
        A list of `Caption` instances.
    """
    ret = []
    timeline = []
    fulltext = ""

    for caption in captions:
        text = _cleanup(caption.text)
        fulltext += text
        for char in text:
            timeline.append(caption)

    nlp = spacy.load("ja_ginza")
    for sentence in nlp(fulltext).sents:
        sentence = str(sentence)
        start, end = timeline[0], timeline[len(sentence) - 1]
        ret.append(_merge(start, end, sentence))
        timeline = timeline[len(sentence):]
    return ret
