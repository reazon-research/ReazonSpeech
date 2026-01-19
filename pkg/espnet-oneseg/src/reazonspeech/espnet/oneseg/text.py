from . import data
from importlib.resources import open_text

__all__ = "cer", "normalize"

#
# Constants

with open_text(data, 'symbol.txt') as fp:
    _SPECIALS = {ord(c.rstrip("\n")): "" for c in fp}

_HAN2ZEN = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９")

def _editdist(s, t):
    """A simple implementation of Wagner-Fischer algorithm"""
    n = len(s)
    m = len(t)
    buf = list(range(m + 1))

    for i in range(1, n + 1):
        tmp = buf[0]
        buf[0] = i

        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                tmp, buf[j] = buf[j], tmp
            else:
                val = min(buf[j], buf[j - 1], tmp) + 1
                tmp, buf[j] = buf[j], val
    return buf[m]

def normalize(text):
    """Trim non-phonatory symbols in the text

    Args:
        text(str): A string to process

    Returns:
        A normalized string
    """
    return text.translate(_SPECIALS).translate(_HAN2ZEN)

def cer(text, asr):
    """Compute CER (Character Error Rate).

    Args:
        text(str): The correct label text
        asr(str): The recognized speech

    Returns:
        The CER between text and asr
    """
    text = normalize(text)
    asr = normalize(asr)
    return _editdist(text, asr) / len(text)
