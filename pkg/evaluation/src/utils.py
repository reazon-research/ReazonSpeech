import re
from typing import TypedDict

import editdistance
import num2words


class CERResult(TypedDict):
    cer: float
    distance: int
    length: int


PUNCTUATIONS = {ord(x): "" for x in "、。「」『』，,？！!!?!?"}
ZENKAKU = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９"
HANKAKU = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ZEN2HAN = str.maketrans(ZENKAKU, HANKAKU)


def normalize(s: str) -> str:
    s = s.translate(PUNCTUATIONS).translate(ZEN2HAN)
    conv = lambda m: num2words.num2words(m.group(0), lang="ja")
    return re.sub(r"\d+\.?\d*", conv, s)


def calculate_cer(reference: str, prediction: str) -> CERResult:
    reference = normalize(reference)
    prediction = normalize(prediction)
    distance = editdistance.eval(reference, prediction)
    return CERResult(cer=distance / len(reference), distance=distance, length=len(reference))
