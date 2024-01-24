from typing import TypedDict, Optional, List


class SingleWordSegment(TypedDict):
    """
    A single word of a speech.
    """
    word: str
    start_seconds: float
    end_seconds: float
    score: float

class SingleCharSegment(TypedDict):
    """
    A single char of a speech.
    """
    char: str
    start_seconds: float
    end_seconds: float
    score: float


class SingleSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech.
    """

    start_seconds: float
    end_seconds: float
    text: str


class SingleAlignedSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """

    start_seconds: float
    end_seconds: float
    text: str
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]
