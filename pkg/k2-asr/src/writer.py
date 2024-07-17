import os
import json

class VTTWriter:
    """WebVTT (Web Video Text Tracks) is a standard caption format defined
    by W3C in 2010. It's supported by major browsers, and can be used with
    HTML5.

    See also: https://www.w3.org/TR/webvtt1/
    """

    ext = 'vtt'

    def __init__(self, fp):
        self.fp = fp

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i.%03i" % (h, m, s, ms)

    def write_header(self):
        self.fp.write("WEBVTT\n\n")

    def write(self, segment):
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write("%s --> %s\n%s\n\n" % (start, end, segment.text))

class SRTWriter:
    """SRT is a subtitle format commonly used by desktop programs. It was
    originally developed by a Windows program SubRip.

    See also: https://www.matroska.org/technical/subtitles.html#srt-subtitles
    """

    ext = 'srt'

    def __init__(self, fp):
        self.fp = fp
        self.index = 0

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i,%03i" % (h, m, s, ms)

    def write_header(self):
        return

    def write(self, segment):
        self.index += 1
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write("%i\n%s --> %s\n%s\n\n" % (self.index, start, end, segment.text))

class ASSWriter:
    """ASS is another common format among desktop apps. It was developed
    by Advanced Sub Station Alpha, and can be used to burn subtitles
    using libass.

    See also: https://github.com/libass/libass
    See also: https://trac.ffmpeg.org/wiki/HowToBurnSubtitlesIntoVideo
    """

    ext = 'ass'

    def __init__(self, fp):
        self.fp = fp

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return "%i:%02i:%02i.%02i" % (h, m, s, cs)

    def write_header(self):
        self.fp.write("""\
[Script Info]
ScriptType: v4.00+
Collisions: Normal
Timer: 100.0000

[V4+ Styles]
Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
""")

    def write(self, segment):
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write("Dialogue: 0,%s,%s,Default,,0,0,0,,%s\n" % (start, end, segment.text))

class JSONWriter:
    """JSON (JavaScript Object Notation) writer"""

    ext = 'json'

    def __init__(self, fp):
        self.fp = fp

    def write_header(self):
        return

    def write(self, ts):
        line = json.dumps({
            "start_seconds": round(ts.start_seconds, 3),
            "end_seconds": round(ts.end_seconds, 3),
            "text": ts.text
        }, ensure_ascii=False)
        self.fp.write(line + "\n")

class TSVWriter:
    """TSV (Tab-separated values) writer"""

    ext = 'tsv'

    def __init__(self, fp):
        self.fp = fp

    def write_header(self):
        self.fp.write("start_seconds\tend_seconds\ttext\n")

    def write(self, segment):
        self.fp.write("%.3f\t%.3f\t%s\n" % (segment.start_seconds, segment.end_seconds, segment.text))

class TextWriter:

    ext = 'txt'

    def __init__(self, fp):
        self.fp = fp

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i.%03i" % (h, m, s, ms)

    def write_header(self):
        return

    def write(self, segment):
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write("[%s --> %s] %s\n" % (start, end, segment.text))

def get_writer(fp, ext=None):
    if ext is None:
        name = getattr(fp, 'name', '')
        ext = os.path.splitext(name)[-1]

    for cls in (VTTWriter, SRTWriter, ASSWriter, JSONWriter, TSVWriter):
        if cls.ext == ext:
            return cls(fp)

    return TextWriter(fp)
