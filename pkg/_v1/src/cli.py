"""USAGE

    reazonspeech [-h] [--to={vtt,srt,ass,json,tsv}] [-o file] audio

OPTIONS

    audio
        Audio file to transcribe. It can be in any format as long
        as librosa.load() can read.

    -h, --help
        Print this help message.

    --to={vtt,srt,ass,json,tsv}
        Output format for transcription

    -o file, --output=file
        File to write transcription

EXAMPLES

    # Transcribe audio file
    $ reazonspeech sample.wav

    # Output subtitles in VTT format
    $ reazonspeech -o sample.vtt sample.webm
"""

import os
import sys
import json
import tqdm
import getopt
import warnings
import librosa
from .transcribe import transcribe, load_default_model, TranscribeConfig

#==============
# Format Writer
#==============

class VTTWriter:
    """WebVTT (Web Video Text Tracks) is a standard caption format defined
    by W3C in 2010. It's supported by major browsers, and can be used with
    HTML5.

    See also: https://www.w3.org/TR/webvtt1/
    """

    ext = 'vtt'

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i.%03i" % (h, m, s, ms)

    def header(self, file):
        file.write("WEBVTT\n\n")

    def caption(self, file, caption):
        start = self._format_time(caption.start_seconds)
        end = self._format_time(caption.end_seconds)
        file.write("%s --> %s\n%s\n\n" % (start, end, caption.text))

class SRTWriter:
    """SRT is a subtitle format commonly used by desktop programs. It was
    originally developed by a Windows program SubRip.

    See also: https://www.matroska.org/technical/subtitles.html#srt-subtitles
    """

    ext = 'srt'

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i,%03i" % (h, m, s, ms)

    def header(self, file):
        self.index = 0

    def caption(self, file, caption):
        self.index += 1
        start = self._format_time(caption.start_seconds)
        end = self._format_time(caption.end_seconds)
        file.write("%i\n%s --> %s\n%s\n\n" % (self.index, start, end, caption.text))

class ASSWriter:
    """ASS is another common format among desktop apps. It was developed
    by Advanced Sub Station Alpha, and can be used to burn subtitles
    using libass.

    See also: https://github.com/libass/libass
    See also: https://trac.ffmpeg.org/wiki/HowToBurnSubtitlesIntoVideo
    """

    ext = 'ass'

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return "%i:%02i:%02i.%02i" % (h, m, s, cs)

    def header(self, file):
        file.write("""\
[Script Info]
ScriptType: v4.00+
Collisions: Normal
Timer: 100.0000

[V4+ Styles]
Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
""")

    def caption(self, file, caption):
        start = self._format_time(caption.start_seconds)
        end = self._format_time(caption.end_seconds)
        file.write("Dialogue: 0,%s,%s,Default,,0,0,0,,%s\n" % (start, end, caption.text))

class JSONWriter:

    ext = 'json'

    def header(self, file):
        return

    def caption(self, file, caption):
        line = json.dumps({
            "start_seconds": round(caption.start_seconds, 3),
            "end_seconds": round(caption.end_seconds, 3),
            "text": caption.text
        }, ensure_ascii=False)
        file.write("%s\n" % line)

class TSVWriter:

    ext = 'tsv'

    def header(self, file):
        file.write("start_seconds\tend_seconds\ttext\n")

    def caption(self, file, caption):
        file.write("%.3f\t%.3f\t%s\n" % (caption.start_seconds, caption.end_seconds, caption.text))


#======
# Main
#======

def get_writer(ext):
    for cls in (VTTWriter, SRTWriter, ASSWriter, JSONWriter, TSVWriter):
        if cls.ext == ext:
            return cls()

def get_default_writer(file):
    # Guess an appropriate format from the file name
    ext = os.path.splitext(file.name)[1][1:]
    writer = get_writer(ext)
    if writer is not None:
        return writer

    # Default to JSON
    return JSONWriter()

def show_usage(file):
    print(__doc__, file=file)

def main():
    outpath = None
    outext = None

    opts, args = getopt.getopt(sys.argv[1:], "ho:", ("help", "output=", "to=",))
    for k, v in opts:
        if k in ("-h", "--help"):
            show_usage(sys.stdout)
            return
        elif k in ("-o", "--output"):
            outpath = v
        elif k == "--to":
            outext = v

    if outpath is not None:
        outfile = open(outpath, 'w')
    else:
        outfile = sys.stdout

    if outext is not None:
        writer = get_writer(outext)
    else:
        writer = get_default_writer(outfile)

    if not writer:
        print("unknown output format", file=sys.stderr)
        show_usage(sys.stderr)
        return 1

    if not args:
        print("no audio file specified", file=sys.stderr)
        show_usage(sys.stderr)
        return 1

    warnings.simplefilter("ignore")

    # Load audio and ASR model
    config = TranscribeConfig()
    audio = librosa.load(args[0], sr=config.samplerate)[0]
    speech2text = load_default_model()

    # Prepare progress bar
    pbar = tqdm.tqdm(total=int(len(audio) / config.samplerate),
                     unit='s', desc='Transcribing',
                     disable=outfile.isatty())

    # Transcribe audio
    writer.header(outfile)

    for caption in transcribe(audio, speech2text):
        writer.caption(outfile, caption)
        pbar.n = round(caption.end_seconds)
        pbar.refresh()

    outfile.close()
    pbar.close()

if __name__ == "__main__":
    sys.exit(main())
