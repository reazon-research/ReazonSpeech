"""USAGE

    reazonspeech-espnet-asr [-h] [--to={vtt,srt,ass,json,tsv}] [-o file] audio

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
    $ reazonspeech-espnet-asr sample.wav

    # Output subtitles in VTT format
    $ reazonspeech-espnet-asr -o sample.vtt sample.webm
"""

import sys
import getopt
import warnings
from .writer import get_writer
from .audio import audio_from_path
from .transcribe import transcribe, load_model

def main():
    outpath = None
    outext = None

    opts, args = getopt.getopt(sys.argv[1:], "ho:", ("help", "output=", "to=",))
    for k, v in opts:
        if k in ("-h", "--help"):
            print(__doc__, file=sys.stderr)
            return
        elif k in ("-o", "--output"):
            outpath = v
        elif k == "--to":
            outext = v

    if not args:
        print("no audio file specified", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 1

    if outpath is not None:
        outfile = open(outpath, 'w')
    else:
        outfile = sys.stdout

    # Suppress warnings from ESPnet
    warnings.simplefilter("ignore")

    # Load audio data and model
    audio = audio_from_path(args[0])
    model = load_model()

    # Perform inference
    ret = transcribe(model, audio)

    with outfile:
        writer = get_writer(outfile, outext)
        writer.write_header()
        for ts in ret.segments:
            writer.write(ts)

if __name__ == "__main__":
    sys.exit(main())
