# This file is a derivative work of "assdumper.cc" by Kohei Suzuki.
# I added a few enhancements to the original code:
#
#  * Port the algorithm from C++ to Python.
#  * Implement new PCR/PTR parsers.
#  * Add relative timestamp support (= compute timestamps from the
#    beginning of the stream).
# ----
# Copyright (c) 2014 Kohei Suzuki
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from .encoding import decode_cprofile
from .interface import Caption
from dataclasses import dataclass, field

__all__ = "get_captions",

# ---------
# Constants
# ---------

_SYNC_BYTE = 0x47
_TABLE_PAT = 0x00
_TABLE_PMT = 0x02
_PROGRAM_NIT = 0x00
_CLOCK_FREQ = 27000000

# ------------
# Parser Utils
# ------------

def _parse_header(packet):
    """Parse MPEG Transport Stream header"""
    return {
        "sync_byte" : packet[0],
        "TEI"       : bool(packet[1] & 0x80),
        "PUSI"      : bool(packet[1] & 0x40),
        "priority"  : bool(packet[1] & 0x20),
        "PID"       : (packet[1] & 0x1f) <<8 | packet[2],
        "TSC"       : (packet[3] & 0xc0),
        "adaptation": bool(packet[3] & 0x20),
        "payload"   : bool(packet[3] & 0x10),
        "counter"   : (packet[3] & 0x0f),
    }

def _parse_pcr(buf):
    """PCR (Program Clock Reference) is a 42 bit counter of
    27Mhz clock.

    +--------+--------+--------+--------+--------+--------+
    |BBBBBBBB|BBBBBBBB|BBBBBBBB|BBBBBBBB|B......E|EEEEEEEE|
    +--------+--------+--------+--------+--------+--------+

    * "B" is 33-bit time based on a 90kHz clock.
    * "E" is 9-bit time based on a 27MHz clock.
    * "." is a reserved bit.
    """
    b0 = buf[0]
    b1 = buf[1]
    b2 = buf[2]
    b3 = buf[3]
    b4 = buf[4]
    b5 = buf[5]
    base = (b0 << 25) | (b1 << 17) | (b2 << 9) | (b3 << 1) | (b4 >> 7)
    ext = ((b4 & 0x01) << 8) | b5
    return base * 300 + ext

def _parse_pts(buf):
    """PTS (Presentation Time Stamp) is a 33-bit counter of
    9000Khz clock.

    +--------+--------+--------+--------+--------+
    |....BBB.|BBBBBBBB|BBBBBBB.|BBBBBBBB|BBBBBBB.|
    +--------+--------+--------+--------+--------+

    * "B" is 33-bit time based on a 90kHz clock.
    * "." is a reserved bit.
    """
    b0 = (buf[0] >> 1) & 0x07
    b1 = buf[1]
    b2 = buf[2] >> 1
    b3 = buf[3]
    b4 = buf[4] >> 1
    base = (b0 << 30) | (b1 << 22) | (b2 << 15) | (b3 << 7) | b4
    return base * 300

def _parse_pat(payload):
    """PAT (Program Association Table) contains a list of programs.

    TABLE HEADER (8 BYTES)

    +--------+--------+--------+--------+--------+--------+--------+--------+
    |TTTTTTTT|      LL|LLLLLLLL|        |        |        |        |        |
    +--------+--------+--------+--------+--------+--------+--------+--------+

    * "T" is a 8-bit identifier. Must be 0x00 for PAT.
    * "L" is a 10-bit length field (= the number of bytes after it)

    TABLE DATA (8 x N BYTES)

    +--------+--------+--------+--------+
    |NNNNNNNN|NNNNNNNN|   PPPPP|PPPPPPPP|
    +--------+--------+--------+--------+

    * "N" is 16-bit program number. 0 means NIT packet.
    * "P" is 13-bit PID.
    """
    table_id = payload[0]
    if table_id != _TABLE_PAT:
        return []

    # (End of Data) = 3-byte header + N bytes
    length = (payload[1] & 0x0f) << 8 | payload[2]
    data = payload[8:3 + length]

    # Trim CRC32 at the end
    data = data[:-4]

    ret = []
    while data:
        program = data[0] << 8 | data[1]
        pid = (data[2] & 0x1f) << 8 | data[3]
        if program != _PROGRAM_NIT:
            ret.append(pid)
        data = data[4:]
    return ret

def _parse_pmt(payload):
    """PMT (Program Mapping Table) records which PID corresponds to
    which data stream.

    NOTE! This function returns the subtitle PID, because we have
    no use for other data streams.
    """
    table_id = payload[0]
    if table_id != _TABLE_PMT:
        return None

    # (End of Data) = 3-byte header + N bytes
    length = (payload[1] & 0x0f) << 8 | payload[2]
    data = payload[8: 3 + length]

    # Trim CRC32 at the end
    data = data[:-4]

    # (Start of Stream) = 4-byte header + N bytes
    meta_length = (data[2] & 0x0f) << 8 | data[3]
    stream = data[4 + meta_length:]

    while stream:
        stream_type = stream[0]
        pid = (stream[1] & 0x1f) << 8 | stream[2]
        nbytes = (stream[3] & 0x0f) << 8 | stream[4]

        if stream_type == 0x06:
            ptr = stream[5:5 + nbytes]
            while ptr:
                if ptr[0] == 0x52 and ptr[2] == 0x87:
                    return pid
                ptr = ptr[2 + ptr[1]:]
        stream = stream[5 + nbytes:]
    return None  # No caption PID

def _parse_caption(payload):
    """Extract the caption (and its timestamp) from packetized
    elementary stream."""
    pts = None
    text = ""

    if payload[7] >> 7:
        pts = _parse_pts(payload[9:])

    header_length = payload[8]
    data_length = payload[11 + header_length] & 0x0f
    data = payload[12 + header_length + data_length:]

    group_id = (data[0] & 0xfc) >> 2
    if group_id in (0x00, 0x20):
        data = data[7 + data[6] * 5:]
    else:
        data = data[6:]

    loop_length = (data[0] << 16) | (data[1] << 8) | data[2]
    data = data[3:3 + loop_length]
    while data:
        unit = data[1]
        size = (data[2] << 16) | (data[3] << 8) | data[4]
        if unit == 0x20:
            text += decode_cprofile(data[8:8 + size])
        data = data[5 + size:]
    return (pts, text)

# --------
# Main API
# --------

@dataclass
class _State:
    clock_init: float = None
    clock_now: float = None
    caption_pid: int = None
    pmt_pids: list = field(default_factory=list)
    captions: list = field(default_factory=list)

    def seconds(self, ts):
        n = ts - self.clock_init
        if n < 0:
            n += _CLOCK_FREQ
        return float(n / _CLOCK_FREQ)

    def done(self):
        ret = []
        self.captions.append((self.clock_now, ""))
        for cur, nex in zip(self.captions, self.captions[1:]):
            if cur[1]:
                start = self.seconds(cur[0])
                end = self.seconds(nex[0])
                ret.append(Caption(start, end, cur[1]))
        return ret

def _captions(fp):
    state = _State()

    while True:
        packet = fp.read(188)
        if len(packet) < 188:
            break

        header = _parse_header(packet)
        if header["sync_byte"] != _SYNC_BYTE:
            raise ValueError("Invalid sync byte: %x" % header["sync_byte"])

        if header["adaptation"]:
            if packet[5] & 0x10:
                state.clock_now = _parse_pcr(packet[6:12])
                if state.clock_init is None:
                    state.clock_init = state.clock_now
            payload = packet[4 + (1 + packet[4]):]
        else:
            payload = packet[4:]

        if header["payload"]:
            if not state.pmt_pids:
                if header["PID"] == 0:
                    state.pmt_pids = _parse_pat(payload[1:])
            elif state.caption_pid is None:
                if header["PID"] in state.pmt_pids:
                    state.caption_pid = _parse_pmt(payload[1:])
            elif state.caption_pid == header["PID"]:
                if header["PUSI"]:
                    try:
                        pts, text = _parse_caption(payload)
                    except IndexError:
                        continue
                    if pts is None:
                        pts = state.clock_now
                    state.captions.append((pts, text))
    return state.done()

def get_captions(path):
    """Read caption from M2TS stream file.

    This scans MPEG transport stream to extracts caption packets.

    Args:
        path (str): Path to a M2TS file.

    Returns:
        A list of `Caption` instances.
    """
    with open(path, 'rb') as fp:
        return _captions(fp)
