import os
import tempfile
import contextlib

@contextlib.contextmanager
def win32_tempfile():
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    try:
        yield tmpf
    finally:
        os.unlink(tmpf.name)

def create_tempfile():
    if os.name == 'nt':
        return win32_tempfile()
    else:
        return tempfile.NamedTemporaryFile()
