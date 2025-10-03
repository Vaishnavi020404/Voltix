# imghdr shim for Python 3.12+
import mimetypes

def what(file, h=None):
    # crude shim: checks MIME by file extension
    if isinstance(file, str):
        t, _ = mimetypes.guess_type(file)
        if t and t.startswith('image/'):
            return t.split('/')[-1]
    return None
