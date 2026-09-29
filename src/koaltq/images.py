from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

from PIL import Image, ImageOps


def clean_png_profile(raw):
    """Remove only a bad ancillary iCCP checksum, never damaged pixel chunks."""
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return raw, False
    result, offset, changed = [raw[:8]], 8, False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError("Truncated PNG chunk")
        size = struct.unpack(">I", raw[offset:offset + 4])[0]
        end = offset + size + 12
        if end > len(raw):
            raise ValueError("Truncated PNG data")
        kind = raw[offset + 4:offset + 8]
        payload = raw[offset + 8:end - 4]
        actual = struct.unpack(">I", raw[end - 4:end])[0]
        valid = (zlib.crc32(kind + payload) & 0xffffffff) == actual
        if not valid and kind == b"iCCP":
            changed = True
        elif not valid:
            raise ValueError(f"PNG checksum failed: {kind!r}")
        else:
            result.append(raw[offset:end])
        offset = end
        if kind == b"IEND":
            break
    return b"".join(result), changed


def load_image(path, *, svg_max_side=2048):
    raw = Path(path).read_bytes()
    raw, repaired = clean_png_profile(raw)
    is_svg = b"<svg" in raw[:8192].lower()
    if is_svg:
        import resvg_py
        raw = resvg_py.svg_to_bytes(svg_string=raw.decode("utf-8-sig"), width=svg_max_side)
    with Image.open(io.BytesIO(raw)) as source:
        source.seek(0)
        source.load()
        fmt, original_size = source.format, source.size
        im = ImageOps.exif_transpose(source).convert("RGBA")
    background = Image.new("RGBA", im.size, "white")
    background.alpha_composite(im)
    return background.convert("RGB"), {
        "format": "SVG" if is_svg else fmt, "size": list(original_size),
        "removed_bad_iccp": repaired,
    }
