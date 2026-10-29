"""Windows/Jupyter 호환 스레드 기반 이미지 읽기·전처리·선행 로딩."""
from __future__ import annotations
import io, re, struct, zlib, hashlib
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps
from .utils import file_sha

LEGACY_IMAGE_POLICY = {'version':'old-start-compatible-v3-svg-hyperlink-fix','frame':0,'background':'white',
                'exif_transpose':True,'svg_width':1024,'png_bad_ancillary_crc':'remove_only',
                'raster_resize':'none','decoder':'Pillow_signature','external_svg_resources':'reject',
                'svg_navigation_links':'static_renderer_no_activation'}
IMAGE_POLICY = {**LEGACY_IMAGE_POLICY, 'version':'white-alpha-visibility-v1',
                'background':'white; dark_gray_64_only_if_visible_alpha_becomes_uniform_white'}


def composite_visible(rgba):
    white = Image.alpha_composite(Image.new('RGBA', rgba.size, (255,255,255,255)), rgba).convert('RGB')
    alpha_min, alpha_max = rgba.getchannel('A').getextrema()
    changed = alpha_min < alpha_max and alpha_max > 0 and white.getextrema() == ((255,255),)*3
    result = Image.alpha_composite(Image.new('RGBA', rgba.size, (64,64,64,255)), rgba).convert('RGB') if changed else white
    def pixel_sha(im):
        return hashlib.sha256(str(im.size).encode('ascii') + im.tobytes()).hexdigest()
    meta = {'background_corrected':changed, 'rendered_rgb_sha256':pixel_sha(result),
            'legacy_white_rgb_sha256':pixel_sha(white),
            'background_rgb':[64,64,64] if changed else [255,255,255]}
    if result is not white: white.close()
    return result, meta


def image_path(root: Path, row: dict) -> Path:
    # URL·record_id를 이미지 경로 추정에 미사용
    # 공개 test 109건처럼 학습 폴더 밖에 이미지가 있는 추가 학습 레코드는 행에 image_root를 가진다
    if row.get('image_root'): root = Path(row['image_root'])
    name = Path(row['image_file'])
    candidates = [root/name, root/'images'/name, root/'images'/name.name]
    for p in candidates:
        if p.resolve().is_relative_to(root.resolve()) and p.is_file(): return p
    raise FileNotFoundError(f"이미지 연결 실패: {row['record_id']} / {row['image_file']}")


def repair_png_ancillary(data: bytes) -> tuple[bytes, list[str]]:
    if not data.startswith(b'\x89PNG\r\n\x1a\n'): raise ValueError('PNG signature 오류')
    pos = 8; chunks = [data[:8]]; removed = []; ended = False
    while pos < len(data):
        if pos+12 > len(data): raise ValueError('PNG chunk 잘림')
        n = struct.unpack('>I', data[pos:pos+4])[0]; end = pos+n+12
        if end > len(data): raise ValueError('PNG 데이터 잘림')
        kind = data[pos+4:pos+8]
        ok = struct.unpack('>I', data[end-4:end])[0] == zlib.crc32(data[pos+4:end-4]) & 0xffffffff
        if ok: chunks.append(data[pos:end])
        elif kind[0] & 32: removed.append(kind.decode('ascii'))
        else: raise ValueError(f'PNG 핵심 chunk CRC 오류: {kind!r}')
        pos = end
        if kind == b'IEND': ended = True; break
    if not ended: raise ValueError('PNG IEND 없음')
    return b''.join(chunks), removed



def validate_svg_resources(data: bytes) -> None:
    """SVG 하이퍼링크와 렌더링용 외부 파일 참조 구분. 원본 바이트 유지."""
    if re.search(br'<!ENTITY\b', data, re.I):
        raise ValueError('SVG ENTITY 선언: 원본 확인 필요')
    if re.search(br'<\?xml-stylesheet\b', data, re.I):
        raise ValueError('외부 스타일시트 SVG: 원본 확인 필요')
    try:
        root = ET.fromstring(data.decode('utf-8-sig'))
    except (UnicodeDecodeError, ET.ParseError) as exc:
        raise ValueError(f'SVG XML 해석 오류: {exc}') from exc

    # href는 노드 종류까지 확인. a는 클릭 목적지, image/use 등은 렌더링 참조
    anchor_tags = {'a', '{http://www.w3.org/2000/svg}a'}
    href_attrs = {'href', '{http://www.w3.org/1999/xlink}href'}
    for element in root.iter():
        for key, raw_value in element.attrib.items():
            if key not in href_attrs:
                continue
            value = raw_value.strip()
            if element.tag in anchor_tags:
                continue
            if not value or value.startswith('#'):
                continue
            # 파일 내 포함된 래스터만 허용. 외부/상대경로와 중첩 SVG는 차단
            if re.match(r'data:image/(?:png|jpeg|jpg|gif|webp);base64,', value, re.I):
                continue
            tag = element.tag.rsplit('}', 1)[-1]
            raise ValueError(f'외부 리소스 SVG: <{tag}>의 href 참조 확인 필요')

        # 스타일 속성·style 태그의 외부 URL 차단
        style_values = list(element.attrib.values())
        if element.tag.rsplit('}', 1)[-1] == 'style' and element.text:
            style_values.append(element.text)
        for value in style_values:
            if re.search(r'@import\b', value, re.I):
                raise ValueError('외부 CSS import SVG: 원본 확인 필요')
            for match in re.finditer(r'url\(\s*([^)]*?)\s*\)', value, re.I):
                reference = match.group(1).strip().strip('"\'').strip()
                if reference and not reference.startswith('#'):
                    raise ValueError('외부 CSS URL SVG: 원본 확인 필요')


def load_image(root: Path, row: dict) -> tuple[Image.Image, dict]:
    path = image_path(root, row); data = path.read_bytes(); removed = []
    svg = bool(re.search(br'<svg(?:\s|>)', data[:65536], re.I)) and data.lstrip(b'\xef\xbb\xbf \r\n\t').startswith(b'<')
    if svg:
        validate_svg_resources(data)
        import resvg_py
        src = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=data.decode('utf-8-sig'), width=1024)))
        fmt = 'SVG'
    else:
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            data, removed = repair_png_ancillary(data)
        src = Image.open(io.BytesIO(data)); fmt = src.format
    with src:
        src.seek(0)
        rgba = ImageOps.exif_transpose(src).convert('RGBA')
        image, rendering = composite_visible(rgba)
        rgba.close()
    return image, {'format':fmt,'width':image.width,'height':image.height,
                   'removed_png_chunks':removed,'image_sha256':file_sha(path), **rendering}


def audit_image(task: tuple[Path, dict]) -> dict:
    root, row = task
    try:
        im, meta = load_image(root, row); im.close()
        return {'record_id':row['record_id'],'ok':True,**meta}
    except Exception as e:
        return {'record_id':row['record_id'],'ok':False,'error':repr(e)}


def prefetch_images(root: Path, rows: list[dict], workers: int = 4, ahead: int = 8):
    # 프로세스 spawn 대신 스레드 사용: 노트북 __main__ 직렬화 문제 회피
    pending = deque(); it = iter(rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        def submit():
            try: r = next(it)
            except StopIteration: return False
            pending.append((r, pool.submit(load_image, root, r))); return True
        for _ in range(max(1, ahead)): submit()
        try:
            while pending:
                r, future = pending.popleft(); image, meta = future.result(); submit()
                yield r, image, meta
        finally:
            for _, f in pending:
                if not f.cancel():
                    try:
                        im, _ = f.result(); im.close()
                    except Exception: pass
