"""CityGML 2.0・GML 3.1.1・xAL・i-UR 3.1 の XSD を、import / include をたどって手元に写す。
写した XSD の中の schemaLocation は手元の相対パスに書き換える（検査をネットに頼らない）。"""
from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

from . import XSD_DIR

ROOT = XSD_DIR
START = [
    "http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd",
    "http://schemas.opengis.net/citygml/building/2.0/building.xsd",
    "http://schemas.opengis.net/citygml/generics/2.0/generics.xsd",
    "http://schemas.opengis.net/citygml/appearance/2.0/appearance.xsd",
] + [f"http://schemas.opengis.net/citygml/{m}/2.0/{f}.xsd" for m, f in (
    ("bridge", "bridge"), ("transportation", "transportation"), ("tunnel", "tunnel"), ("landuse", "landUse"),
    ("cityobjectgroup", "cityObjectGroup"), ("cityfurniture", "cityFurniture"), ("vegetation", "vegetation"),
    ("waterbody", "waterBody"), ("relief", "relief"))]
# PLATEAU の zip に同梱の i-UR（公開 URL は 404）。import 先を手元の写しに書き換える
LOCAL = [XSD_DIR / "iur" / "uro" / "3.1" / "urbanObject.xsd", XSD_DIR / "iur" / "urf" / "3.1" / "urbanFunction.xsd"]
LOC = re.compile(r'(schemaLocation\s*=\s*")([^"]+)(")')


def local_path(url: str) -> Path:
    u = urlparse(url)
    return ROOT / u.netloc / u.path.lstrip("/")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "schema-mirror"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    seen, queue = set(), list(START)
    while queue:
        url = queue.pop()
        if url in seen:
            continue
        seen.add(url)
        dst = local_path(url)
        orig = dst.with_suffix(dst.suffix + ".orig")      # 書き換える前の元（何度回しても同じ結果にする）
        if orig.exists():
            text = orig.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                data = fetch(url)
            except Exception as e:  # noqa: BLE001
                print("取れない:", url, e)
                continue
            text = data.decode("utf-8", errors="replace")
            orig.parent.mkdir(parents=True, exist_ok=True)
            orig.write_text(text, encoding="utf-8")
        refs = []

        def repl(m):
            ref = m.group(2).strip()
            absu = urljoin(url, ref)
            refs.append(absu)
            rel = os.path.relpath(local_path(absu), dst.parent).replace("\\", "/")
            return m.group(1) + rel + m.group(3)

        new = LOC.sub(repl, text)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(new, encoding="utf-8")
        queue += [r for r in refs if r not in seen]
    print("写した XSD:", len(seen))
    for p in LOCAL:
        orig = p.with_suffix(p.suffix + ".orig")
        if not orig.exists():
            orig.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        text = orig.read_text(encoding="utf-8")

        def repl2(m):
            ref = m.group(2).strip()
            if ref.startswith("http"):
                return m.group(1) + os.path.relpath(local_path(ref), p.parent).replace("\\", "/") + m.group(3)
            return m.group(0)

        p.write_text(LOC.sub(repl2, text), encoding="utf-8")
        print("書き換え:", p.name)


if __name__ == "__main__":
    sys.exit(main())
