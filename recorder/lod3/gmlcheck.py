"""書き出した CityGML を XML スキーマ（CityGML 2.0 ＋ i-UR 3.1）で検査し、LOD3 の立体が閉じているかを読み直して確かめる。"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from lxml import etree
from pyproj import Transformer

from . import XSD_DIR

SCH = XSD_DIR
URO = XSD_DIR / "iur" / "uro" / "3.1" / "urbanObject.xsd"
B = "{http://www.opengis.net/citygml/building/2.0}"
GM = "{http://www.opengis.net/gml}"
XL = "{http://www.w3.org/1999/xlink}"


def driver_xsd() -> etree.XMLSchema:
    imports = {
        "http://www.opengis.net/citygml/2.0": SCH / "schemas.opengis.net/citygml/2.0/cityGMLBase.xsd",
        "http://www.opengis.net/citygml/building/2.0": SCH / "schemas.opengis.net/citygml/building/2.0/building.xsd",
        "http://www.opengis.net/citygml/generics/2.0": SCH / "schemas.opengis.net/citygml/generics/2.0/generics.xsd",
        "http://www.opengis.net/citygml/appearance/2.0": SCH / "schemas.opengis.net/citygml/appearance/2.0/appearance.xsd",
        "https://www.geospatial.jp/iur/uro/3.1": URO,
    }
    body = "".join(f'<xs:import namespace="{ns}" schemaLocation="{p.resolve().as_uri()}"/>' for ns, p in imports.items())
    xsd = f'<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">{body}</xs:schema>'
    return etree.XMLSchema(etree.fromstring(xsd.encode(), base_url=SCH.resolve().as_uri() + "/"))


def check_solids(root) -> list[dict]:
    """lod3Solid の参照先の多角形を集め、向きつきの辺がちょうど逆向きの対になるかを見る。"""
    tr = Transformer.from_crs("EPSG:6668", "EPSG:6675", always_xy=True)
    polys = {p.get(GM + "id"): p for p in root.iter(GM + "Polygon")}
    out = []
    for bel in root.iter(B + "Building"):
        sol = bel.find(B + "lod3Solid")
        if sol is None:
            continue
        refs = [m.get(XL + "href").lstrip("#") for m in sol.iter(GM + "surfaceMember")]
        missing = [r for r in refs if r not in polys]
        cnt = defaultdict(int)
        vol = 0.0
        for r in refs:
            if r not in polys:
                continue
            for lr in polys[r].iter(GM + "LinearRing"):
                v = np.array(lr.find(GM + "posList").text.split(), float).reshape(-1, 3)
                if not np.allclose(v[0], v[-1]):
                    out.append({"building": bel.get(GM + "id"), "error": f"輪が閉じていない {r}"})
                x, y = tr.transform(v[:, 1], v[:, 0])
                P = np.column_stack([x, y, v[:, 2]])[:-1]
                P -= P.mean(0) * 0  # 絶対座標のまま（丸めは 1 mm）
                K = [tuple(np.round(p, 3)) for p in P]
                for i in range(len(K)):
                    cnt[(K[i], K[(i + 1) % len(K)])] += 1
                A = 0.5 * np.sum(np.cross(P - P[0], np.roll(P, -1, axis=0) - P[0]), axis=0)
                vol += float((P[0] - [0, 0, 0]) @ A) / 3.0 if False else float(P[0] @ A) / 3.0
        bad = sum(1 for e, n in cnt.items() if n != 1 or cnt.get((e[1], e[0]), 0) != 1)
        out.append({"building": bel.get(GM + "id"), "polygons": len(refs), "missing_refs": len(missing),
                    "unmatched_edges": bad, "signed_volume_m3": round(vol, 1)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recorder lod3 check")
    ap.add_argument("gml")
    args = ap.parse_args(argv)
    doc = etree.parse(args.gml, etree.XMLParser(huge_tree=True))
    xs = driver_xsd()
    ok = xs.validate(doc)
    print("XML スキーマ:", "合格" if ok else "不合格")
    if not ok:
        errs = list(xs.error_log)
        print("誤りの数:", len(errs))
        for e in errs[:25]:
            print(f"  {e.line}: {e.message[:300]}")
    for r in check_solids(doc.getroot()):
        print("立体:", r)
    ids = [el.get(GM + "id") for el in doc.getroot().iter() if el.get(GM + "id")]
    dup = len(ids) - len(set(ids))
    print("gml:id の重複:", dup)
    return 0 if ok and dup == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
