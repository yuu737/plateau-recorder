"""他の棟による遮蔽：カメラから面の点までの線分が、PLATEAU の他の建物の壁を先に横切るか。

campus-recorder の判定は「手で書いた exclude」と「自分の棟の他の面」だけで、手前の別の棟を見ていなかった
（2026-11-17、タワー75 の東面 f15 で手前の棟が面に貼り付いていた）。ここでは PLATEAU（名古屋 2022、
3 次メッシュ 52365768）の全建物の LOD1 の壁（足元の辺を最低〜最高の標高まで鉛直に立てた板）を遮蔽物にする。
閉じた立体を通り抜ける線分は、必ずどこかの壁を横切る（カメラも面の点も地面より上にあるため）。

限界：木・車・人・地形は入っていない。LOD1 は棟の一番高い所まで立てるので、低い部分のある棟は遮蔽を大きめに
見る。3 次メッシュの外の建物は入っていない。写真の中で実際に何が写っているかの見分けは、係（Claude）がする。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .lod3 import SOURCE_GML

MARGIN_M = 0.5      # 面の点の手前 0.5 m 以内の壁は数えない（面と同じ線の上の隣の棟の壁を遮蔽にしないため）
NS = {"bldg": "http://www.opengis.net/citygml/building/2.0", "gml": "http://www.opengis.net/gml"}


@lru_cache(maxsize=4)
def _load(gml: str, epsg: int, origin_lon: float, origin_lat: float) -> tuple:
    from lxml import etree
    from pyproj import Transformer
    from shapely.geometry import Polygon

    fwd = Transformer.from_crs("EPSG:6668", f"EPSG:{epsg}", always_xy=True)
    ox, oy = fwd.transform(origin_lon, origin_lat)
    tree = etree.parse(gml, etree.XMLParser(huge_tree=True))
    out = []
    for bel in tree.iter(f"{{{NS['bldg']}}}Building"):
        solid = bel.find(".//bldg:lod1Solid", NS)
        if solid is None:
            continue
        rings = []
        for pl in solid.iter(f"{{{NS['gml']}}}posList"):
            v = np.array(pl.text.split(), float).reshape(-1, 3)      # 緯度 経度 標高
            x, y = fwd.transform(v[:, 1], v[:, 0])
            rings.append(np.column_stack([np.asarray(x) - ox, np.asarray(y) - oy, v[:, 2]]))
        if not rings:
            continue
        zmin = min(float(r[:, 2].min()) for r in rings)
        zmax = max(float(r[:, 2].max()) for r in rings)
        bottoms = [r for r in rings if np.ptp(r[:, 2]) < 1e-3 and abs(r[0, 2] - zmin) < 1e-3]
        ring = bottoms[0] if bottoms else max(rings, key=lambda r: Polygon(r[:, :2]).area)
        poly = Polygon(ring[:, :2])
        if not poly.is_valid:
            poly = poly.buffer(0)
        out.append({"id": bel.get(f"{{{NS['gml']}}}id"), "poly": poly, "zmin": zmin, "zmax": zmax})
    return tuple(out)


def plateau_buildings(crs: dict, gml: Path | None = None) -> list[dict]:
    """PLATEAU の全建物の足元（原点からの m）と標高の範囲。"""
    return list(_load(str(gml or SOURCE_GML), int(crs.get("epsg", 6675)), float(crs["origin_lon"]), float(crs["origin_lat"])))


def self_ids(b: dict, bldgs: list[dict]) -> set[str]:
    """対象の棟そのものの PLATEAU の建物（足元の根拠に書かれた id と、足元が一番重なる物）。"""
    from shapely.geometry import Polygon

    ids = set()
    fp = b.get("footprint") or {}
    for ev in fp.get("evidence", []) if isinstance(fp, dict) else []:
        if isinstance(ev, str) and ev.startswith("plateau:"):
            ids.add(ev.split(":", 1)[1])
    poly = Polygon(fp["polygon"]) if isinstance(fp, dict) and fp.get("polygon") else None
    if poly is not None and poly.is_valid and bldgs:
        best = max(bldgs, key=lambda x: x["poly"].intersection(poly).area if x["poly"].intersects(poly) else 0.0)
        if best["poly"].intersects(poly):
            ids.add(best["id"])
    return ids


class Walls:
    """遮蔽物の壁（鉛直の板）。z は対象の棟の z（標高 − ground_z）。"""

    def __init__(self, b: dict, gml: Path | None = None):
        import shapely
        from shapely.geometry import LineString

        gz = b["ground_z"]
        gz = float(gz["value"] if isinstance(gz, dict) else gz)
        bl = plateau_buildings(b["crs"], gml)
        self.skipped = sorted(self_ids(b, bl))
        segs = []
        for x in bl:
            if x["id"] in self.skipped:
                continue
            for p in (list(x["poly"].geoms) if hasattr(x["poly"], "geoms") else [x["poly"]]):
                for ring in [p.exterior, *p.interiors]:
                    c = np.asarray(ring.coords)[:, :2]
                    for a, q in zip(c[:-1], c[1:]):
                        if np.hypot(*(q - a)) >= 0.05:
                            segs.append((x["id"], a, q, x["zmin"] - gz, x["zmax"] - gz))
        self.segs = segs
        self.tree = shapely.STRtree([LineString([s[1], s[2]]) for s in segs]) if segs else None

    def near(self, C: np.ndarray, A2: np.ndarray, B2: np.ndarray) -> list[tuple]:
        """カメラと面の両端の三角形（平面）にかかる壁だけ。"""
        from shapely.geometry import Polygon

        if self.tree is None:
            return []
        wedge = Polygon([C[:2], A2, B2])
        if not wedge.is_valid or wedge.area < 1e-6:
            return []
        return [self.segs[j] for j in self.tree.query(wedge, predicate="intersects")]

    def occluded(self, C: np.ndarray, P: np.ndarray, A2: np.ndarray, B2: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """C から P（…×3）への線分が壁を面の手前で横切るか。返り値（遮蔽の真偽、遮った棟の id）。"""
        shp = P.shape[:-1]
        Pf = P.reshape(-1, 3)
        d = Pf - C
        dist = np.linalg.norm(d, axis=1)
        lam_max = 1.0 - MARGIN_M / np.maximum(dist, 1e-6)
        occ = np.zeros(len(Pf), bool)
        hit_ids: set[str] = set()
        for bid, p0, p1, z0, z1 in self.near(C, A2, B2):
            u = p1 - p0
            L = float(np.hypot(*u))
            u = u / L
            nrm = np.array([u[1], -u[0]])
            den = d[:, :2] @ nrm
            num = float((p0 - C[:2]) @ nrm)
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = num / den
            ok = (lam > 1e-3) & (lam < lam_max)
            if not ok.any():
                continue
            X = C + lam[:, None] * d
            ss = (X[:, :2] - p0) @ u
            hit = ok & (ss >= 0) & (ss <= L) & (X[:, 2] >= z0) & (X[:, 2] <= z1)
            if hit.any():
                hit_ids.add(bid)
                occ |= hit
        return occ.reshape(shp), sorted(hit_ids)
