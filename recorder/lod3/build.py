"""campus-recorder の棟 JSON → PLATEAU 形式の LOD3（CityGML 2.0 ＋ i-UR 3.1）。

やること
1. 体：塊（blocks）を高さの段ごとに足し、彫り込み（部品の cut）を引く（manifold3d）。
   LOD3.x の取得基準（壁・付属物の短辺など）に満たない段差・彫り込みは省く。
2. 面：体の三角形を平面ごとにまとめ、向きで 壁・屋根・地面・軒下（下向き）・床（上向き、上に体がある）に分ける。
3. 窓：棟 JSON の面ごとの窓・ガラス面（face の s・z）を、同じ平面の壁に穴として開ける。
   面の線に壁が無い（彫り込みの中）ときは、奥の平行な壁を探す。
4. 付属物：部品（彫り込みでないもの）を BuildingInstallation に。体と重なる所は引く。基準未満は省く。
5. 根拠：各地物に gen 属性（値・状態・±・根拠の写真）を付け、全文は evidence.json に。
6. PLATEAU の元の建物（LOD0・LOD1・属性）を写し、LOD3 の要素とデータ品質属性を足して書き出す。

座標は棟 JSON の局所座標（EPSG:6675、原点 35.1355, 136.9755 からの m。z は ground_z からの m）で作り、
最後に EPSG:6697（緯度・経度・標高）に直す。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import manifold3d as M
import numpy as np
import shapely
from lxml import etree
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box as sbox
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .. import parts_geom as G  # 純 Python。math だけ
from ..schema import bay_edges
from . import OUT_DIR, ROOT, SOURCE_GML
from . import quality as Q

CR = ROOT     # 棟 JSON（data/buildings）と屋上の照合（out/<棟>/roof.json）の置き場

NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml",
    "gen": "http://www.opengis.net/citygml/generics/2.0",
    "app": "http://www.opengis.net/citygml/appearance/2.0",
    "uro": "https://www.geospatial.jp/iur/uro/3.1",
    "xlink": "http://www.w3.org/1999/xlink",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}
B, GM, GEN, URO, CORE, XL = (f"{{{NS[k]}}}" for k in ("bldg", "gml", "gen", "uro", "core", "xlink"))

# LOD3.x の取得基準（標準製品仕様書 4.2.1.4.2〜）。短辺・面積の判定に使う
LOD_RULES = {
    "3.0": {"cut": lambda s: s[0] >= 3.0,
            "win": lambda w, h: min(w, h) >= 1.0,
            "inst": lambda s: s[1] >= 3.0 or (s[1] >= 1.0 and s[1] * s[2] >= 3.0),
            "sliver": 0.6, "hmerge": 0.5, "simplify": 0.05},
    "3.1": {"cut": lambda s: s[0] >= 1.0 and s[0] * s[1] >= 3.0,
            "win": lambda w, h: min(w, h) >= 1.0,
            "inst": lambda s: s[1] >= 3.0 or (s[1] >= 1.0 and s[1] * s[2] >= 3.0),
            "sliver": 0.3, "hmerge": 0.3, "simplify": 0.03},
    "3.2": {"cut": lambda s: s[0] >= 1.0 or s[0] * s[1] >= 1.0,
            "win": lambda w, h: w * h >= 1.0,
            "inst": lambda s: s[1] >= 1.0 or s[1] * s[2] >= 1.0,
            "sliver": 0.2, "hmerge": 0.1, "simplify": 0.02},
    "3.3": {"cut": lambda s: True, "win": lambda w, h: True, "inst": lambda s: True, "sliver": 0.05,
            "hmerge": 0.0, "simplify": 0.005},
}
SKIP_PART_TYPES = {"patch", "planter", "pipes", "asset"}   # 色の塗り分け・植栽・細い管・CC0 の飾り
# 建物に付かない物（都市設備）。LOD3 の BuildingInstallation には入れない
FURNITURE_WORDS = ("ベンチ", "机", "ごみ", "掲示板", "自転車", "植え込み", "プランター", "灰皿", "自販機", "自動販売機")


# ------------------------------------------------------------------ 値と根拠

def V(m, default=None):
    if isinstance(m, (int, float)) and not isinstance(m, bool):
        return float(m)
    if isinstance(m, dict) and m.get("status") != "unknown" and m.get("value") is not None:
        return float(m["value"])
    return default


def meas(m) -> dict | None:
    """根拠つきの値を短い辞書に。"""
    if not isinstance(m, dict) or "status" not in m:
        return None
    return {k: m.get(k) for k in ("value", "unit", "tol", "status", "evidence", "method") if m.get(k) not in (None, "", [])}


def uid(*key) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "meijo-lod3/" + "/".join(map(str, key))))


# ------------------------------------------------------------------ 座標

class Geo:
    def __init__(self, crs: dict, ground_z: float):
        epsg = crs.get("epsg", 6675)
        fwd = Transformer.from_crs("EPSG:6668", f"EPSG:{epsg}", always_xy=True)
        self.inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:6668", always_xy=True)
        self.ox, self.oy = fwd.transform(crs["origin_lon"], crs["origin_lat"])
        self.gz = ground_z

    def latlonh(self, pts: np.ndarray) -> np.ndarray:
        lon, lat = self.inv.transform(pts[:, 0] + self.ox, pts[:, 1] + self.oy)
        return np.column_stack([lat, lon, pts[:, 2] + self.gz])


# ------------------------------------------------------------------ manifold の道具

def cs_from_shapely(geom) -> M.CrossSection:
    contours = []
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.is_empty:
            continue
        p = orient(p, 1.0)
        contours.append([tuple(c) for c in list(p.exterior.coords)[:-1]])
        for r in p.interiors:
            contours.append([tuple(c) for c in list(r.coords)[:-1]])
    return M.CrossSection(contours, M.FillRule.EvenOdd)


def prism_xy(geom, z0: float, z1: float) -> M.Manifold:
    return M.Manifold.extrude(cs_from_shapely(geom), z1 - z0).translate((0.0, 0.0, z0))


def prim_manifold(pr: dict, axes) -> M.Manifold | None:
    """parts_geom の素片（prism・ext_a・ext_n）→ 世界座標の manifold。"""
    x0, y0, ux, uy, nx, ny = axes
    # 局所 (a, n, z) → 世界 の 4×3 変換
    T = np.array([[ux, nx, 0.0, x0], [uy, ny, 0.0, y0], [0.0, 0.0, 1.0, 0.0]])
    k = pr["k"]
    try:
        if k == "prism":
            poly = Polygon(pr["pts"])
            if not poly.is_valid or poly.area < 1e-6:
                poly = poly.buffer(0)
            if poly.area < 1e-6:
                return None
            m = M.Manifold.extrude(cs_from_shapely(poly), pr["z1"] - pr["z0"]).translate((0, 0, pr["z0"]))
        elif k == "ext_a":      # (a, z) の多角形を n0→n1 に押し出す
            poly = Polygon(pr["pts"]).buffer(0)
            if poly.area < 1e-6:
                return None
            m = M.Manifold.extrude(cs_from_shapely(poly), pr["n1"] - pr["n0"]).translate((0, 0, pr["n0"]))
            # 押し出し後の (X=a, Y=z, Z=n) → (a, n, z)
            m = m.transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], float))
            if m.volume() < 0:
                m = m.mirror((0, 0, 0))
        elif k == "ext_n":      # (n, z) の多角形を a0→a1 に押し出す
            poly = Polygon(pr["pts"]).buffer(0)
            if poly.area < 1e-6:
                return None
            m = M.Manifold.extrude(cs_from_shapely(poly), pr["a1"] - pr["a0"]).translate((0, 0, pr["a0"]))
            # (X=n, Y=z, Z=a) → (a, n, z)
            m = m.transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], float))
        else:
            return None
    except Exception:
        return None
    if m.is_empty():
        return None
    # 鏡映の変換でも manifold は向きを保つ（行列式が負なら裏返す）
    return m.transform(T)


def mesh_of(m: M.Manifold):
    me = m.to_mesh64()
    return np.asarray(me.vert_properties, float)[:, :3], np.asarray(me.tri_verts, np.int64)


# ------------------------------------------------------------------ 面（平面ごと）

def plane_basis(n: np.ndarray):
    n = n / np.linalg.norm(n)
    if abs(n[2]) > 0.99:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = np.cross([0.0, 0.0, 1.0], n)
        e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


class Facet:
    """1 つの平面の上の多角形（穴あり）。2D は (e1, e2) の座標で、反時計回り＝外向き。"""

    def __init__(self, n, d, poly2d, vmap):
        self.n, self.e1, self.e2 = plane_basis(np.asarray(n, float))
        self.d = float(d)
        self.poly = poly2d
        self.vmap = vmap          # 2D 座標 → 元の 3D 頂点（形を崩さないため）
        self.kind = None
        self.meta = {}

    def to3d(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        out = []
        for s, t in xy:
            p = self.vmap.get((round(s, 6), round(t, 6)))
            out.append(p if p is not None else self.n * self.d + s * self.e1 + t * self.e2)
        return np.array(out)

    def to2d(self, P) -> np.ndarray:
        P = np.asarray(P, float)
        return np.column_stack([P @ self.e1, P @ self.e2])


def facets_from_mesh(Vx, T, ang_tol=1e-3, d_tol=1e-3) -> list[Facet]:
    P0, P1, P2 = Vx[T[:, 0]], Vx[T[:, 1]], Vx[T[:, 2]]
    N = np.cross(P1 - P0, P2 - P0)
    area2 = np.linalg.norm(N, axis=1)
    keep = area2 > 1e-10
    N[keep] /= area2[keep, None]
    D = np.einsum("ij,ij->i", N, P0)
    # 同じ平面のまとまり：法線と距離で束ね、辺のつながりで分ける
    keys = defaultdict(list)
    for i in np.nonzero(keep)[0]:
        k = (round(N[i, 0] / ang_tol), round(N[i, 1] / ang_tol), round(N[i, 2] / ang_tol), round(D[i] / d_tol))
        keys[k].append(i)
    facets = []
    for k, tris in keys.items():
        tris = np.array(tris)
        # 辺でつながる三角形を union-find
        parent = {int(t): int(t) for t in tris}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        edge_owner = {}
        for t in tris:
            a, b, c = T[t]
            for e in ((a, b), (b, c), (c, a)):
                key = (min(e), max(e))
                if key in edge_owner:
                    ra, rb = find(int(t)), find(edge_owner[key])
                    if ra != rb:
                        parent[ra] = rb
                else:
                    edge_owner[key] = int(t)
        groups = defaultdict(list)
        for t in tris:
            groups[find(int(t))].append(int(t))
        for g in groups.values():
            n = N[g].mean(axis=0)
            n /= np.linalg.norm(n)
            d = float(np.mean(D[g]))
            f = Facet(n, d, None, {})
            polys = []
            for t in g:
                P = Vx[T[t]]
                xy = f.to2d(P)
                for (s, tt), p in zip(xy, P):
                    f.vmap[(round(s, 6), round(tt, 6))] = p
                polys.append(Polygon(xy))
            u = unary_union(polys)
            # 幅 1 mm 未満の帯（別々に計算した壁の線が 0.1 mm ずれてできる）を落とす。穴と外周が線で接するのを防ぐ
            jm = {"join_style": "mitre", "mitre_limit": 20.0}
            u2 = u.buffer(-5e-4, **jm).buffer(5e-4, **jm)
            if not u2.is_empty and abs(u2.area - u.area) < max(1e-3, 1e-3 * u.area):
                u = u2
            u = orient_all(u)
            for p in (u.geoms if isinstance(u, MultiPolygon) else [u]):
                # 面積ほぼ 0 の細い切れ端（3 点がほぼ一直線）は捨てる。継ぎ目は T 字の直しで閉じる
                if p.area > 1e-6 and not (p.area < 1e-2 and 2 * p.area / p.length < 2e-3):
                    ff = Facet(n, d, p, f.vmap)
                    facets.append(ff)
    return facets


def orient_all(g):
    if isinstance(g, MultiPolygon):
        return MultiPolygon([orient(p, 1.0) for p in g.geoms if not p.is_empty])
    if isinstance(g, Polygon):
        return orient(g, 1.0)
    polys = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
    return MultiPolygon([orient(p, 1.0) for p in polys])


def polys_of(g) -> list[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    out = []
    for p in getattr(g, "geoms", []):
        out += polys_of(p)
    return out


# ------------------------------------------------------------------ 棟 JSON の読み取り

def face_axes(f):
    (x0, y0), (x1, y1) = f["from"], f["to"]
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    return np.array([x0, y0], float), np.array([ux, uy]), np.array([uy, -ux]), L


def face_window_rects(f: dict) -> list[dict]:
    """windows.face_window_rects と同じ式（柱割り）＋ glazing。各矩形に根拠を付ける。"""
    out = []
    bays = f.get("bays") or {}
    _, _, _, Lgeom = face_axes(f)
    L = V(f.get("length")) or Lgeom
    edges, _ = bay_edges(bays, L)
    cnt, pitch = V(bays.get("count")), V(bays.get("pitch"))
    bay_ev = {k: meas(bays.get(k)) for k in ("count", "pitch", "pier_width", "margin_from", "margin_to") if meas(bays.get(k))}
    if edges and cnt and pitch:
        pitch = (edges[-1] - edges[0]) / cnt
        margin = edges[0]
        pier = V(bays.get("pier_width")) or 0.6
        for wi, w in enumerate(f.get("windows") or []):
            zb, zt = V(w.get("z_bottom")), V(w.get("z_top"))
            if zb is None or zt is None:
                out.append({"face": f["id"], "src": f"faces.{f['id']}.windows[{wi}]", "row": w.get("name", "窓"),
                            "skip": "上端か下端が unknown"})
                continue
            ww = V(w.get("width")) or (pitch - pier)
            per = int(w.get("per_bay") or 1)
            sub = ww / per
            gap = (V(w.get("gap")) if V(w.get("gap")) is not None else 0.15) if per > 1 else 0.1
            ev = {"z_bottom": meas(w.get("z_bottom")), "z_top": meas(w.get("z_top")), "width": meas(w.get("width")),
                  "bays": bay_ev}
            if not V(w.get("width")):
                ev["width_rule"] = "柱間隔 − 柱型の幅（width が無い）"
            for i in w.get("bays") or range(1, int(cnt) + 1):
                for k in range(per):
                    sc = margin + (i - 1) * pitch + pitch / 2 + (k - (per - 1) / 2) * sub
                    ws = sub - gap
                    out.append({"face": f["id"], "src": f"faces.{f['id']}.windows[{wi}]", "row": w.get("name", "窓"),
                                "bay": int(i), "k": k, "s0": sc - ws / 2, "s1": sc + ws / 2, "z0": zb, "z1": zt,
                                "kind": "window", "ev": ev})
    for gi, gz in enumerate(f.get("glazing") or []):
        zt = V(gz.get("z_top"))
        if zt is None:
            out.append({"face": f["id"], "src": f"faces.{f['id']}.glazing[{gi}]", "skip": "上端が unknown"})
            continue
        a0 = V(gz.get("along0"), 0.0)
        a1 = V(gz.get("along1"), Lgeom)
        out.append({"face": f["id"], "src": f"faces.{f['id']}.glazing[{gi}]", "row": gz.get("name", "ガラス面"),
                    "s0": a0, "s1": a1, "z0": V(gz.get("z_bottom"), 0.0), "z1": zt, "kind": "glazing",
                    "ev": {k: meas(gz.get(k)) for k in ("along0", "along1", "z_bottom", "z_top") if meas(gz.get(k))}})
    return out


def _set_path(d: dict, path: tuple, value: float):
    o = d
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = {**o[path[-1]], "value": value}


def face_window_rects_sigma(f: dict) -> list[dict]:
    """窓の矩形に、縁の位置の標準偏差を付ける。入力の値（余白・柱間隔・幅・上下端…）を 1 つずつ ± だけずらして
    作り直し、縁の動きを二乗和で足す（一次の誤差伝播）。± の無い値が効く縁は「不明」にする。"""
    base = face_window_rects(f)
    idx = [i for i, r in enumerate(base) if "skip" not in r]
    inputs = []
    bays = f.get("bays") or {}
    for k in ("margin_from", "end_margin", "margin_to", "pitch", "pier_width"):
        if isinstance(bays.get(k), dict) and "status" in bays[k]:
            inputs.append((("bays", k), bays[k]))
    if isinstance(f.get("length"), dict):
        inputs.append((("length",), f["length"]))
    for wi, w in enumerate(f.get("windows") or []):
        for k in ("z_bottom", "z_top", "width", "gap"):
            if isinstance(w.get(k), dict) and "status" in w[k]:
                inputs.append((("windows", wi, k), w[k]))
    for gi, g in enumerate(f.get("glazing") or []):
        for k in ("along0", "along1", "z_bottom", "z_top"):
            if isinstance(g.get(k), dict) and "status" in g[k]:
                inputs.append((("glazing", gi, k), g[k]))
    acc = {i: np.zeros(4) for i in idx}
    missing = {i: [] for i in idx}
    used = {i: [] for i in idx}          # この窓の縁を実際に動かした値（出どころと根拠はこれだけで出す）
    keys = ("s0", "s1", "z0", "z1")
    for path, m in inputs:
        v0 = V(m)
        if v0 is None:
            continue
        s = Q.sigma_of(m)
        f2 = copy.deepcopy(f)
        _set_path(f2, path, v0 + (s if s else 0.01))
        r2 = face_window_rects(f2)
        if len(r2) != len(base):
            continue
        pkey = ".".join(map(str, path)).replace("windows.", "窓の行").replace("glazing.", "ガラス面")
        for i in idx:
            if "skip" in r2[i]:
                continue
            d = np.array([r2[i][k] - base[i][k] for k in keys])
            if not np.any(np.abs(d) > 1e-9):
                continue
            used[i].append((pkey, m, float(np.abs(d).max()) if s else 0.0))
            if s is None:
                missing[i].append(pkey)
            elif s > 0:
                acc[i] += d * d
    for i in idx:
        sd = np.sqrt(acc[i])
        base[i]["sigma"] = dict(zip(keys, map(float, sd)))
        base[i]["sigma_h"] = None if missing[i] else float(max(sd[0], sd[1]))
        base[i]["sigma_v"] = None if missing[i] else float(max(sd[2], sd[3]))
        base[i]["used"] = {k: meas(m) for k, m, _ in used[i]}
        base[i]["main_input"] = max(used[i], key=lambda x: x[2])[0] if used[i] else None
        if missing[i]:
            base[i]["sigma_missing"] = missing[i]
    return base


def flat_ev(ev: dict | None) -> dict:
    """{key: meas} の入れ子（bays など）を平らに。"""
    out = {}
    for k, m in (ev or {}).items():
        if isinstance(m, dict) and "status" in m:
            out[k] = m
        elif isinstance(m, dict):
            for kk, mm in m.items():
                if isinstance(mm, dict) and "status" in mm:
                    out[f"{k}.{kk}"] = mm
    return out


def feature_quality(meta: dict, extra_h: float | None = None, extra_v: float | None = None,
                    extra_lineage: list[str] | None = None) -> dict:
    """地物の水平・標高の標準偏差（値の ± の大きい方）と等級、出どころ（弱い方）。"""
    sh, sv = [], []
    unknown = False
    lins = list(extra_lineage or [])
    for k, m in flat_ev(meta.get("ev")).items():
        lins.append(Q.lineage_of(m))
        if not Q.is_length(m) and m.get("unit") not in Q.COUNT_UNITS:
            continue
        s = Q.sigma_of(m)
        if s is None:
            if m.get("unit") in (None, "m"):
                unknown = True
            continue
        if k == "part" or k.endswith(".part"):
            sh.append(s)
            sv.append(s)
        elif Q.axis_of(k) == "v":
            sv.append(s)
        else:
            sh.append(s)
    if extra_h is not None:
        sh.append(extra_h)
    if extra_v is not None:
        sv.append(extra_v)
    s_h = max(sh) if sh else None
    s_v = max(sv) if sv else None
    g = Q.worst(Q.grade(s_h, "h"), Q.grade(s_v, "v"))
    if unknown and g != "2500外":
        g = "不明"
    lin = min((x for x in lins if x != "none"), key=lambda x: Q.LINEAGE_RANK[x], default="none")
    return {"sigma_h": s_h, "sigma_v": s_v, "grade": g, "lineage": lin}


# ------------------------------------------------------------------ 体

def snap_polys(polys: list[Polygon], ref: list[Polygon], tol: float) -> list[Polygon]:
    """塊の頂点を、足元（ref）と先に処理した塊の頂点・辺に吸着させる。
    campus-recorder は Blender の足し算を通すために塊を 1〜13 cm 外へ広げているので、それを戻す。"""
    ref_v, ref_e = [], []

    def add_ref(p: Polygon):
        for ring in [p.exterior, *p.interiors]:
            c = np.array(ring.coords)[:-1]
            ref_v.extend(map(tuple, c))
            for i in range(len(c)):
                ref_e.append((c[i], c[(i + 1) % len(c)]))

    for p in ref:
        add_ref(p)
    out = []
    for p in polys:
        rings = []
        for ring in [p.exterior, *p.interiors]:
            c = np.array(ring.coords)[:-1]
            new = []
            for v in c:
                best, bd = None, tol
                for rv in ref_v:
                    d = math.hypot(v[0] - rv[0], v[1] - rv[1])
                    if d < bd:
                        best, bd = rv, d
                if best is None:
                    for a, e in ref_e:
                        ab = e - a
                        L2 = float(ab @ ab)
                        if L2 < 1e-12:
                            continue
                        t = float((v - a) @ ab) / L2
                        if 0 <= t <= 1:
                            q = a + t * ab
                            d = float(np.hypot(*(v - q)))
                            if d < bd:
                                best, bd = tuple(q), d
                new.append(best if best is not None else tuple(v))
            rings.append(new)
        q = Polygon(rings[0], rings[1:]).buffer(0)
        if not q.is_empty:
            out.append(q)
            add_ref(q if isinstance(q, Polygon) else max(q.geoms, key=lambda g: g.area))
    return out


def build_body(b: dict, rule: dict, report: dict):
    H = V(b.get("parapet_top"))
    blocks = b.get("blocks") or [{"id": "main", "polygon": b["footprint"]["polygon"], "top_z": b.get("parapet_top")}]
    gofs = [V(f.get("ground_offset")) for f in b.get("faces", [])]
    base = min([0.0] + [g for g in gofs if g is not None])
    report["base_rel"] = base

    lv = []
    raised = []
    for blk in blocks:
        top = V(blk.get("top_z"), H)
        if top is None:
            report.setdefault("warnings", []).append(f"塊 {blk['id']} の高さが unknown。省いた")
            continue
        poly = Polygon(blk["polygon"]).buffer(0)
        bot = V(blk.get("bottom_z"))
        if bot is not None and bot > 0:
            raised.append((blk, poly, bot, top))
        else:
            lv.append((blk, poly, top))

    # 塊の輪郭を足元の線に吸着（低い塊から。足元に近い物ほど先に基準になる）
    if lv:
        order = sorted(range(len(lv)), key=lambda i: lv[i][2])
        fp = Polygon(b["footprint"]["polygon"]).buffer(0)
        snapped = snap_polys([lv[i][1] for i in order], [fp], tol=0.15)
        if len(snapped) == len(order):
            lv = [(lv[i][0], snapped[j], lv[i][2]) for j, i in enumerate(order)]
    # 別々に測った塊の高さが基準の差より近ければ 1 つにまとめる（面積の重みの平均）。高さ数 cm の壁を作らない
    hm = rule["hmerge"]
    tops = sorted(((t, p.area, blk["id"]) for blk, p, t in lv), reverse=True)
    groups = []
    for t, a, bid in tops:
        if groups and groups[-1][0][0] - t < hm:
            groups[-1].append((t, a, bid))
        else:
            groups.append([(t, a, bid)])
    remap, merges = {}, []
    for g in groups:
        rep_t = round(sum(t * a for t, a, _ in g) / sum(a for _, a, _ in g), 3)
        for t, a, bid in g:
            remap[bid] = rep_t
        if len({t for t, _, _ in g}) > 1:
            merges.append({"blocks": [bid for _, _, bid in g], "tops": [t for t, _, _ in g], "used": rep_t})
    lv = [(blk, p, remap[blk["id"]]) for blk, p, t in lv]
    report["height_merges"] = merges

    # 高さの段ごとの領域（上の段ほど小さい）。基準未満の細い段差は下の段にそろえる（LOD の一般化）
    heights = sorted({round(t, 4) for _, _, t in lv}, reverse=True)
    regions = [unary_union([p for _, p, t in lv if t >= h - 1e-4]) for h in heights]
    s = rule["sliver"]
    tol_s = rule["simplify"]
    jm = {"join_style": "mitre", "mitre_limit": 10.0}
    # 一番下（足元全体）の細い出っ張りと細かいギザギザを落とす
    if regions:
        R = regions[-1]
        R = R.buffer(-s / 2, **jm).buffer(s / 2, **jm).intersection(R.buffer(1e-6, **jm)) if s > 0 else R
        regions[-1] = shapely.set_precision(R.buffer(0).simplify(tol_s, preserve_topology=True).buffer(0), 1e-4)
    for i in range(len(regions) - 2, -1, -1):       # 下から 2 段目 → 一番上
        lower = regions[i + 1]
        R = regions[i].intersection(lower)
        if R.is_empty:
            regions[i] = R
            continue
        snapped = snap_polys(polys_of(R), polys_of(lower), tol=0.15)
        R = unary_union(snapped).simplify(tol_s, preserve_topology=True).buffer(0).intersection(lower)
        ring = lower.difference(R)                   # 下の段だけの所
        opened = ring.buffer(-s / 2, **jm).buffer(s / 2, **jm) if s > 0 else ring
        thin = ring.difference(opened)
        R = unary_union([R, thin]).intersection(lower)
        R = R.buffer(-s / 2, **jm).buffer(s / 2, **jm).intersection(lower) if s > 0 else R
        regions[i] = shapely.set_precision(R.buffer(0), 1e-4)
    report["levels"] = [{"top_rel": h, "area_m2": round(R.area, 1)} for h, R in zip(heights, regions)]

    body = None
    for h, R in zip(heights, regions):
        if R.is_empty:
            continue
        m = prism_xy(R, base, h)
        body = m if body is None else body + m
    for blk, poly, bot, top in raised:
        m = prism_xy(poly, bot, top)
        body = m if body is None else body + m
    level_tops = dict(zip(heights, [[blk["id"] for blk, _, t in lv if t >= h - 1e-4] for h in heights]))
    return body, base, heights, regions, blocks


def collect_parts(b: dict, faces: dict, base: float):
    """部品 → (彫る素片, 置く素片)。彫る素片は世界座標の manifold。
    地面から始まる彫る素片は体の底より下まで伸ばす（底に薄い板を残さない）。"""
    cuts, insts, skipped = [], [], []
    for pi, pt in enumerate(b.get("parts") or []):
        t = pt.get("type")
        name = pt.get("name", t)
        if t in SKIP_PART_TYPES:
            skipped.append({"part": pi, "name": name, "type": t, "why": "LOD3 の付属物でない（塗り分け・植栽・管・飾り）"})
            continue
        furniture = any(w in name for w in FURNITURE_WORDS)
        r = G.part_prims(pt, faces)
        if not r:
            skipped.append({"part": pi, "name": name, "type": t, "why": "形を作れない（unknown か対象外の type）"})
            continue
        frame, cut, prims = r
        axes = G.frame_axes(frame, faces)
        for c in G.cut_list(cut):
            if c["k"] == "prism" and c["z0"] <= 0.05:
                c = {**c, "z0": base - 0.5}
            m = prim_manifold(c, axes)
            if m is not None:
                cuts.append({"part": pi, "name": name, "type": t, "m": m, "axes": axes, "pt": pt})
        ms = [m for m in (prim_manifold(p, axes) for p in prims) if m is not None]
        if ms and furniture:
            skipped.append({"part": pi, "name": name, "type": t, "why": "建物に付かない物（都市設備）"})
        elif ms:
            insts.append({"part": pi, "name": name, "type": t, "ms": ms, "axes": axes, "pt": pt})
    return cuts, insts, skipped


def dims_in_axes(m: M.Manifold, axes) -> list[float]:
    Vx, _ = mesh_of(m)
    if len(Vx) == 0:
        return [0.0, 0.0, 0.0]
    x0, y0, ux, uy, nx, ny = axes
    a = (Vx[:, 0] - x0) * ux + (Vx[:, 1] - y0) * uy
    n = (Vx[:, 0] - x0) * nx + (Vx[:, 1] - y0) * ny
    z = Vx[:, 2]
    return sorted([float(np.ptp(a)), float(np.ptp(n)), float(np.ptp(z))])


# ------------------------------------------------------------------ 窓を壁に

# 窓の出どころ："model"＝棟 JSON の柱割りから並べた窓、"photos"＝写真 1 枚ずつの正面化で検出し、写真どうしで照合した窓
# （recorder facade の out/facade/<棟>/<面>/fused.json。その面に無ければ棟 JSON の窓）
WINDOW_SOURCE = "model"
MIN_PHOTOS = 2
FACADE_DIR = ROOT / "out" / "facade"


def photo_window_rects(bid: str, f: dict, min_n: int = MIN_PHOTOS) -> list[dict] | None:
    """recorder facade fuse の窓（写った写真が min_n 枚以上）を、place_openings の矩形の形にする。無ければ None。
    外壁の読み係が写真ごとの位置合わせを確かめた面（data/reports/<棟>/gaiheki_<面>.json に photo_alignment がある面）だけ。
    窓の並びがくり返す面では、柱の間・階の整数倍ずれて位置合わせされた写真が混じり、検出器だけでは見分けられないため
    （11/17：本部棟 f6 で 5 枚、f4 でも疑い）。"""
    gk = gaiheki_reports(bid).get(f["id"])
    if not gk or not gk.get("photo_alignment"):
        return None
    # 面ごとの決定（data/window_source.json）で photos にした面だけ。重ね画像で確かめた結果を記録してある
    ws = ROOT / "data" / "window_source.json"
    dec = (json.loads(ws.read_text(encoding="utf-8")).get("faces") or {}).get(f"{bid}/{f['id']}") if ws.exists() else None
    if not dec or dec.get("use") != "photos":
        return None
    # 並び方で整えた窓（final.json）があればそれ、無ければ写真どうしで照合しただけの窓（fused.json）
    p = next((q for q in (FACADE_DIR / bid / f["id"] / "final.json", FACADE_DIR / bid / f["id"] / "fused.json") if q.exists()), None)
    if p is None:
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for w in doc.get("windows", []):
        if w["n_photos"] < min_n:
            continue
        sig = {k: w.get(f"sigma_{k}") for k in ("s0", "s1", "z0", "z1")}
        miss = [k for k, v in sig.items() if v is None]
        out.append({"face": f["id"], "src": f"facade:{f['id']}:{w['id']}", "row": "写真の窓", "photo_window": w["id"],
                    "regularized": w.get("regularized"), "source_file": p.name,
                    "s0": w["s0"], "s1": w["s1"], "z0": w["z0"], "z1": w["z1"], "kind": "window",
                    "sigma": sig, "sigma_h": None if miss else max(sig["s0"], sig["s1"]),
                    "sigma_v": None if miss else max(sig["z0"], sig["z1"]), "n_photos": w["n_photos"],
                    "photo_evidence": w.get("evidence") or [], "sigma_missing": miss, "ev": {}})
    return out


def gaiheki_reports(bid: str) -> dict[str, dict]:
    """外壁の読み係（Claude）の報告 data/reports/<棟>/gaiheki_<面>.json を面ごとに。無ければ空。"""
    out = {}
    for p in sorted((ROOT / "data" / "reports" / bid).glob("gaiheki_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # 正式な報告は gaiheki_<面>.json だけ（gaiheki_f6_sonnet.json のような比べるための答えは読まない）
        if d.get("face") and p.stem == f"gaiheki_{d['face']}":
            d["_file"] = str(p.relative_to(ROOT)).replace("\\", "/")
            out[d["face"]] = d
    return out


def material_attrs(parent, gk: dict | None, where: str) -> dict | None:
    """外壁の読み係の材質（where が一致する物）を gen 属性に。出どころは PLATEAU の属性の出どころのコード 802（写真判読）。"""
    if not gk:
        return None
    m = next((x for x in gk.get("materials") or [] if str(x.get("where")) == where), None)
    if m is None:
        return None

    def s(name, val):
        if val:
            a = E(GEN + "stringAttribute", parent, name=name)
            E(GEN + "value", a, val if isinstance(val, str) else json.dumps(val, ensure_ascii=False))

    s("外装材", m.get("material") + (f"（{m['color']}）" if m.get("color") else ""))
    s("外装材.確からしさ", {"high": "高い（複数の写真ではっきり）", "mid": "中", "low": "低い"}.get(m.get("how_sure"), m.get("how_sure")))
    s("外装材.出どころ", f"802 写真判読（外壁の読み係・{(gk.get('made_by') or {}).get('model', 'Claude')}、{gk['_file']}）")
    s("外装材.根拠の写真", ", ".join(m.get("photos") or []))
    if m.get("note"):
        s("外装材.手がかり", m["note"])
    return m


THUMB_DIR = OUT_DIR / "thumbs"      # 窓ごとの根拠の写真の切り抜き（表示画面が読む）。main で --out に合わせる
PUBLIC_FACADE = ROOT / "out" / "facade_public"
THUMB_MAX = 4


def window_thumbs(bid: str, oid: str, srcs: list[dict]) -> list[str]:
    """写真から出した窓の根拠の写真を、公開版（塗りつぶし済み）の面の画像から切り抜き、窓の枠（最終の縁）を赤で描く。
    返り値は THUMB_DIR からの相対パス。写真が多ければ、検出の点の高い順に THUMB_MAX 枚。"""
    import cv2

    out = []
    for x in srcs:
        ev = sorted(x.get("photo_evidence") or [], key=lambda e: -float(e.get("score") or 0))
        for e in ev[:THUMB_MAX]:
            d = PUBLIC_FACADE / bid / x["face"]
            mj = d / f"{e['photo']}.json"
            if not mj.exists():
                continue
            meta = json.loads(mj.read_text(encoding="utf-8"))
            img = cv2.imdecode(np.fromfile(str(d / meta["image"]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            ppm = meta["px_per_m"]

            def px(s_, z_):
                return (int(round((s_ - meta["origin_m"]["s"]) * ppm)), int(round((meta["origin_m"]["z_top"] - z_) * ppm)))
            (x0, y0), (x1, y1) = px(x["s0"], x["z1"]), px(x["s1"], x["z0"])
            pad = int(1.0 * ppm)
            H, W = img.shape[:2]
            cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
            if cx1 - cx0 < 10 or cy1 - cy0 < 10:
                continue
            crop = img[cy0:cy1, cx0:cx1].copy()
            cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (40, 40, 230), 2)
            k = min(1.0, 320 / max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else crop
            rel = f"{oid}_{len(out) + 1}.jpg"
            THUMB_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(THUMB_DIR / rel))
            out.append({"file": rel, "photo": e["photo"]})
    return out


def door_rects(bid: str, f: dict) -> list[dict]:
    """外壁の読み係が読んだ扉（exceptions の kind: door、確からしさ mid 以上）。位置は目盛りの読みなので、
    σ は読みの揺れ（報告の reading_error_m、無ければ 0.3 m）× SIGMA_PER_TOL。"""
    gk = gaiheki_reports(bid).get(f["id"])
    if not gk:
        return []
    err = float(gk.get("reading_error_m") or 0.3) * Q.SIGMA_PER_TOL
    out = []
    for i, e in enumerate(gk.get("exceptions") or []):
        if e.get("kind") != "door" or e.get("how_sure") not in ("high", "mid"):
            continue
        if any(e.get(k) is None for k in ("s0", "s1", "z0", "z1")):
            continue
        out.append({"face": f["id"], "src": f"gaiheki:{f['id']}:exceptions[{i}]", "row": e.get("what", "扉"),
                    "s0": float(e["s0"]), "s1": float(e["s1"]), "z0": float(e["z0"]), "z1": float(e["z1"]), "kind": "door",
                    "claude_read": True, "sigma_h": err, "sigma_v": err, "photo_evidence": e.get("photos") or [],
                    "how_sure": e.get("how_sure"), "note": e.get("note"), "gaiheki_file": gk["_file"], "ev": {}})
    return out


DOOR_WORDS = ("扉", "ドア", "シャッター")


def patch_door_rects(b: dict | None, f: dict) -> list[dict]:
    """棟 JSON の parts のうち、面に描いた扉（type patch、名前に 扉・ドア・シャッター）を扉の矩形に。
    campus-recorder は扉を色の塗り分け（patch）として持っていた（位置・高さ・± は値ごとに根拠つき）。
    σ は along0・along1（横）と z0・z1（縦）の ± から（σ ＝ ± × SIGMA_PER_TOL）。"""
    out = []
    for i, p in enumerate((b or {}).get("parts") or []):
        if p.get("type") != "patch" or p.get("face") != f["id"] or not any(w in (p.get("name") or "") for w in DOOR_WORDS):
            continue
        ms = {k: p.get(k) for k in ("along0", "along1", "z0", "z1")}
        vals = {k: V(m) for k, m in ms.items()}
        if any(v is None for v in vals.values()):
            continue
        sig = {k: Q.sigma_of(m) for k, m in ms.items()}
        miss = [f"parts[{i}].{k}" for k, s in sig.items() if s is None]
        out.append({"face": f["id"], "src": f"parts[{i}]", "row": p.get("name", "扉"),
                    "s0": min(vals["along0"], vals["along1"]), "s1": max(vals["along0"], vals["along1"]),
                    "z0": min(vals["z0"], vals["z1"]), "z1": max(vals["z0"], vals["z1"]), "kind": "door",
                    "sigma_h": None if miss else max(sig["along0"], sig["along1"]),
                    "sigma_v": None if miss else max(sig["z0"], sig["z1"]),
                    "used": {f"扉.{k}": meas(m) for k, m in ms.items()}, "ev": {k: meas(m) for k, m in ms.items()},
                    **({"sigma_missing": miss} if miss else {})})
    return out


def place_openings(walls: list[Facet], faces: dict, rule: dict, report: dict, bid: str | None = None, b: dict | None = None):
    placed, skipped = [], []
    report["window_source"] = {"mode": WINDOW_SOURCE, "photo_faces": [], "min_photos": MIN_PHOTOS}
    for f in faces.values():
        p0, u, nf, L = face_axes(f)
        rects = face_window_rects_sigma(f)
        if bid:
            rects += door_rects(bid, f)
        rects += patch_door_rects(b, f)
        if WINDOW_SOURCE == "photos" and bid:
            pr = photo_window_rects(bid, f)
            if pr is not None:
                # 柱割りの窓だけを写真の窓に替える。ガラス面・扉は棟 JSON のまま
                rects = [r for r in rects if not str(r.get("src", "")).startswith(f"faces.{f['id']}.windows")] + pr
                report["window_source"]["photo_faces"].append({"face": f["id"], "windows": len(pr)})
        for r in rects:
            if "skip" in r:
                skipped.append({**{k: r[k] for k in ("face", "src") if k in r}, "why": r["skip"]})
                continue
            w, h = r["s1"] - r["s0"], r["z1"] - r["z0"]
            if not rule["win"](w, h):
                skipped.append({"face": r["face"], "src": r["src"], "bay": r.get("bay"), "why": f"LOD の基準未満（{w:.2f}×{h:.2f} m）"})
                continue
            # 面の線上の点 → 世界
            A = p0 + u * r["s0"]
            Bp = p0 + u * r["s1"]
            best = None
            for wf in walls:
                if wf.kind != "wall":
                    continue
                n2 = wf.n[:2]
                if float(n2 @ nf) < 0.995:
                    continue
                off = float(np.array([A[0], A[1], 0.0]) @ wf.n) - wf.d    # 正なら壁より外、負なら奥
                if off > 0.4 or off < -25.0:
                    continue
                sa = float(np.array([A[0], A[1], 0.0]) @ wf.e1)
                sb = float(np.array([Bp[0], Bp[1], 0.0]) @ wf.e1)
                rect = sbox(min(sa, sb), r["z0"], max(sa, sb), r["z1"])
                inter = wf.poly.intersection(rect).area / rect.area
                if inter < 0.6:
                    continue
                score = (abs(off) > 0.4, -inter, abs(off))
                if best is None or score < best[0]:
                    best = (score, wf, rect, off, inter)
            if best is None:
                skipped.append({"face": r["face"], "src": r["src"], "bay": r.get("bay"), "why": "置ける壁が無い（彫り込みの中か、壁の外）"})
                continue
            _, wf, rect, off, inter = best
            wf.meta.setdefault("rects", []).append({**r, "rect": rect, "offset": off, "overlap": inter})
    # 壁ごとに：重なる矩形をまとめ、壁から引く
    for wf in walls:
        rs = wf.meta.get("rects")
        if not rs:
            continue
        inner = wf.poly.buffer(-0.02, join_style="mitre")
        groups = []
        for r in rs:
            g = r["rect"].intersection(inner)
            if g.area < 1e-4:
                continue
            hit = [gr for gr in groups if gr["geom"].intersects(g)]
            if hit:
                geom = unary_union([g] + [gr["geom"] for gr in hit])
                src = sum((gr["src"] for gr in hit), [r])
                groups = [gr for gr in groups if gr not in hit]
                groups.append({"geom": geom, "src": src})
            else:
                groups.append({"geom": g, "src": [r]})
        holes = unary_union([gr["geom"] for gr in groups]) if groups else None
        if holes is None or holes.is_empty:
            continue
        wf.poly = orient_all(wf.poly.difference(holes))
        wf.meta["openings"] = []
        for gr in groups:
            for p in polys_of(orient_all(gr["geom"])):
                kind = "door" if all(x.get("kind") == "door" for x in gr["src"]) else "window"
                wf.meta["openings"].append({"poly": p, "kind": kind, "src": gr["src"]})
                placed.append({"face": gr["src"][0]["face"], "n_src": len(gr["src"]), "area": round(p.area, 2),
                               "recessed": any(x["offset"] < -0.4 for x in gr["src"])})
    report["openings_placed"] = len(placed)
    report["openings_recessed"] = sum(1 for p in placed if p["recessed"])
    report["openings_skipped"] = skipped
    return placed


# ------------------------------------------------------------------ T 字の継ぎ目を直す

def weld(rings: list[np.ndarray], tol=1e-3) -> list[np.ndarray]:
    """tol 以内の点を 1 つにまとめる（別々に計算した同じ角が 0.1 mm ずれるのを直す）。"""
    reps, grid = [], defaultdict(list)

    def rep(p):
        k = tuple((p // tol).astype(int))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for i in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if np.linalg.norm(reps[i] - p) <= tol:
                            return reps[i]
        reps.append(p)
        grid[k].append(len(reps) - 1)
        return p

    out = []
    for R in rings:
        new = [rep(np.asarray(p, float)) for p in R]
        keep = [new[0]] + [q for a, q in zip(new, new[1:]) if np.linalg.norm(q - a) > 0]
        if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) == 0:
            keep = keep[:-1]
        out.append(despike(keep))
    return out


def despike(pts: list) -> np.ndarray:
    """まとめた後に残る「行って戻る」辺（幅 0 の切れ込み・とげ）を除く。"""
    pts = [np.asarray(p, float) for p in pts]
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            if np.array_equal(a, c):
                # a → b → a：b と 2 つ目の a を消す
                idx = sorted({i, (i + 1) % n}, reverse=True)
                for j in idx:
                    pts.pop(j)
                changed = True
                break
            u, v = b - a, c - b
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu == 0 or nv == 0:
                pts.pop(i)
                changed = True
                break
            if np.linalg.norm(np.cross(u, v)) < 1e-6 * nu * nv and float(u @ v) < 0:
                pts.pop(i)                    # 真後ろへ折り返す点
                changed = True
                break
    return np.array(pts)


def fix_tjunctions(rings: list[np.ndarray], tol=1e-3) -> list[np.ndarray]:
    rings = weld(rings, tol)
    allv = np.unique(np.vstack(rings), axis=0)
    cell = 2.0
    grid = defaultdict(list)
    for i, v in enumerate(allv):
        grid[tuple((v // cell).astype(int))].append(i)
    out = []
    for ring in rings:
        new = []
        n = len(ring)
        for i in range(n):
            p, q = ring[i], ring[(i + 1) % n]
            new.append(p)
            d = q - p
            L2 = float(d @ d)
            if L2 < 1e-12:
                continue
            lo, hi = np.minimum(p, q) - tol, np.maximum(p, q) + tol
            cands = set()
            for cx in range(int(lo[0] // cell), int(hi[0] // cell) + 1):
                for cy in range(int(lo[1] // cell), int(hi[1] // cell) + 1):
                    for cz in range(int(lo[2] // cell), int(hi[2] // cell) + 1):
                        cands.update(grid.get((cx, cy, cz), ()))
            ins = []
            Ld = math.sqrt(L2)
            for ci in cands:
                v = allv[ci]
                t = float((v - p) @ d) / L2
                if t * Ld <= tol or (1 - t) * Ld <= tol:
                    continue
                if np.linalg.norm(p + t * d - v) < tol:
                    ins.append((t, v))
            ins.sort(key=lambda x: x[0])
            new += [v for _, v in ins]
        arr = np.array(new)
        # 連続するほぼ同じ点を除く（検査の丸め 1e-4 より細かい差は同じ点）
        keep = [0] + [i for i in range(1, len(arr)) if np.linalg.norm(arr[i] - arr[i - 1]) > tol]
        arr = arr[keep]
        if len(arr) > 1 and np.linalg.norm(arr[0] - arr[-1]) <= tol:
            arr = arr[:-1]
        out.append(arr)
    return out


# ------------------------------------------------------------------ 1 棟

def _closed_ok(body) -> bool:
    """体の面（平面ごと、T 字を直した後）が、どの辺もちょうど 2 枚の面で逆向きに共有されているか。"""
    Vx, T = mesh_of(body)
    rings = []
    for f in facets_from_mesh(Vx, T):
        rings += [f.to3d(np.array(f.poly.exterior.coords)[:-1])] + [f.to3d(np.array(r.coords)[:-1]) for r in f.poly.interiors]
    return check_closed(fix_tjunctions(rings))["unmatched_edges"] == 0


def _classify(Vx, T, facets):
    """面の種類：壁・屋根・底・軒下（下向きで底でない）・床（上向きで上に体がある）。"""
    zmin = float(Vx[:, 2].min())
    for f in facets:
        nz = f.n[2]
        if abs(nz) < 0.02:
            f.kind = "wall"
        elif nz > 0:
            c = f.poly.representative_point()
            P = f.to3d([(c.x, c.y)])[0]
            hits = ray_up_hits(Vx, T, P + np.array([0, 0, 1e-3]))
            f.kind = "floor" if hits else "roof"
        else:
            zc = f.to3d([tuple(f.poly.representative_point().coords)[0]])[0][2]
            f.kind = "ground" if abs(zc - zmin) < 0.05 else "ceiling"


def build_lod2_facets(b: dict, roof_json: dict | None) -> tuple[list, dict]:
    """棟 JSON から LOD2（塊と屋根の形。彫り込み・開口・付属物なし）。塊の段差の取り方は LOD3.0 と同じ基準。"""
    rule = LOD_RULES["3.0"]
    report = {"id": b["id"], "warnings": []}
    faces = {f["id"]: f for f in b.get("faces", []) if f.get("from") and f.get("to")}
    body, base, heights, regions, blocks = build_body(b, rule, report)
    Vx, T = mesh_of(body)
    facets = facets_from_mesh(Vx, T)
    _classify(Vx, T, facets)
    attach_evidence(b, facets, faces, blocks, [], roof_json)
    report["volume_m3"] = round(body.volume(), 1)
    return facets, report


def write_lod2(bel, b: dict, facets, geo: Geo, evidence: dict) -> list:
    """PLATEAU の Building 要素 bel に LOD2（lod2Solid と、lod2MultiSurface の境界面）を足す。LOD3 を足した後に呼ぶ。
    並びは CityGML 2.0 の決まり：lod2Solid は lod1 の後、境界面は他の boundedBy の前（付属物より後）。"""
    bid = b["id"]
    items = [(f, [f.to3d(np.array(f.poly.exterior.coords)[:-1])] + [f.to3d(np.array(r.coords)[:-1]) for r in f.poly.interiors])
             for f in facets]
    fixed = fix_tjunctions([r for _, rs in items for r in rs])
    k = 0
    bounded, refs, rings_all = [], [], []
    for i, (f, rs) in enumerate(items):
        rs2 = fixed[k:k + len(rs)]
        k += len(rs)
        rings_all += rs2
        bb = E(B + "boundedBy")
        sid = "surf2_" + uid(bid, "lod2", i)
        sf = E(B + SURF_TAG[f.kind], bb, **{GM + "id": sid})
        meta = {kk: f.meta.get(kk) for kk in ("face", "face_name", "note", "ev") if f.meta.get(kk)}
        qual = feature_quality(meta)
        gen_attrs(sf, meta, qual)
        evidence[sid] = {"kind": f.kind, "lod": "2", "quality": qual, **{kk: f.meta.get(kk) for kk in ("face", "note") if f.meta.get(kk)}}
        ms = E(B + "lod2MultiSurface", sf)
        sm = E(GM + "surfaceMember", E(GM + "MultiSurface", ms))
        pid = "poly2_" + uid(bid, "lod2", i)
        gml_polygon(sm, rs2, geo, pid)
        refs.append(pid)
        bounded.append(bb)
    ls = E(B + "lod2Solid")
    cs = E(GM + "CompositeSurface", E(GM + "exterior", E(GM + "Solid", ls)))
    for pid in refs:
        E(GM + "surfaceMember", cs, **{XL + "href": "#" + pid})
    kids = list(bel)
    pos = 0
    for i, ch in enumerate(kids):
        if etree.QName(ch).namespace == NS["bldg"] and etree.QName(ch).localname in (
                "measuredHeight", "storeysAboveGround", "storeysBelowGround", "lod0FootPrint", "lod0RoofEdge",
                "lod1Solid", "lod1MultiSurface", "lod1TerrainIntersection"):
            pos = i + 1
    bel.insert(pos, ls)
    kids = list(bel)
    first = next((i for i, ch in enumerate(kids) if etree.QName(ch).localname == "boundedBy"), None)
    if first is None:
        first = next(i for i, ch in enumerate(kids) if etree.QName(ch).localname == "lod3Solid")
    for j, bb in enumerate(bounded):
        bel.insert(first + j, bb)
    # データ品質属性：LOD2 の形の出典と、LOD の区分（京都 2025 の公式データと同じく 2.0 と 3.x を並べる）
    dq = bel.find(f".//{URO}DataQualityAttribute")
    if dq is not None:
        cs_g = "../../codelists/DataQualityAttribute_geometrySrcDesc.xml"
        g3 = next((i for i, ch in enumerate(dq) if etree.QName(ch).localname == "geometrySrcDescLod3"), None)
        if g3 is not None:
            for j, code in enumerate(("000", "801", "901")):
                dq.insert(g3 + j, E(URO + "geometrySrcDescLod2", None, code, codeSpace=cs_g))
        lt = next((i for i, ch in enumerate(dq) if etree.QName(ch).localname == "lodType"), None)
        if lt is not None:
            dq.insert(lt, E(URO + "lodType", None, "2.0", codeSpace="../../codelists/Building_lodType.xml"))
    return rings_all


def build_building(b: dict, lod: str, roof_json: dict | None):
    rule = LOD_RULES[lod]
    report = {"id": b["id"], "name": b["name"], "lod": lod, "warnings": []}
    faces = {f["id"]: f for f in b.get("faces", []) if f.get("from") and f.get("to")}
    body, base, heights, regions, blocks = build_body(b, rule, report)

    cuts, insts, skipped_parts = collect_parts(b, faces, base)
    report["parts_skipped"] = skipped_parts
    used_cuts = []
    for c in cuts:
        m = c["m"]
        eff = m ^ body
        dims = dims_in_axes(eff, c["axes"]) if not eff.is_empty() else [0, 0, 0]
        ok = rule["cut"](dims)
        c["dims"] = dims
        rec = {"part": c["part"], "name": c["name"], "dims_m": [round(x, 2) for x in dims], "used": bool(ok)}
        report.setdefault("cuts", []).append(rec)
        if ok:
            c["m_ext"] = m
            used_cuts.append(c)
    if used_cuts:
        body0 = body
        body = body0 - M.Manifold.batch_boolean([c["m_ext"] for c in used_cuts], M.OpType.Add)
        # 彫り込みで 2 つの塊が辺だけで接する形（1 本の辺を 4 枚の面が共有）になると、CityGML の立体として閉じない。
        # その時は、閉じるまで彫り込みを 1 つずつ外して試す（外した物と理由は報告に残す）
        if not _closed_ok(body):
            for c in sorted(used_cuts, key=lambda c: min(c["dims"])):
                trial = [x for x in used_cuts if x is not c]
                tb = body0 - M.Manifold.batch_boolean([x["m_ext"] for x in trial], M.OpType.Add) if trial else body0
                if _closed_ok(tb):
                    used_cuts, body = trial, tb
                    for rec in report.get("cuts", []):
                        if rec["part"] == c["part"] and rec["used"] and rec["dims_m"] == [round(x, 2) for x in c["dims"]]:
                            rec["used"] = False
                            rec["why"] = "使うと 2 つの塊が辺だけで接し、立体が閉じなくなるので外した"
                    break
    report["volume_m3"] = round(body.volume(), 1)
    report["genus"] = body.genus()

    Vx, T = mesh_of(body)
    facets = facets_from_mesh(Vx, T)
    _classify(Vx, T, facets)

    placed = place_openings(facets, faces, rule, report, b.get("id"), b)

    # 付属物：体と重なる所を引く
    inst_out = []
    for it in insts:
        m = M.Manifold.batch_boolean(it["ms"], M.OpType.Add) if len(it["ms"]) > 1 else it["ms"][0]
        m = m - body
        if m.is_empty() or m.volume() < 1e-3:
            report["parts_skipped"].append({"part": it["part"], "name": it["name"], "type": it["type"], "why": "体の中に収まる"})
            continue
        dims = dims_in_axes(m, it["axes"])
        if not rule["inst"](dims):
            report["parts_skipped"].append({"part": it["part"], "name": it["name"], "type": it["type"],
                                            "why": f"LOD の基準未満（{dims[0]:.2f}×{dims[1]:.2f}×{dims[2]:.2f} m）"})
            continue
        iv, it_t = mesh_of(m)
        it["facets"] = facets_from_mesh(iv, it_t)
        it["dims"] = dims
        inst_out.append(it)
    report["installations"] = [{"part": it["part"], "name": it["name"], "type": it["type"],
                                "dims_m": [round(x, 2) for x in it["dims"]]} for it in inst_out]

    # 根拠：壁↔面、屋根↔塊、軒下↔彫り込み
    attach_evidence(b, facets, faces, blocks, used_cuts, roof_json)
    return facets, inst_out, report


def ray_up_hits(Vx, T, P) -> bool:
    """P から真上への光線が三角形に当たるか（Möller–Trumbore を z 方向に特化）。"""
    A, Bv, C = Vx[T[:, 0]], Vx[T[:, 1]], Vx[T[:, 2]]
    nz = np.cross(Bv - A, C - A)
    flat = np.abs(nz[:, 2]) > 1e-6 * np.maximum(np.linalg.norm(nz, axis=1), 1e-12)   # 鉛直の三角形は真上の光線に当たらない
    A, Bv, C = A[flat], Bv[flat], C[flat]
    x, y = P[0], P[1]
    # 2D の点 (x, y) が三角形の中か（重心座標）
    v0 = C[:, :2] - A[:, :2]
    v1 = Bv[:, :2] - A[:, :2]
    v2 = np.array([x, y]) - A[:, :2]
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-12
    u = np.where(ok, (d11 * d20 - d01 * d21) / np.where(ok, den, 1), -1)
    v = np.where(ok, (d00 * d21 - d01 * d20) / np.where(ok, den, 1), -1)
    inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    z = A[:, 2] + u * (C[:, 2] - A[:, 2]) + v * (Bv[:, 2] - A[:, 2])
    return bool(np.any(inside & (z > P[2] + 1e-4)))


def attach_evidence(b, facets, faces, blocks, used_cuts, roof_json):
    for f in facets:
        if f.kind == "wall":
            best = None
            for fc in faces.values():
                p0, u, nf, L = face_axes(fc)
                if float(f.n[:2] @ nf) < 0.995:
                    continue
                off = abs(float(np.array([p0[0], p0[1], 0.0]) @ f.n) - f.d)
                if off > 0.4:
                    continue
                s0 = float(np.array([p0[0], p0[1], 0]) @ f.e1)
                s1 = float(np.array([*(p0 + u * L), 0]) @ f.e1)
                lo, hi = min(s0, s1), max(s0, s1)
                bx = f.poly.bounds
                ov = min(hi, bx[2]) - max(lo, bx[0])
                if ov > 0.5 and (best is None or ov > best[0]):
                    best = (ov, fc)
            if best:
                fc = best[1]
                f.meta["face"] = fc["id"]
                f.meta["face_name"] = fc.get("name")
                f.meta["ev"] = {k: meas(fc.get(k)) for k in ("length", "top_z", "ground_offset") if meas(fc.get(k))}
                bays = fc.get("bays") or {}
                f.meta["ev"].update({f"bays.{k}": meas(bays.get(k)) for k in ("count", "pitch") if meas(bays.get(k))})
                if "top_z" not in f.meta["ev"]:
                    # 面に上端の記録が無い：壁の上端に合う塊の高さ、無ければ棟の笠木
                    bev = blocks_ev(f, blocks)
                    if bev:
                        f.meta["ev"].update(bev)
                    else:
                        ztop = float(f.to3d(np.array(f.poly.exterior.coords)[:-1])[:, 2].max())
                        if V(b.get("parapet_top")) is not None and abs(V(b["parapet_top"]) - ztop) < 1.0:
                            f.meta["ev"]["parapet_top"] = meas(b["parapet_top"])
            else:
                cut = cut_for(f, used_cuts)
                if cut:
                    f.meta.update(cut)
                else:
                    f.meta["note"] = "塊の段差の壁（高さは塊の top_z、位置は塊の輪郭）"
                    f.meta["ev"] = blocks_ev(f, blocks)
        elif f.kind == "roof":
            f.meta["ev"] = blocks_ev(f, blocks)
            if not f.meta["ev"]:
                # 塊に当たらない屋根：空に開いた彫り込みの底か、棟の笠木の高さを根拠にする
                zc = float(f.to3d([tuple(f.poly.representative_point().coords)[0]])[0][2])
                cut = cut_for(f, used_cuts)
                if cut:
                    f.meta.update(cut)
                    f.meta["note"] = f"空に開いた彫り込みの底（高さ {zc:.2f} m）"
                elif V(b.get("parapet_top")) is not None and abs(V(b["parapet_top"]) - zc) < 1.0:
                    f.meta["ev"] = {"parapet_top": meas(b["parapet_top"])}
                else:
                    f.meta["note"] = f"高さ {zc:.2f} m の屋根。合う塊・笠木の記録が無い"
            f.meta["roof_src"] = roof_source(roof_json)
        elif f.kind in ("ceiling", "floor"):
            cut = cut_for(f, used_cuts)
            if cut:
                f.meta.update(cut)
        elif f.kind == "ground":
            f.meta["note"] = "建物の底（面の前の地面の高さのうち一番低い所）"
            f.meta["ev"] = {k: meas(b.get(k)) for k in ("ground_z",) if meas(b.get(k))}


def blocks_ev(f, blocks):
    """屋根：その点を含み高さの合う塊。壁：壁の線の近くに輪郭があり、上端の高さが合う塊。"""
    out = {}
    if f.kind == "wall":
        ext = f.to3d(np.array(f.poly.exterior.coords)[:-1])
        ztop = float(ext[:, 2].max())
        line = shapely.LineString([ext[np.argmin(ext @ f.e1)][:2], ext[np.argmax(ext @ f.e1)][:2]])
        for blk in blocks:
            top = blk.get("top_z")
            if V(top) is None:
                continue
            if Polygon(blk["polygon"]).exterior.distance(line.interpolate(0.5, normalized=True)) < 1.0 and abs(V(top) - ztop) < 1.0:
                out[f"blocks.{blk['id']}.top_z"] = meas(top)
        return out
    P = f.to3d([tuple(f.poly.representative_point().coords)[0]])[0]
    z = float(P[2])
    for blk in blocks:
        top = blk.get("top_z")
        if Polygon(blk["polygon"]).buffer(0.5).contains(shapely.Point(P[0], P[1])) and V(top) is not None and abs(V(top) - z) < 0.6:
            out[f"blocks.{blk['id']}.top_z"] = meas(top)
    return out


def cut_for(f, used_cuts):
    c = f.poly.representative_point()
    P = f.to3d([(c.x, c.y)])[0]
    for cu in used_cuts:
        bb = cu["m_ext"].bounding_box()
        if all(bb[i] - 0.05 <= P[i] <= bb[i + 3] + 0.05 for i in range(3)):
            pt = cu["pt"]
            return {"cut": cu["name"], "ev": {"part": {k: pt.get(k) for k in ("status", "evidence", "method", "tol") if pt.get(k)}}}
    return None


def roof_source(roof_json):
    if not roof_json:
        return {"how": "屋上の輪郭は塊の輪郭（PLATEAU の足元＝航空測量）", "aerial": None}
    c = (roof_json.get("roof_edge_offset_candidates") or [{}])[0]
    return {"how": "屋上の輪郭は塊の輪郭（PLATEAU の足元＝航空測量）。国土地理院の航空写真（シームレス z18）で屋上の縁を照合",
            "aerial": {"lean_m": roof_json.get("lean_m"), "confidence": c.get("confidence"), "ambiguous": roof_json.get("ambiguous"),
                       "source": roof_json.get("roof_edge_source")}}


# ------------------------------------------------------------------ CityGML

def E(tag, parent=None, text=None, **attrib):
    el = etree.SubElement(parent, tag, attrib) if parent is not None else etree.Element(tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def poslist(P: np.ndarray, geo: Geo) -> str:
    Q = geo.latlonh(np.vstack([P, P[:1]]))
    return " ".join(f"{a:.10f} {o:.10f} {h:.4f}" for a, o, h in Q)


def gml_polygon(parent, rings3d: list[np.ndarray], geo: Geo, pid: str) -> list[str]:
    pol = E(GM + "Polygon", parent, **{GM + "id": pid})
    rids = []
    for i, R in enumerate(rings3d):
        wrap = E(GM + ("exterior" if i == 0 else "interior"), pol)
        rid = f"line_{pid[5:]}_{i}"
        lr = E(GM + "LinearRing", wrap, **{GM + "id": rid})
        E(GM + "posList", lr, poslist(R, geo))
        rids.append(rid)
    return rids


def gen_attrs(parent, meta: dict, qual: dict | None = None):
    """根拠を gen 属性に。地物の精度（等級・水平σ・標高σ・出どころ）と、値ごとに 値・±・出どころ・根拠の写真。"""
    def s(name, val):
        if val is None or val == "" or val == []:
            return
        a = E(GEN + "stringAttribute", parent, name=name)
        E(GEN + "value", a, val if isinstance(val, str) else json.dumps(val, ensure_ascii=False))

    if qual:
        s("精度.等級", Q.GRADE_JA[qual["grade"]])
        if qual.get("sigma_h") is not None:
            s("精度.水平σ", f"{qual['sigma_h']:.2f} m")
        if qual.get("sigma_v") is not None:
            s("精度.標高σ", f"{qual['sigma_v']:.2f} m")
        s("出どころ（いちばん弱い値）", Q.LINEAGE_JA[qual["lineage"]])
    for k in ("face", "face_name", "cut", "note"):
        s({"face": "出典の面", "face_name": "面の名前", "cut": "彫り込みの部品", "note": "メモ"}[k], meta.get(k))
    if meta.get("photo_evidence"):
        s("根拠の写真", ", ".join(meta["photo_evidence"]))
    for k, m in flat_ev(meta.get("ev")).items():
        unit = m.get("unit") or ""
        if m.get("value") is not None:
            s(f"{k}.値", f"{m.get('value')} {unit}".strip())
        if m.get("tol") is not None:
            s(f"{k}.±", f"{m.get('tol')} {unit}".strip())
        s(f"{k}.出どころ", Q.LINEAGE_JA[Q.lineage_of(m)])
        ev = m.get("evidence")
        if ev:
            s(f"{k}.根拠", ", ".join(ev) if isinstance(ev, list) else str(ev))
    if meta.get("roof_src"):
        s("屋上の出典", meta["roof_src"]["how"])
        if meta["roof_src"].get("aerial"):
            s("航空写真での照合", meta["roof_src"]["aerial"])
        if meta["roof_src"].get("lean"):
            s("航空写真の倒れ込み", meta["roof_src"]["lean"])


SURF_TAG = {"wall": "WallSurface", "roof": "RoofSurface", "ground": "GroundSurface",
            "ceiling": "OuterCeilingSurface", "floor": "OuterFloorSurface"}


RULER_TEXT = ("各地物の「精度.等級」は、棟 JSON の値の ± を標準偏差とみなし（控えめな読み）、作業規程の準則の地図情報レベルの"
              "標準偏差（PLATEAU 標準製品仕様書 6.3.3：500＝水平 0.25 m・標高 0.25 m、1000＝0.70・0.33、2500＝1.75・0.66）と比べた"
              "もの。水平と標高の悪い方を採る。棟の記録の窓は柱割り・幅・上下端の ± を縁の位置に伝えた値。写真から出した窓は、"
              "写真 1 枚ずつ面に正面化して縁を測り、写真どうしのばらつきから出した σ。建物の中での位置の確かさであり、"
              "地図上の位置は PLATEAU の足元（地図情報レベル 2500）に従う。屋上・底の輪郭も足元に従うので 2500 より良くならない。"
              "± が実際に当たるか（独立に測った正解が ± に入る割合）はまだ確かめていない。")


def write_lod3(bel, b: dict, facets, insts, geo: Geo, evidence: dict, lod: str, textured: bool = False,
               grades: dict | None = None, wall_textured: bool = False):
    """PLATEAU の Building 要素 bel に LOD3 を足す。"""
    bid = b["id"]
    grades = grades if grades is not None else defaultdict(list)
    gks = gaiheki_reports(bid)          # 外壁の読み係の報告（材質）
    mats_written: list = []
    # 挿入位置：lod1Solid（か lod0 系）の直後、uro:* の前
    kids = list(bel)
    idx = 0
    for i, k in enumerate(kids):
        if etree.QName(k).namespace == NS["bldg"]:
            idx = i + 1
    new = []
    solid_refs = []
    rings_all = []
    tex_polys = []     # 壁と開口の多角形（面ごとに写真の質感を貼る時に使う）
    # まず全ての輪を集めて T 字を直す
    items = []        # (owner, kind, rings)
    for f in facets:
        rings = [f.to3d(np.array(f.poly.exterior.coords)[:-1])] + [f.to3d(np.array(r.coords)[:-1]) for r in f.poly.interiors]
        items.append(("surf", f, rings))
        for o in f.meta.get("openings", []):
            items.append(("open", (f, o), [f.to3d(np.array(o["poly"].exterior.coords)[:-1])]))
    flat = [r for _, _, rs in items for r in rs]
    fixed = fix_tjunctions(flat)
    k = 0
    for i, (kind, own, rs) in enumerate(items):
        n = len(rs)
        items[i] = (kind, own, fixed[k:k + n])
        k += n
    rings_all = [r for _, _, rs in items for r in rs]

    # 付属物
    for it in insts:
        oi = E(B + "outerBuildingInstallation")
        bi = E(B + "BuildingInstallation", oi, **{GM + "id": "inst_" + uid(bid, "inst", it["part"])})
        pev = {}
        if it["pt"].get("status"):
            pev["part"] = {kk: it["pt"].get(kk) for kk in ("status", "evidence", "tol") if it["pt"].get(kk)}
        for kk, vv in it["pt"].items():          # 部品の中の値ごとの根拠（幅・高さ・段数など）
            if meas(vv):
                pev[kk] = meas(vv)
        meta = {"note": f"{it['type']}：{it['name']}", "ev": pev}
        qual = feature_quality(meta)
        gen_attrs(bi, meta, qual)
        grades["installation"].append(qual["grade"])
        evidence[bi.get(GM + "id")] = {"kind": "installation", "part_index": it["part"], "name": it["name"], "type": it["type"],
                                         "status": it["pt"].get("status"), "evidence": it["pt"].get("evidence"),
                                         "method": it["pt"].get("method"), "quality": qual}
        g = E(B + "lod3Geometry", bi)
        ms = E(GM + "MultiSurface", g)
        frs = []
        for f in it["facets"]:
            frs.append([f.to3d(np.array(f.poly.exterior.coords)[:-1])] + [f.to3d(np.array(r.coords)[:-1]) for r in f.poly.interiors])
        flat_i = fix_tjunctions([r for rs in frs for r in rs])
        kk = 0
        for j, rs in enumerate(frs):
            sm = E(GM + "surfaceMember", ms)
            gml_polygon(sm, flat_i[kk:kk + len(rs)], geo, "poly_" + uid(bid, "inst", it["part"], j))
            kk += len(rs)
        new.append(oi)

    # 境界面と開口
    roof_rings = []
    opens_of = defaultdict(list)
    for j, (kind2, own2, rs2) in enumerate(items):
        if kind2 == "open":
            opens_of[id(own2[0])].append(j)
    for i, (kind, own, rs) in enumerate(items):
        if kind != "surf":
            continue
        f = own
        sid = f"{SURF_TAG[f.kind][:4].lower()}_" + uid(bid, "surf", i)
        bb = E(B + "boundedBy")
        sf = E(B + SURF_TAG[f.kind], bb, **{GM + "id": sid})
        if f.kind in ("roof", "ground"):
            # 屋上の輪郭・底の輪郭は PLATEAU の足元（地図情報レベル 2500）に従う
            qual = feature_quality(f.meta, extra_h=Q.FOOTPRINT_SIGMA_H, extra_lineage=["public"])
        else:
            qual = feature_quality(f.meta)
        gen_attrs(sf, f.meta, qual)
        if f.kind == "wall":
            mt = material_attrs(sf, gks.get(f.meta.get("face")), "wall")
            if mt:
                mats_written.append(f.meta.get("face"))
        grades[f.kind].append(qual["grade"])
        evidence[sid] = {"kind": f.kind, "quality": qual,
                         **{kk: f.meta.get(kk) for kk in ("face", "face_name", "cut", "note", "ev", "roof_src") if f.meta.get(kk)}}
        ms = E(B + "lod3MultiSurface", sf)
        msm = E(GM + "MultiSurface", ms)
        sm = E(GM + "surfaceMember", msm)
        pid = "poly_" + uid(bid, "surf", i)
        rids = gml_polygon(sm, rs, geo, pid)
        solid_refs.append(pid)
        if f.kind == "wall" and f.meta.get("face"):
            tex_polys.append({"face": f.meta["face"], "pid": pid, "rids": rids, "rings": rs})
        if f.kind == "roof":
            roof_rings.append({"pid": pid, "rids": rids, "rings": rs})
        # 開口
        for j in opens_of[id(f)]:
            _, own2, rs2 = items[j]
            o = own2[1]
            op = E(B + "opening", sf)
            tag = "Door" if o["kind"] == "door" else "Window"
            oid = ("door_" if tag == "Door" else "win_") + uid(bid, "open", j)
            w = E(B + tag, op, **{GM + "id": oid})
            src = o["src"][0]
            if src.get("claude_read"):
                wmeta = {"face": src["face"], "note": f"{src.get('row')}（外壁の読み係が正面化した写真の目盛りで読んだ。確からしさ {src.get('how_sure')}。"
                                                       f"{src.get('note') or ''}）", "photo_evidence": src.get("photo_evidence"), "ev": {}}
                qual = feature_quality(wmeta, extra_h=src["sigma_h"], extra_v=src["sigma_v"], extra_lineage=["claude_read"])
                gen_attrs(w, wmeta, qual)
                grades[tag.lower()].append(qual["grade"])
                evidence[oid] = {"kind": tag.lower(), "quality": qual, "source": "gaiheki",
                                 "sources": [{kk: x.get(kk) for kk in ("face", "src", "s0", "s1", "z0", "z1", "how_sure", "gaiheki_file",
                                                                       "photo_evidence", "offset")} for x in o["src"]]}
                lm = E(B + "lod3MultiSurface", w)
                mm = E(GM + "MultiSurface", lm)
                smm = E(GM + "surfaceMember", mm)
                pid2 = "poly_" + uid(bid, "open", j)
                rids2 = gml_polygon(smm, rs2, geo, pid2)
                if f.meta.get("face"):
                    tex_polys.append({"face": f.meta["face"], "pid": pid2, "rids": rids2, "rings": rs2})
                solid_refs.append(pid2)
                continue
            if src.get("photo_window"):
                # 写真から出した窓：σ は写真どうしのばらつき、根拠は写真と面の画像の画素の範囲
                n = max(x.get("n_photos", 0) for x in o["src"])
                photos = sorted({e["photo"] for x in o["src"] for e in x.get("photo_evidence", [])})
                wmeta = {"face": src["face"],
                         "note": f"写真 {n} 枚の正面化で窓を検出し、写真どうしで照合（{', '.join(x['photo_window'] for x in o['src'])}）",
                         "photo_evidence": photos, "ev": {}}
                hs = [x.get("sigma_h") for x in o["src"]]
                vs = [x.get("sigma_v") for x in o["src"]]
                unk = any(h is None for h in hs) or any(v is None for v in vs)
                qual = feature_quality(wmeta, extra_h=None if unk else max(hs), extra_v=None if unk else max(vs),
                                       extra_lineage=["photo_multi"])
                if unk:
                    qual["grade"] = "不明"
                else:
                    qual["edge_sigma_h"], qual["edge_sigma_v"] = max(hs), max(vs)
                gen_attrs(w, wmeta, qual)
                grades[tag.lower()].append(qual["grade"])
                evidence[oid] = {"kind": tag.lower(), "quality": qual, "source": "photos",
                                 "sources": [{kk: x.get(kk) for kk in ("face", "src", "s0", "s1", "z0", "z1", "sigma", "n_photos",
                                                                       "photo_evidence", "offset")} for x in o["src"]],
                                 "thumbs": window_thumbs(bid, oid, o["src"])}
                lm = E(B + "lod3MultiSurface", w)
                mm = E(GM + "MultiSurface", lm)
                smm = E(GM + "surfaceMember", mm)
                pid2 = "poly_" + uid(bid, "open", j)
                rids2 = gml_polygon(smm, rs2, geo, pid2)
                if f.meta.get("face"):
                    tex_polys.append({"face": f.meta["face"], "pid": pid2, "rids": rids2, "rings": rs2})
                solid_refs.append(pid2)
                continue
            # 根拠は「この窓の縁を実際に動かした値」だけ（使われない柱型の幅などは出さない）
            used_ev = {}
            for x in o["src"]:
                used_ev.update(x.get("used") or {})
            wmeta = {"face": src["face"], "note": f"{src.get('row')}（{src['src']}" + (f"、柱間 {src['bay']}" if src.get("bay") else "") + "）"
                     + (f"。{len(o['src'])} 個の矩形が重なったので 1 つにした" if len(o["src"]) > 1 else "")
                     + ("。面の線の奥の壁に置いた" if src.get("offset", 0) < -0.4 else "")
                     + (f"。縁の位置にいちばん効く値：{src['main_input']}" if src.get("main_input") else ""),
                     "ev": used_ev or {kk: vv for kk, vv in (src.get("ev") or {}).items() if isinstance(vv, dict)}}
            hs = [x.get("sigma_h") for x in o["src"]]
            vs = [x.get("sigma_v") for x in o["src"]]
            unk = any(h is None for h in hs) or any(v is None for v in vs)
            # 窓の縁の位置の σ（柱割り・幅・上下端の ± を伝えた値）を主に使う
            qual = feature_quality(wmeta, extra_h=None if unk else max(hs), extra_v=None if unk else max(vs))
            if unk:
                qual["grade"] = "不明"
                wmeta["note"] += "。縁の位置に ± の無い値が効く：" + ", ".join(sorted({m for x in o["src"] for m in x.get("sigma_missing", [])}))
            else:
                qual["edge_sigma_h"], qual["edge_sigma_v"] = max(hs), max(vs)
            gen_attrs(w, wmeta, qual)
            grades[tag.lower()].append(qual["grade"])
            evidence[oid] = {"kind": tag.lower(), "quality": qual,
                             "sources": [{kk: x.get(kk) for kk in ("face", "src", "row", "bay", "k", "s0", "s1", "z0", "z1", "kind", "ev",
                                                                   "offset", "sigma", "sigma_missing")} for x in o["src"]]}
            lm = E(B + "lod3MultiSurface", w)
            mm = E(GM + "MultiSurface", lm)
            smm = E(GM + "surfaceMember", mm)
            pid2 = "poly_" + uid(bid, "open", j)
            rids2 = gml_polygon(smm, rs2, geo, pid2)
            if f.meta.get("face"):
                tex_polys.append({"face": f.meta["face"], "pid": pid2, "rids": rids2, "rings": rs2})
            solid_refs.append(pid2)
        new.append(bb)

    # 閉じた立体
    ls = E(B + "lod3Solid")
    so = E(GM + "Solid", ls)
    ex = E(GM + "exterior", so)
    cs = E(GM + "CompositeSurface", ex)
    for pid in solid_refs:
        E(GM + "surfaceMember", cs, **{XL + "href": "#" + pid})
    new.append(ls)

    for j, el in enumerate(new):
        bel.insert(idx + j, el)

    # 建物に物差しの説明と等級の内訳（core:creationDate の後＝gen 属性の場所）
    summary = {k: {g: v.count(g) for g in Q.GRADE_ORDER if v.count(g)} for k, v in grades.items() if v}
    pos = 0
    for i2, ch in enumerate(bel):
        if etree.QName(ch).localname in ("description", "name", "boundedBy", "creationDate", "terminationDate") and etree.QName(ch).namespace in (NS["gml"], NS["core"]):
            pos = i2 + 1
    for name, val in (("LOD3の精度の物差し", RULER_TEXT), ("LOD3の等級の内訳", json.dumps(summary, ensure_ascii=False))):
        a = etree.Element(GEN + "stringAttribute", name=name)
        E(GEN + "value", a, val)
        bel.insert(pos, a)
        pos += 1

    # データ品質属性
    dq = bel.find(f".//{URO}DataQualityAttribute")
    if dq is not None:
        cs_g = "../../codelists/DataQualityAttribute_geometrySrcDesc.xml"
        cs_a = "../../codelists/DataQualityAttribute_appearanceSrcDesc.xml"
        last_g = None
        for ch in dq:
            if etree.QName(ch).localname.startswith("geometrySrcDesc"):
                last_g = ch
        pos = list(dq).index(last_g) + 1 if last_g is not None else 0
        for j, code in enumerate(("000", "801", "901")):     # 公共測量成果（足元）・現地調査（写真の実測）・推定
            el = E(URO + "geometrySrcDescLod3", None, code, codeSpace=cs_g)
            dq.insert(pos + j, el)
        last_a = None
        for ch in dq:
            if etree.QName(ch).localname.startswith(("appearanceSrcDesc", "thematicSrcDesc")):
                last_a = ch
        pos = list(dq).index(last_a) + 1
        # 見た目の出典：1＝空中写真（屋根）、4＝手持ちカメラで撮った写真（壁・窓・扉）、99＝未作成
        codes = (["1"] if textured else []) + (["4"] if wall_textured else []) or ["99"]
        for j, code in enumerate(codes):
            dq.insert(pos + j, E(URO + "appearanceSrcDescLod3", None, code, codeSpace=cs_a))
        dq.insert(pos + len(codes), E(URO + "lodType", None, lod, codeSpace="../../codelists/Building_lodType.xml"))
    return rings_all, roof_rings, tex_polys


def write_wall_textures(bn: str, b: dict, tex_polys: list, appearance, img_dir: Path, app_dir_name: str) -> dict:
    """面ごとに公開版の写真から質感を合成し、app:ParameterizedTexture として壁と開口の多角形に貼る。戻り値は面ごとの記録。"""
    import cv2

    from .. import facade as FA
    from . import wall_texture as WT

    APP = f"{{{NS['app']}}}"
    gks = gaiheki_reports(bn)
    byface = defaultdict(list)
    for tp in tex_polys:
        byface[tp["face"]].append(tp)
    out = {}
    for f in b.get("faces") or []:
        fid = f["id"]
        if fid not in byface or not (f.get("from") and f.get("to")):
            continue
        rect = FA.face_rect(b, f)
        if rect is None:
            continue
        views, skipped = WT.face_views(bn, fid, gks.get(fid), root=CR)
        img, info = WT.compose(rect, views, pitch_m=V((f.get("bays") or {}).get("pitch")))
        info["skipped"] = skipped
        out[fid] = info
        if img is None:
            continue
        timg, k, size = WT.to_pow2(img)
        name = f"{bn}_{fid}.jpg"
        img_dir.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", timg, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(img_dir / name))
        info.update({"image": f"{app_dir_name}/{name}", "image_px": list(size), "scale": round(k, 4)})
        p0, u, _, _ = face_axes(f)
        sd = E(APP + "surfaceDataMember", appearance)
        pt = E(APP + "ParameterizedTexture", sd, **{GM + "id": "tex_" + uid(bn, "wall", fid)})
        E(APP + "imageURI", pt, f"{app_dir_name}/{name}")
        E(APP + "mimeType", pt, "image/jpeg")
        for tp in byface[fid]:
            tcl = E(APP + "TexCoordList", E(APP + "target", pt, uri="#" + tp["pid"]))
            for rid, R in zip(tp["rids"], tp["rings"]):
                uv = WT.tex_coords(np.vstack([R, R[:1]]), p0, u, rect, WT.PPM, k, size)
                E(APP + "textureCoordinates", tcl, " ".join(f"{a:.6f} {c:.6f}" for a, c in uv), ring="#" + rid)
    return out


# ------------------------------------------------------------------ 立体の検査

def check_closed(rings: list[np.ndarray]) -> dict:
    cnt = defaultdict(int)
    for R in rings:
        Rk = [tuple(np.round(p, 4)) for p in R]
        for i in range(len(Rk)):
            a, c = Rk[i], Rk[(i + 1) % len(Rk)]
            cnt[(a, c)] += 1
    bad = [e for e, n in cnt.items() if n != 1 or cnt.get((e[1], e[0]), 0) != 1]
    # 符号つき体積（外向きなら正）：輪ごとに (1/3)・p0・面積ベクトル
    vol = 0.0
    for R in rings:
        A = 0.5 * np.sum(np.cross(R, np.roll(R, -1, axis=0)), axis=0)
        vol += float(R[0] @ A) / 3.0
    return {"directed_edges": len(cnt), "unmatched_edges": len(bad), "signed_volume_m3": round(vol, 1),
            "examples": [[list(map(float, p)) for p in e] for e in bad[:5]]}


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="recorder lod3 build")
    ap.add_argument("--buildings", nargs="+", default=["honbu", "library", "tower75", "kyotsu-minami"])
    ap.add_argument("--lod", nargs="+", default=["3.0"], choices=list(LOD_RULES),
                    help="LOD3 の区分（いくつでも。2 つ以上なら <dataset>_lod30 のように区分ごとに書き出す）")
    ap.add_argument("--no-lod2", action="store_true", help="棟 JSON から作る LOD2（塊と屋根、開口なし）を書かない")
    ap.add_argument("--plateau", default=str(SOURCE_GML))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--dataset", default=str(OUT_DIR / "dataset"), help="PLATEAU と同じ並び（udx/bldg・codelists・schemas）")
    ap.add_argument("--roof-texture", action="store_true", help="屋根に航空写真を貼る（既定は貼らない。航空写真は屋上の縁の照合にだけ使う）")
    ap.add_argument("--wall-texture", action="store_true",
                    help="壁・窓・扉に公開版の写真から作った質感を貼る（面ごとに合成。recorder/lod3/wall_texture.py）")
    ap.add_argument("--windows", default="model", choices=["model", "photos"],
                    help="窓の出どころ。photos＝recorder facade で写真どうしを照合した窓（その面に無ければ棟 JSON の窓）")
    ap.add_argument("--min-photos", type=int, default=2, help="--windows photos で使う窓の、写った写真の枚数の下限")
    ap.add_argument("--sigma-per-tol", type=float, default=0.5, help="σ = ± × これ（0.5：± を 95%% の幅とみなす。1.0：± をそのまま σ）")
    args = ap.parse_args(argv)
    Q.SIGMA_PER_TOL = args.sigma_per_tol
    global THUMB_DIR
    THUMB_DIR = Path(args.out) / "thumbs"
    if THUMB_DIR.exists():
        import shutil
        shutil.rmtree(THUMB_DIR)
    global WINDOW_SOURCE, MIN_PHOTOS
    WINDOW_SOURCE, MIN_PHOTOS = args.windows, args.min_photos
    global RULER_TEXT
    RULER_TEXT = RULER_TEXT.replace("± を標準偏差とみなし（控えめな読み）",
                                    f"± を σ×{1 / args.sigma_per_tol:g} の幅とみなし（σ = ± × {args.sigma_per_tol:g}。"
                                    "棟の記録の ± は物差しの許容を積んだ幅で、標準偏差ではない）")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 同じ棟 JSON から、LOD3 の区分ごとに一斉に書き出す（どの区分にも LOD0・LOD1 は PLATEAU のまま、LOD2 は棟 JSON から）
    lods = list(dict.fromkeys(args.lod))
    all_thumbs: dict = {}
    for lod in lods:
        ds = Path(args.dataset) if len(lods) == 1 else Path(f"{args.dataset}_lod{lod.replace('.', '')}")
        all_thumbs.update(_build_one(args, lod, ds, out))
    # 表示画面が読む：地物の id → 根拠の写真の切り抜き（公開版の写真から。thumbs/ からの相対パス）
    json.dump(all_thumbs, open(out / "thumbs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _build_one(args, lod: str, dataset_dir: Path, out: Path) -> dict:
    """1 つの LOD3 区分の書き出し。戻り値は根拠の写真の切り抜きの一覧（地物の id → 画像）。"""
    parser = etree.XMLParser(huge_tree=True, remove_blank_text=True)
    src = etree.parse(args.plateau, parser)
    root = src.getroot()
    by_id = {el.get(GM + "id"): el for el in root.iter(B + "Building")}

    model = etree.Element(CORE + "CityModel", nsmap={k: v for k, v in NS.items() if k != "xsi"} | {"xsi": NS["xsi"]})
    model.set(f"{{{NS['xsi']}}}schemaLocation", root.get(f"{{{NS['xsi']}}}schemaLocation"))
    evidence, reports = {}, []

    # 1. 形
    built = []
    for bn in args.buildings:
        b = json.load(open(CR / "data" / "buildings" / f"{bn}.json", encoding="utf-8"))
        rj = None
        for cand in (CR / "out" / bn / "roof.json", CR / "out" / bn / "kansatsu" / "roof" / "roof.json"):
            if cand.exists():
                rj = json.load(open(cand, encoding="utf-8"))
                break
        facets, insts, rep = build_building(b, lod, rj)
        built.append((bn, b, rj, facets, insts, rep))

    # 2. 航空写真と倒れ込み（高さ 1 m あたりの屋根のずれ）
    from . import aerial as A
    geo0 = Geo(built[0][1]["crs"], 0.0)
    allxy = np.vstack([np.array(b["footprint"]["polygon"]) for _, b, *_ in built])
    lo, hi = allxy.min(0) - [25, 45], allxy.max(0) + [25, 15]
    ll = geo0.latlonh(np.array([[lo[0], lo[1], 0], [hi[0], hi[1], 0]]))
    mos = A.Mosaic(ll[0, 1], ll[0, 0], ll[1, 1], ll[1, 0])
    leans = {}
    for bn, b, rj, facets, insts, rep in built:
        geo_rel = Geo(b["crs"], 0.0)
        segs = []
        for f in facets:
            if f.kind == "roof":
                P = f.to3d(np.array(f.poly.exterior.coords))
                segs += [(P[i][:2], P[i + 1][:2], float(P[i][2])) for i in range(len(P) - 1)]
        sc = A.scan_lean(mos, geo_rel, segs)
        ok, why = False, ""
        if rj and rj.get("lean_m") and rj.get("height_m") is None:
            pass
        if rj and rj.get("lean_by_block"):
            lb = max(rj["lean_by_block"], key=lambda x: x["top_z"] or 0)
            kj = np.array(lb["lean_m"]) / float(lb["top_z"])
            d = float(np.hypot(*(np.array(sc["k"]) - kj)))
            ok = d < 0.03
            why = f"campus-recorder の roof.json（{lb['id'] or '全体'}、高さ {lb['top_z']} m で {lb['lean_m']} m）との差 {d:.3f} m/m"
        elif sc["second_ratio"] is not None and sc["second_ratio"] < 0.95:
            ok, why = True, f"縁の重なりの 1 位が 2 位より明らかに高い（2 位/1 位 {sc['second_ratio']}）"
        else:
            why = f"縁の重なりの 1 位と 2 位が近い（2 位/1 位 {sc['second_ratio']}）。屋上が足元より小さい（引っ込み）ため縁が合わない"
        leans[bn] = {"scan": sc, "reliable": ok, "why": why}
    good = [leans[bn]["scan"]["k"] for bn in leans if leans[bn]["reliable"]]
    k_mean = np.mean(np.array(good, float), axis=0) if good else np.array([0.0, -0.2])
    for bn in leans:
        if leans[bn]["reliable"]:
            leans[bn]["k"] = [float(x) for x in leans[bn]["scan"]["k"]]
            leans[bn]["how"] = "この棟の屋根の縁を写真に合わせた"
        else:
            leans[bn]["k"] = [round(float(x), 4) for x in k_mean]
            leans[bn]["how"] = f"写真に合わせられた {len(good)} 棟の平均を使った"

    # 3. 書き出し（屋根に航空写真を貼る）
    app_dir_name = "52365768_bldg_6697_appearance"
    ds_bldg = Path(dataset_dir) / "udx" / "bldg"
    (ds_bldg / app_dir_name).mkdir(parents=True, exist_ok=True)
    appearance = E(f"{{{NS['app']}}}Appearance")
    E(f"{{{NS['app']}}}theme", appearance, "rgbTexture")
    APP = f"{{{NS['app']}}}"
    for bn, b, rj, facets, insts, rep in built:
        k = leans[bn]["k"]
        for f in facets:
            if f.kind == "roof":
                f.meta.setdefault("roof_src", {})["lean"] = {"k_m_per_m": k, "how": leans[bn]["how"], "check": leans[bn]["why"]}
        gid = next(e.split(":", 1)[1] for e in b["footprint"]["evidence"] if e.startswith("plateau:"))
        bel = copy.deepcopy(by_id[gid])
        geo = Geo(b["crs"], V(b["ground_z"]))
        grades = defaultdict(list)
        rings, roof_rings, tex_polys = write_lod3(bel, b, facets, insts, geo, evidence, lod, textured=args.roof_texture,
                                                 grades=grades, wall_textured=args.wall_texture)
        if not args.no_lod2:
            f2, rep2 = build_lod2_facets(b, rj)
            rings2 = write_lod2(bel, b, f2, geo, evidence)
            rep["lod2"] = {"surfaces": dict(sorted(((kk, sum(1 for f in f2 if f.kind == kk)) for kk in SURF_TAG), key=lambda x: x[0])),
                           "volume_m3": rep2["volume_m3"], "closed": check_closed(rings2)}
        rep["grades"] = {kk: {g: v.count(g) for g in Q.GRADE_ORDER if v.count(g)} for kk, v in grades.items()}
        rep["roof_lean"] = leans[bn]
        if args.roof_texture:
            # 屋根の各頂点 → 写真の画素（倒れ込みを足す）
            geo_rel = Geo(b["crs"], 0.0)
            allr = np.vstack([R for rr in roof_rings for R in rr["rings"]])
            Qp = np.column_stack([allr[:, 0] + k[0] * allr[:, 2], allr[:, 1] + k[1] * allr[:, 2], allr[:, 2]])
            llq = geo_rel.latlonh(Qp)
            px, py = mos.to_px(llq[:, 1], llq[:, 0])
            im, (x0, y0), scale = A.crop_texture(mos, px, py)
            Wp, Hp = im.width / scale, im.height / scale
            img_name = f"{bn}_roof.jpg"
            im.save(ds_bldg / app_dir_name / img_name, "JPEG", quality=90)
            sd = E(APP + "surfaceDataMember", appearance)
            pt = E(APP + "ParameterizedTexture", sd, **{GM + "id": "tex_" + uid(bn, "roof")})
            E(APP + "imageURI", pt, f"{app_dir_name}/{img_name}")
            E(APP + "mimeType", pt, "image/jpeg")
            for rr in roof_rings:
                tg = E(APP + "target", pt, uri="#" + rr["pid"])
                tcl = E(APP + "TexCoordList", tg)
                for rid, R in zip(rr["rids"], rr["rings"]):
                    RR = np.vstack([R, R[:1]])
                    Qr = np.column_stack([RR[:, 0] + k[0] * RR[:, 2], RR[:, 1] + k[1] * RR[:, 2], RR[:, 2]])
                    lr_ = geo_rel.latlonh(Qr)
                    qx, qy = mos.to_px(lr_[:, 1], lr_[:, 0])
                    u = (qx - (x0 - 0.5)) / Wp
                    v = 1.0 - (qy - (y0 - 0.5)) / Hp
                    E(APP + "textureCoordinates", tcl, " ".join(f"{a:.6f} {c:.6f}" for a, c in zip(u, v)), ring="#" + rid)
            rep["roof_texture"] = {"image": f"{app_dir_name}/{img_name}", "source": A.SOURCE}
        if args.wall_texture:
            rep["wall_texture"] = write_wall_textures(bn, b, tex_polys, appearance, ds_bldg / app_dir_name, app_dir_name)
        rep["closed"] = check_closed(rings)
        rep["plateau_gml_id"] = gid
        rep["surfaces"] = dict(sorted(((kk, sum(1 for f in facets if f.kind == kk)) for kk in SURF_TAG), key=lambda x: x[0]))
        rep["windows"] = sum(len(f.meta.get("openings", [])) for f in facets)
        com = E(CORE + "cityObjectMember", model)
        com.append(bel)
        reports.append(rep)
        print(f"{bn}: 面 {rep['surfaces']} 窓 {rep['windows']}（省いた {len(rep['openings_skipped'])}） 付属物 {len(insts)} "
              f"体積 {rep['volume_m3']} m3 閉じていない辺 {rep['closed']['unmatched_edges']} "
              f"倒れ込み k={leans[bn]['k']}（{leans[bn]['how']}）")
    if args.roof_texture or args.wall_texture:
        am = E(APP + "appearanceMember", model)
        am.append(appearance)

    tag = lod.replace(".", "")
    gml_path = ds_bldg / "52365768_bldg_6697_op.gml"
    etree.ElementTree(model).write(str(gml_path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    # PLATEAU の配布物と同じ並び（udx/bldg・codelists・schemas）にする。codeSpace は ../../codelists/… を指す
    import shutil

    from . import CODELISTS_DIR, XSD_DIR
    ds = Path(dataset_dir)
    shutil.copytree(CODELISTS_DIR, ds / "codelists", dirs_exist_ok=True)
    shutil.copytree(XSD_DIR / "iur", ds / "schemas" / "iur", dirs_exist_ok=True)
    json.dump(evidence, open(out / f"evidence_lod{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 表示画面が読む：地物の id → 根拠の写真の切り抜き（公開版の写真から。thumbs/ からの相対パス）
    thumbs = {k: v["thumbs"] for k, v in evidence.items() if v.get("thumbs")}
    json.dump(reports, open(out / f"report_lod{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("書き出し:", gml_path)
    return thumbs


if __name__ == "__main__":
    main()
