"""国土地理院の航空写真（全国最新写真（シームレス）、z18）を取り、屋根の倒れ込みを測り、屋根の画像を切り出す。

航空写真は正射写真だが、建物は高さに比例して写真の中で倒れて写る（屋根が足元からずれる）。
高さ 1 m あたりのずれ k = (kx, ky) m/m を、屋根の縁（明るさの勾配）といちばん重なる値として探す。
"""
from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import PLATEAU_DIR

TILE_DIR = PLATEAU_DIR / "gsi_seamlessphoto_z18"     # 取ったタイルの写し（git 管理外）
URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
Z = 18
SOURCE = "国土地理院 全国最新写真（シームレス） https://maps.gsi.go.jp/development/ichiran.html"


def lonlat_to_tile(lon, lat, z=Z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lr = np.radians(lat)
    y = (1.0 - np.log(np.tan(lr) + 1.0 / np.cos(lr)) / math.pi) / 2.0 * n
    return x, y


class Mosaic:
    def __init__(self, lon0, lat0, lon1, lat1, z=Z):
        tx0, ty1 = lonlat_to_tile(lon0, lat0, z)
        tx1, ty0 = lonlat_to_tile(lon1, lat1, z)
        self.z = z
        self.ix0, self.iy0 = int(math.floor(min(tx0, tx1))), int(math.floor(min(ty0, ty1)))
        self.ix1, self.iy1 = int(math.floor(max(tx0, tx1))), int(math.floor(max(ty0, ty1)))
        W, H = (self.ix1 - self.ix0 + 1) * 256, (self.iy1 - self.iy0 + 1) * 256
        self.img = np.zeros((H, W, 3), np.uint8)
        TILE_DIR.mkdir(parents=True, exist_ok=True)
        self.n_fetched = 0
        for tx in range(self.ix0, self.ix1 + 1):
            for ty in range(self.iy0, self.iy1 + 1):
                p = TILE_DIR / f"{z}_{tx}_{ty}.jpg"
                if not p.exists():
                    req = urllib.request.Request(URL.format(z=z, x=tx, y=ty), headers={"User-Agent": "lod3-roof-texture"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        p.write_bytes(r.read())
                    self.n_fetched += 1
                im = np.asarray(Image.open(p).convert("RGB"))
                ox, oy = (tx - self.ix0) * 256, (ty - self.iy0) * 256
                self.img[oy:oy + 256, ox:ox + 256] = im
        g = self.img.astype(float).mean(axis=2)
        gy, gx = np.gradient(g)
        self.grad = np.hypot(gx, gy)

    def to_px(self, lon, lat):
        x, y = lonlat_to_tile(np.asarray(lon, float), np.asarray(lat, float), self.z)
        return (x - self.ix0) * 256.0 - 0.5, (y - self.iy0) * 256.0 - 0.5     # 画素の中心が整数

    def sample(self, a, px, py):
        """双一次補間。"""
        px = np.clip(px, 0, a.shape[1] - 1.001)
        py = np.clip(py, 0, a.shape[0] - 1.001)
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - x0, py - y0
        return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
                + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)


def edge_samples(segs, step=0.25):
    """屋根の縁の線分 [(p, q, z)] → 点 (x, y, z) の並び。"""
    pts = []
    for p, q, z in segs:
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(L / step))
        for t in np.linspace(0, 1, n, endpoint=False) + 0.5 / n:
            pts.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), z))
    return np.array(pts)


def scan_lean(mos: Mosaic, geo, segs, kx_rng=(-0.06, 0.06), ky_rng=(-0.32, 0.02), step=0.004):
    """屋根の縁の点を (x + kx·z, y + ky·z) にずらし、写真の勾配の合計が最大になる k を探す。"""
    P = edge_samples(segs)
    best = None
    scores = {}
    kxs = np.arange(kx_rng[0], kx_rng[1] + 1e-9, step)
    kys = np.arange(ky_rng[0], ky_rng[1] + 1e-9, step)
    for kx in kxs:
        for ky in kys:
            Q = np.column_stack([P[:, 0] + kx * P[:, 2], P[:, 1] + ky * P[:, 2], P[:, 2]])
            ll = geo.latlonh(Q)
            px, py = mos.to_px(ll[:, 1], ll[:, 0])
            s = float(np.mean(mos.sample(mos.grad, px, py)))
            scores[(round(kx, 4), round(ky, 4))] = s
            if best is None or s > best[0]:
                best = (s, kx, ky)
    vals = np.array(list(scores.values()))
    # 2 番目の山（1 位から 1.5 m 以上離れた所）
    s1, kx1, ky1 = best
    zmax = float(P[:, 2].max())
    second = max((v for (kx, ky), v in scores.items() if math.hypot(kx - kx1, ky - ky1) * zmax > 1.5), default=0.0)
    return {"k": [round(kx1, 4), round(ky1, 4)], "score": round(s1, 2), "score_median": round(float(np.median(vals)), 2),
            "second_ratio": round(second / s1, 3) if s1 else None, "n_points": len(P),
            "lean_at_top_m": [round(kx1 * zmax, 2), round(ky1 * zmax, 2)], "z_top": zmax}


def crop_texture(mos: Mosaic, px, py, margin=4, scale=4):
    """屋根の画素の範囲を切り出して scale 倍（双三次）。返り値 (画像, 左上の画素, 倍率)。"""
    x0, y0 = int(math.floor(np.min(px))) - margin, int(math.floor(np.min(py))) - margin
    x1, y1 = int(math.ceil(np.max(px))) + margin, int(math.ceil(np.max(py))) + margin
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mos.img.shape[1], x1), min(mos.img.shape[0], y1)
    im = Image.fromarray(mos.img[y0:y1, x0:x1])
    im = im.resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.BICUBIC)
    return im, (x0, y0), scale


def jpeg_bytes(im: Image.Image, q=90) -> bytes:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return b.getvalue()
