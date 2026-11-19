"""質感のアトラス（recorder/lod3/atlas.py）：重ならずに詰める、1 辺 2 の累乗・4096 px まで、余白、テクスチャ座標。"""
import numpy as np
import pytest

from recorder.lod3 import atlas as AT


def _img(w, h, c):
    a = np.zeros((h, w, 3), np.uint8)
    a[:] = c
    return a


def test_pack_no_overlap_and_pow2_pages():
    at = AT.Atlas()
    sizes = [(3978, 1814), (3954, 2024), (1211, 4092), (341, 4092), (500, 300), (456, 552), (2000, 1000)]
    for i, (w, h) in enumerate(sizes):
        at.add(_img(w, h, (i * 30 % 255, 50, 100)))
    at.pack()
    for W, H in at.sizes:
        assert W <= AT.PAGE and H <= AT.PAGE and W & (W - 1) == 0 and H & (H - 1) == 0
    boxes = []
    for i, (w, h) in enumerate(sizes):
        r = at.rect(i)
        W, H = at.sizes[r["page"]]
        assert r["x"] >= AT.MARGIN and r["y"] >= AT.MARGIN                     # 左上から詰め、外側に余白
        assert r["x"] + w + AT.MARGIN <= W and r["y"] + h + AT.MARGIN <= H
        boxes.append((r["page"], r["x"] - AT.MARGIN, r["y"] - AT.MARGIN, w + 2 * AT.MARGIN, h + 2 * AT.MARGIN))
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            pa, xa, ya, wa, ha = boxes[a]
            pb, xb, yb, wb, hb = boxes[b]
            if pa == pb:
                assert xa + wa <= xb or xb + wb <= xa or ya + ha <= yb or yb + hb <= ya


def test_page_image_and_uv_point_to_the_same_pixels():
    at = AT.Atlas()
    a = at.add(_img(100, 50, (10, 20, 30)))
    b = at.add(_img(40, 80, (200, 100, 0)))
    at.pack()
    img = at.page_image(0)
    W, H = at.sizes[0]
    assert img.shape == (H, W, 3)
    for i, c in ((a, (10, 20, 30)), (b, (200, 100, 0))):
        h, w = at.items[i].shape[:2]
        (u0, v0), (u1, v1) = at.uv(i, [0.5, w - 0.5], [0.5, h - 0.5])              # 画像の左上と右下の画素の中心
        for u, v in ((u0, v0), (u1, v1)):
            x, y = int(u * W), int((1.0 - v) * H)
            assert tuple(img[y, x]) == c
        r = at.rect(i)
        assert tuple(img[r["y"] - 1, r["x"] - 1]) == c                         # 余白は縁の画素のくり返し
    assert tuple(img[H - 1, W - 1]) in ((90, 90, 90), (10, 20, 30), (200, 100, 0))


def test_too_large_image_is_refused():
    at = AT.Atlas()
    with pytest.raises(ValueError):
        at.add(_img(4095, 10, (0, 0, 0)))
