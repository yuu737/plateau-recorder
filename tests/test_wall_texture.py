"""壁の質感の合成（recorder/lod3/wall_texture.py）：足切り、遮蔽の中央値、塗りつぶしの除外、2 の累乗、テクスチャ座標。"""
import json

import cv2
import numpy as np
import pytest

from recorder.lod3 import wall_texture as WT


def _write_view(d, fac, photo, img, mask, s, ztop, rms=2.0, usable=50.0, redact=None):
    d.mkdir(parents=True, exist_ok=True)
    fac.mkdir(parents=True, exist_ok=True)
    meta = {"photo": photo, "image": f"{photo}.jpg", "mask": f"{photo}_mask.png", "px_per_m": 40.0,
            "origin_m": {"s": s, "z_top": ztop}, "usable_m2": usable, "median_photo_px_per_m": 100.0,
            "registration": {"reproj_rms_px": rms}}
    cv2.imwrite(str(d / meta["image"]), img)
    cv2.imwrite(str(fac / meta["mask"]), (mask * 255).astype(np.uint8))
    if redact is not None:
        meta["redact_mask"] = f"{photo}_redact.png"
        cv2.imwrite(str(d / meta["redact_mask"]), (redact * 255).astype(np.uint8))
    (d / f"{photo}.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def face(tmp_path):
    d, fac = tmp_path / WT.PUBLIC / "b1" / "f1", tmp_path / WT.FACADE / "b1" / "f1"
    H, W = 200, 400                          # 面 10 m × 5 m（40 px/m）
    wall = np.full((H, W, 3), (60, 140, 200), np.uint8)
    full = np.ones((H, W), bool)
    for i, ph in enumerate(["A", "B", "C"]):
        img = wall.copy()
        if ph == "A":
            img[50:150, 100:200] = (0, 120, 0)            # 木（1 枚にだけ写る遮蔽）
        if ph == "B":
            img = np.clip(img.astype(int) * 1.3, 0, 255).astype(np.uint8)     # 明るい写真
        _write_view(d, fac, ph, img, full, 0.0, 5.0)
    red = np.zeros((H, W), bool)
    red[0:40, 0:80] = True                                # 塗りつぶした箱
    boxed = wall.copy()
    boxed[red] = (128, 128, 128)
    _write_view(d, fac, "D", boxed, full, 0.0, 5.0, redact=red)
    _write_view(d, fac, "E", wall, full, 0.0, 5.0, rms=20.0)                  # 位置合わせが悪い
    _write_view(d, fac, "F", wall, full, 0.0, 5.0, usable=1.0)                # 使える所が小さい
    return tmp_path


def test_face_views_filters_and_orders(face):
    gk = {"photo_alignment": [{"photo": "C", "usable": True, "shift_m": 10.6}]}     # 読み係が「ずれ」とした写真
    use, skipped = WT.face_views("b1", "f1", gk, root=face)
    assert sorted(v["photo"] for v in use) == ["A", "B", "D"]
    why = {s["photo"]: s["why"] for s in skipped}
    assert "ずれ" in why["C"] and "位置合わせの誤差" in why["E"] and "小さい" in why["F"]
    d = next(v for v in use if v["photo"] == "D")
    assert d["redacted_m2"] == pytest.approx(40 * 80 / 1600, abs=0.1) and not d["mask"][0:40, 0:80].any()


def test_compose_removes_occluder_and_matches_colour(face):
    use, _ = WT.face_views("b1", "f1", None, root=face)
    img, info = WT.compose((0.0, 10.0, 0.0, 5.0), use)
    assert img.shape == (200, 400, 3) and info["coverage"] == 1.0 and info["coverage_median"] == 1.0
    # 1 枚にだけ写った木は中央値で消え、明るい写真は倍率で合わせられて、壁の色にそろう
    assert np.abs(img[100, 150].astype(int) - (60, 140, 200)).max() <= 8
    assert np.abs(img[150, 350].astype(int) - (60, 140, 200)).max() <= 8
    # 塗りつぶしの箱は使わない（灰色が残らない）
    assert np.abs(img[10, 10].astype(int) - (60, 140, 200)).max() <= 8
    gB = next(u["gain_bgr"] for u in info["used"] if u["photo"] == "B")
    assert all(g < 0.9 for g in gB)


def test_unseen_area_is_filled_from_the_next_bay():
    H, W, p = 40, 200, 50                     # 柱間 50 px のくり返し
    out = np.zeros((H, W, 3), np.uint8)
    for x in range(W):
        out[:, x] = (200, 100, 50) if (x % p) < 10 else (30, 30, 30)
    seen = np.ones((H, W), bool)
    seen[:, 60:100] = False                    # 垂れ幕で隠れた所
    truth = out.copy()
    out[:, 60:100] = 0
    got, info = WT.fill_unseen(out, seen, p)
    assert np.array_equal(got[:, 60:100], truth[:, 60:100])       # となりの柱間から写すと元に戻る
    assert info["filled_by_repeat"] == pytest.approx(40 * 40 / (H * W)) and info["filled_by_inpaint"] == 0.0
    got2, info2 = WT.fill_unseen(out.copy(), seen, None)           # 柱間が分からなければ周りから
    assert info2["filled_by_inpaint"] > 0 and info2["filled_by_repeat"] == 0.0


def test_unseen_area_is_filled_and_pow2():
    img = np.zeros((10, 10, 3), np.uint8)
    out, k, size = WT.to_pow2(np.zeros((300, 3000, 3), np.uint8))
    assert size == (2048, 256) and out.shape == (256, 2048, 3) and k == pytest.approx(2048 / 3000)
    assert tuple(out[-1, -1]) == WT.PAD_RGB[::-1]
    out, k, size = WT.to_pow2(img)
    assert size == (16, 16) and k == 1.0


def test_tex_coords_follow_the_face():
    p0, u = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    P = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 0.0, 0.0]])
    uv = WT.tex_coords(P, p0, u, (0.0, 10.0, 0.0, 5.0), 40.0, 1.0, (512, 256))
    assert uv[0] == pytest.approx((0.0, 1.0)) and uv[1] == pytest.approx((400 / 512, 1.0))
    assert uv[2] == pytest.approx((400 / 512, 1.0 - 200 / 256))
