"""写真 1 枚ずつの窓を写真をまたいで束ね、値と σ を出す（recorder.facade）。検出器は使わない（箱を手で作る）。"""
import math

import pytest

from recorder import facade


def box(photo, s0, s1, z0, z1, score=0.5):
    return {"photo": photo, "score": score, "px": [0, 0, 1, 1], "s0": s0, "s1": s1, "z0": z0, "z1": z1}


def test_iou_m():
    a = {"s0": 0, "s1": 2, "z0": 0, "z1": 2}
    assert facade.iou_m(a, a) == pytest.approx(1.0)
    assert facade.iou_m(a, {"s0": 1, "s1": 3, "z0": 0, "z1": 2}) == pytest.approx(1 / 3)
    assert facade.iou_m(a, {"s0": 5, "s1": 6, "z0": 0, "z1": 2}) == 0.0


def test_same_window_from_three_photos_gives_median_and_sigma():
    boxes = [box("A", 1.00, 2.00, 5.00, 6.50), box("B", 1.10, 2.10, 5.10, 6.40), box("C", 0.90, 1.95, 4.95, 6.55),
             box("A", 4.00, 5.00, 5.00, 6.50)]                     # 別の窓（A にだけ写る）
    wins = facade.fuse_boxes(boxes)
    assert len(wins) == 2
    w = max(wins, key=lambda w: w["n_photos"])
    assert w["n_photos"] == 3 and w["s0"] == pytest.approx(1.00)
    sd = float(__import__("numpy").std([1.00, 1.10, 0.90], ddof=1))
    assert w["sd_s0"] == pytest.approx(sd, abs=1e-3)
    assert w["sigma_s0"] == pytest.approx(math.sqrt(sd * sd / 3 + facade.FLOOR_M ** 2), abs=1e-3)
    assert {e["photo"] for e in w["evidence"]} == {"A", "B", "C"}
    one = min(wins, key=lambda w: w["n_photos"])
    assert one["n_photos"] == 1 and one["sigma_s0"] is None       # 1 枚だけ：照合なし、σ は出さない


def test_one_photo_contributes_one_box_per_window():
    # 同じ写真の重なった 2 つの箱は、同じ窓の束に入れない（写真どうしの照合にならないため）
    wins = facade.fuse_boxes([box("A", 1.0, 2.0, 5.0, 6.5, 0.9), box("A", 1.05, 2.05, 5.0, 6.5, 0.4)])
    assert sorted(w["n_photos"] for w in wins) == [1, 1]


def test_px_m_roundtrip():
    meta = {"px_per_m": 40.0, "origin_m": {"s": 12.5, "z_top": 20.0}}
    s, z = facade.px_to_m(meta, 80, 120)
    assert (s, z) == pytest.approx((14.5, 17.0))
    assert facade.m_to_px(meta, s, z) == pytest.approx((80, 120))


def test_tiles_cover_tall_image():
    ts = facade._tiles(900, 2900)
    assert ts[0][:2] == (0, 0) and max(t[3] for t in ts) == 2900 and all(t[2] == 900 for t in ts)
