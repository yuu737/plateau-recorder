"""LOD3 の書き出しが読む、写真から出した窓（out/facade/…/final.json）と外壁の読み係の報告（扉・材質・使えない写真）。"""
import json

import pytest

from recorder import facade
from recorder.lod3 import build, quality


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "FACADE_DIR", tmp_path / "out" / "facade")
    return tmp_path


FACE = {"id": "f6", "from": [0.0, 0.0], "to": [50.0, 0.0]}


def test_photo_windows_need_two_photos_and_carry_sigma(root):
    d = root / "out/facade/b1/f6"
    d.mkdir(parents=True)
    wins = [{"id": "w001", "n_photos": 3, "s0": 1.0, "s1": 5.3, "z0": 5.8, "z1": 8.75,
             "sigma_s0": 0.06, "sigma_s1": 0.05, "sigma_z0": 0.07, "sigma_z1": 0.05, "evidence": [{"photo": "P1"}], "regularized": "そのまま"},
            {"id": "w002", "n_photos": 1, "s0": 6.0, "s1": 7.0, "z0": 1.0, "z1": 2.0,
             "sigma_s0": None, "sigma_s1": None, "sigma_z0": None, "sigma_z1": None, "evidence": [{"photo": "P2"}]}]
    (d / "final.json").write_text(json.dumps({"windows": wins}), encoding="utf-8")
    # 外壁の読み係が位置合わせを確かめていない面では使わない
    assert build.photo_window_rects("b1", FACE) is None
    rep = root / "data/reports/b1"
    rep.mkdir(parents=True)
    (rep / "gaiheki_f6.json").write_text(json.dumps({"face": "f6", "photo_alignment": [{"photo": "P1", "usable": True}]}),
                                         encoding="utf-8")
    # 面ごとの決定（data/window_source.json）が photos でなければ使わない
    assert build.photo_window_rects("b1", FACE) is None
    (root / "data/window_source.json").write_text(json.dumps({"faces": {"b1/f6": {"use": "photos"}}}), encoding="utf-8")
    r = build.photo_window_rects("b1", FACE)
    assert [x["photo_window"] for x in r] == ["w001"]
    assert r[0]["sigma_h"] == pytest.approx(0.06) and r[0]["sigma_v"] == pytest.approx(0.07)
    assert r[0]["source_file"] == "final.json" and r[0]["kind"] == "window"
    assert build.photo_window_rects("b1", {**FACE, "id": "f9"}) is None       # その面に無ければ棟 JSON の窓のまま


def test_doors_from_facade_reader(root):
    rep = root / "data/reports/b1"
    rep.mkdir(parents=True)
    doc = {"face": "f6", "made_by": {"model": "claude-opus-5-5"}, "reading_error_m": 0.3,
           "exceptions": [{"what": "扉（正面玄関）", "kind": "door", "s0": 15.0, "s1": 16.2, "z0": 0.0, "z1": 2.5, "how_sure": "mid",
                           "photos": ["P1"]},
                          {"what": "扉？", "kind": "door", "s0": 1.0, "s1": 2.0, "z0": 0.0, "z1": 2.5, "how_sure": "low"},
                          {"what": "ピロティ", "kind": "opening", "s0": 17, "s1": 30, "z0": 0, "z1": 4.8, "how_sure": "high"}],
           "materials": [{"where": "wall", "material": "磁器タイル", "color": "黄土色", "how_sure": "mid", "photos": ["P1"]}]}
    (rep / "gaiheki_f6.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    # 比べるための別の答え（gaiheki_f6_sonnet.json）は正式な報告を上書きしない
    (rep / "gaiheki_f6_sonnet.json").write_text(json.dumps({"face": "f6", "exceptions": [], "materials": []}), encoding="utf-8")
    ds = build.door_rects("b1", FACE)
    assert len(ds) == 1 and ds[0]["kind"] == "door" and ds[0]["claude_read"]
    assert ds[0]["sigma_h"] == pytest.approx(0.3 * quality.SIGMA_PER_TOL)
    assert build.gaiheki_reports("b1")["f6"]["materials"][0]["material"] == "磁器タイル"


def test_unusable_photos_are_excluded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rep = tmp_path / "data/reports/b1"
    rep.mkdir(parents=True)
    (rep / "gaiheki_f6.json").write_text(json.dumps({"face": "f6", "photo_alignment": [
        {"photo": "P1", "shift_m": 0.0, "usable": True}, {"photo": "P2", "shift_m": 10.6, "usable": False, "note": "2 間ずれ"}]},
        ensure_ascii=False), encoding="utf-8")
    bad = facade.unusable_photos("b1", "f6")
    assert set(bad) == {"P2"} and "10.6" in bad["P2"]


def test_new_lineage_keys_have_codes():
    for k in ("photo_multi", "claude_read"):
        assert quality.LINEAGE_CODE[k] == "801" and k in quality.LINEAGE_RANK
