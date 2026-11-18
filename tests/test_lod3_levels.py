"""同じ棟 JSON から LOD を一斉に（11/19）：面に描いた扉を Door に、LOD2 を足す、区分ごとに書き出す、閉じない彫り込みを外す。

本物の棟 JSON・PLATEAU のファイルを読むテストは、出力を一時フォルダに書く（out/ は書き換えない）。
"""
import json
from pathlib import Path

import pytest
from lxml import etree

from recorder.lod3 import build, quality

REPO = Path(__file__).resolve().parent.parent
HAS_DATA = (REPO / "data" / "buildings" / "library.json").exists() and Path(build.SOURCE_GML).exists()


def test_patch_doors_become_door_rects():
    b = {"parts": [
        {"type": "patch", "name": "茶色の両開きの扉（列 8）", "face": "f1",
         "along0": {"value": 39.6, "unit": "m", "tol": 1.24, "status": "estimated"},
         "along1": {"value": 41.2, "unit": "m", "tol": 1.29, "status": "estimated"},
         "z0": {"value": 2.0, "unit": "m", "tol": 0.6, "status": "estimated"},
         "z1": {"value": 4.3, "unit": "m", "tol": 0.6, "status": "estimated"}},
        {"type": "patch", "name": "ガラリ", "face": "f1", "along0": 1, "along1": 2, "z0": 0, "z1": 1},      # 扉でない
        {"type": "composite", "name": "扉の上の庇", "face": "f1"},                                          # 塗り分けでない
        {"type": "patch", "name": "扉（別の面）", "face": "f2",
         "along0": {"value": 1, "unit": "m", "tol": 0.1}, "along1": {"value": 2, "unit": "m", "tol": 0.1},
         "z0": {"value": 0, "unit": "m", "tol": 0.1}, "z1": {"value": 2, "unit": "m", "tol": 0.1}}]}
    rs = build.patch_door_rects(b, {"id": "f1"})
    assert len(rs) == 1
    r = rs[0]
    assert r["kind"] == "door" and r["src"] == "parts[0]" and (r["s0"], r["s1"], r["z0"], r["z1"]) == (39.6, 41.2, 2.0, 4.3)
    assert r["sigma_h"] == pytest.approx(1.29 * quality.SIGMA_PER_TOL) and r["sigma_v"] == pytest.approx(0.6 * quality.SIGMA_PER_TOL)
    assert set(r["used"]) == {"扉.along0", "扉.along1", "扉.z0", "扉.z1"}
    assert build.patch_door_rects(None, {"id": "f1"}) == []


@pytest.mark.skipif(not HAS_DATA, reason="棟 JSON か PLATEAU の元ファイルが無い")
def test_build_several_lods_with_lod2(tmp_path):
    out = tmp_path / "lod3"
    build.main(["--buildings", "library", "--lod", "3.0", "3.2", "--out", str(out), "--dataset", str(out / "dataset")])
    for tag in ("30", "32"):
        gml = out / f"dataset_lod{tag}" / "udx" / "bldg" / "52365768_bldg_6697_op.gml"
        assert gml.exists()
        doc = etree.parse(str(gml))
        ns = {"bldg": build.NS["bldg"], "uro": build.NS["uro"]}
        bl = doc.find(".//bldg:Building", ns)
        names = [etree.QName(c).localname for c in bl]
        # CityGML 2.0 の並び：lod1Solid → lod2Solid → 付属物 → 境界面 → lod3Solid
        assert names.index("lod1Solid") < names.index("lod2Solid") < names.index("boundedBy") < names.index("lod3Solid")
        assert bl.findall(".//bldg:lod2MultiSurface", ns) and bl.findall(".//bldg:lod3MultiSurface", ns)
        assert [e.text for e in bl.findall(".//uro:lodType", ns)] == ["2.0", f"3.{tag[1]}"]
        rep = json.loads((out / f"report_lod{tag}.json").read_text(encoding="utf-8"))[0]
        assert rep["closed"]["unmatched_edges"] == 0 and rep["lod2"]["closed"]["unmatched_edges"] == 0
        ev = json.loads((out / f"evidence_lod{tag}.json").read_text(encoding="utf-8"))
        assert any(v.get("kind") == "door" for v in ev.values())                 # 面に描いた扉が Door になる
    assert (out / "thumbs.json").exists()
