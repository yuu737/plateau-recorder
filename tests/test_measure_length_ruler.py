"""物差しはタイルに限らない（11/19）：巻尺か距離計で直接測った長さ（柱型の幅など）→ 同じ正面化画像の px の比 → 面の値。

棟 JSON・物差し台帳の本物は書かない（一時フォルダの写しだけ）。
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from recorder import __main__ as cli
from recorder import measure, schema

REPO = Path(__file__).resolve().parent.parent
TAPE = "本部棟正面柱92cm"
F6 = "PXL_20261103_042900775"


def piers(width=4400, height=300, left=634.0, spacing=637.4, pier=109.3, n=6, noise=4.0, seed=0):
    """明るい柱型（幅 pier px）が spacing px ごとに並び、その間が暗い窓の凹み。縁は小数 px（線形に混ぜる）。"""
    rng = np.random.default_rng(seed)
    x = np.arange(width) + 0.5
    prof = np.full(width, 60.0)
    for k in range(n):
        l = left + k * spacing
        r = l + pier
        cover = np.clip(np.minimum(x + 0.5, r) - np.maximum(x - 0.5, l), 0, 1)
        prof = prof + cover * (190.0 - 60.0)
    img = np.tile(prof, (height, 1)) + rng.normal(0, noise, (height, width))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_length_tape_needs_a_name_and_keeps_tile_files_apart(tmp_path):
    with pytest.raises(ValueError):
        measure.tape_link(TAPE, "0→92.0", 1, "length_h")                       # 何を測ったかが無い
    with pytest.raises(ValueError):
        measure.tape_link(TAPE, "0→92.0", 1, "length_h", ref="柱型")            # 名前は英数字
    with pytest.raises(ValueError):
        measure.tape_link(TAPE, "5.4→86.2", 7, "tile_pitch_h", ref="pier_f6")  # タイルに名前は付けない
    ln = measure.tape_link(TAPE, "0→92.0", 1, "length_h", ref="pier_f6", ref_name="f6 の柱型の幅（1 階）", tool="巻尺")
    assert ln["value"] == pytest.approx(0.92) and ln["tol"] == pytest.approx(0.005) and ln["ref"] == "pier_f6"
    # 同じ写真でタイルも読んだ：ファイルは別、突き合わせも別
    rep = tmp_path / "rep"
    measure.write_tape(measure.tape_link(TAPE, "5.4→86.2", 7, "tile_pitch_h"), building="honbu", face="f6", reports=rep)
    p = measure.write_tape(ln, building="honbu", face="f6", reports=rep)
    assert p.name == f"sokuryo_tape_{TAPE}__length_h_pier_f6.json"
    assert (rep / "honbu" / f"sokuryo_tape_{TAPE}.json").exists()
    tile = measure.pitch_summary("honbu", "tile_pitch_h", rep)
    pier = measure.pitch_summary("honbu", "length_h", rep, ref="pier_f6")
    assert tile["value"] == pytest.approx(0.115429, abs=1e-6) and pier["value"] == pytest.approx(0.92)
    assert measure.pitch_summary("honbu", "length_h", rep, ref="door_f6")["value"] is None
    with pytest.raises(ValueError):
        measure.pitch_summary("honbu", "length_h", rep)                          # 長さは名前で選ぶ


def test_element_spans_recovers_width_and_spacing():
    img = piers()
    r = measure.element_spans(img, box=[0, 0, 4400, 300], width_px=(80, 140))
    assert r["ok"] and len(r["elements"]) == 6
    assert np.mean(r["widths"]) == pytest.approx(109.3, abs=0.6)
    assert np.mean(r["spacings"]) == pytest.approx(637.4, abs=0.6)
    # 明暗の向きを間違えると、窓の凹み（暗い所）を拾う。幅の範囲で外れる
    assert not measure.element_spans(img, box=[0, 0, 4400, 300], width_px=(80, 140), bright=False)["ok"]


def test_element_spans_drops_a_missing_element_from_spacings():
    img = piers()
    img[:, 634 + 2 * 637:634 + 2 * 637 + 115] = 60            # 3 本目の柱型を木で隠した
    r = measure.element_spans(img, box=[0, 0, 4400, 300], width_px=(80, 140))
    assert r["ok"] and len(r["elements"]) == 5
    assert r["dropped_spacings"] and all(abs(s - 637.4) < 1.0 for s in r["spacings"])


def test_spacing_range_skips_an_extra_edge():
    img = piers(n=2, left=634.0)
    img[:, 634 + 176:634 + 176 + 100] = 190                    # 窓枠などの余計な明るい帯（柱型と同じくらいの幅）
    r = measure.element_spans(img, box=[0, 0, 4400, 300], width_px=(80, 140))
    assert not r["ok"]                                          # 隣どうしだと間がそろわない
    r = measure.element_spans(img, box=[0, 0, 4400, 300], width_px=(80, 140), spacing_px=(600, 680))
    assert r["ok"] and r["spacings"] == [pytest.approx(637.4, abs=0.6)]
    assert len(r["widths"]) == 2 and all(abs(w - 109.3) < 0.8 for w in r["widths"])   # 余計な帯の幅は入れない


def test_span_to_chain_is_measured_and_matches_schema(tmp_path):
    rows = [measure.element_spans(piers(seed=s), box=[0, 0, 4400, 300], width_px=(80, 140)) for s in (1, 2)]
    sp = measure.span_summary(rows, photo=F6, face="f6")
    assert sp["ok"] and sp["px_tol"] >= 1.0
    rep = tmp_path / "rep"
    measure.write_tape(measure.tape_link(TAPE, "0→92.0", 1, "length_h", ref="pier_f6"), building="honbu", face="f6",
                       reports=rep)
    s = measure.pitch_summary("honbu", "length_h", rep, ref="pier_f6")
    m = measure.build_chain(building="honbu", face="f6", pitch=s, ratios=[measure.span_ratio(sp, "spacing/width")])
    assert m["status"] == "measured" and m["measured_kind"] == "chain"
    assert [ln["link"] for ln in m["chain"]] == ["tape", "px_ratio"]
    assert m["value"] == pytest.approx(0.92 * 637.4 / 109.3, rel=0.005)
    assert m["chain"][1]["px_tol"] == sp["px_tol"]
    assert m["tol"] >= schema.chain_tol(m["chain"], m["value"]) - 1e-12
    assert "f6 の柱型" not in m["method"] and "pier_f6" in m["method"]          # 説明が無ければ名前で書く
    # 別の面に当てると estimated（仮定の輪）
    m4 = measure.build_chain(building="honbu", face="f4", pitch=s, ratios=[measure.span_ratio(sp, "spacing/width")])
    assert m4["status"] == "estimated" and any(ln["link"] == "assume" for ln in m4["chain"])


def test_cli_length_tape_span_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDER_LOCK_DIR", str(tmp_path / "locks"))
    rep = tmp_path / "rep"
    assert cli.main(["measure", "tape", TAPE, "--read", "0→92.0", "--n", "1", "--what", "length_h", "--ref", "pier_f6",
                     "--ref-name", "f6 の柱型の幅（1 階）", "--tool", "巻尺", "--building", "honbu", "--face", "f6",
                     "--reports", str(rep)]) == 0
    assert cli.main(["measure", "tape", TAPE, "--read", "0→92.0", "--n", "1", "--what", "length_h",
                     "--building", "honbu", "--reports", str(rep)]) == 2                       # 名前が無い
    jpg = tmp_path / "f6_hi.jpg"
    cv2.imwrite(str(jpg), cv2.cvtColor(piers(), cv2.COLOR_GRAY2BGR))
    assert cli.main(["measure", "span", str(jpg), "--photo", F6, "--face", "f6", "--box", "0,0,4400,150",
                     "--box", "0,150,4400,300", "--width-px", "80", "140"]) == 0
    sj = tmp_path / "f6_hi_span.json"
    sp = json.loads(sj.read_text(encoding="utf-8"))
    assert sp["ok"] and sp["n_widths"] == 12 and sp["width_px"] == pytest.approx(109.3, abs=0.6)
    assert cli.main(["measure", "chain", "honbu", "--face", "f6", "--key", "bay_pitch_by_pier", "--what", "length_h",
                     "--ref", "pier_f6", "--span", str(sj), "--reports", str(rep)]) == 0
    d = json.loads((rep / "honbu" / "sokuryo_chain_f6.json").read_text(encoding="utf-8"))
    m = d["values"]["bay_pitch_by_pier"]
    assert m["status"] == "measured" and m["value"] == pytest.approx(5.365, abs=0.03)
    assert "f6 の柱型の幅（1 階）" in m["method"]
    # 別の面の span は使わない
    assert cli.main(["measure", "chain", "honbu", "--face", "f4", "--key", "x", "--what", "length_h",
                     "--ref", "pier_f6", "--span", str(sj), "--reports", str(rep)]) == 2
    # 台帳（写し）には名前付きの id で書く
    rul = tmp_path / "rulers.json"
    shutil.copy(REPO / "data" / "rulers.json", rul)
    assert cli.main(["measure", "pitch", "honbu", "--what", "length_h", "--ref", "pier_f6", "--write-rulers",
                     "--rulers", str(rul), "--reports", str(rep)]) == 0
    r = next(x for x in json.loads(rul.read_text(encoding="utf-8"))["rulers"] if x["id"] == "length_h_pier_f6_honbu")
    assert r["value"] == pytest.approx(0.92) and "そのまま長さにした" in r["method"]
