"""検収係のまとめ（data/reports/<棟>/kenshu_summary.json）の面ごとの合否を読む。

元は進捗報告（report.py、PLATEAU 版で削除）の中にあった。係の実行役（agents）が使う。
"""

from __future__ import annotations

import json
from pathlib import Path


def _is_weak(r: dict) -> bool:
    return r.get("weak") is True or str(r.get("note") or "").startswith("弱い合格")


def _face_rows(faces) -> list[dict]:
    """faces の 2 つの形（{面: {...}} と [{face, ...}]）を [{face, ...}] にそろえる。"""
    if isinstance(faces, dict):
        return [dict(v, face=k) for k, v in faces.items() if isinstance(v, dict)]
    return [v for v in (faces or []) if isinstance(v, dict) and v.get("face")]


def _face_verdict(r: dict):
    """面の判定：pass があればそれ。無ければ面の線（lines の ok）から（全部 ok で合格、1 本でも NG で不合格）。どちらも無ければ None。"""
    if "pass" in r:
        return r.get("pass")
    lines = [x for x in (r.get("lines") or []) if isinstance(x, dict) and isinstance(x.get("ok"), bool)]
    if lines:
        return all(x["ok"] for x in lines)
    return None


def kenshu_faces(root: Path, bid: str) -> dict | None:
    """検収係のまとめ（data/reports/<棟>/kenshu_summary.json）の面ごとの合否。無ければ None。
    faces は {面: {pass, note}} と [{face, pass, …}] の 2 つの形がある。弱い合格は weak: true か note の頭が「弱い合格」の物。
    11/8：検収係は再検収の回を round_<…>（made_at つき）に足し、上の faces を書き直さない事がある（本部棟の k401 で
    f6・f8 が faces では合格のまま、最新の回では不合格）。faces は検収係が手で書く欄で、道具は書かない。
    そこで面ごとに「その面が出てくる一番新しい回（made_at の順。上の faces は一番古い回として扱う）」の判定を使う。
    どの回にも出ない面は数えない。最新の回と faces が食い違う面は warnings に書く（止めない）。"""
    p = root / "data" / "reports" / bid / "kenshu_summary.json"
    if not p.exists():
        return None
    try:
        ks = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    base = {r["face"]: r for r in _face_rows(ks.get("faces"))}
    rounds = sorted(((str(v.get("made_at") or ""), k, v) for k, v in ks.items()
                     if k.startswith("round_") and isinstance(v, dict) and v.get("faces") and v.get("made_at")))
    use: dict[str, tuple[str, dict]] = {f: ("faces", r) for f, r in base.items()}
    for _, k, v in rounds:
        for r in _face_rows(v["faces"]):
            use[r["face"]] = (k, r)
    out = {"pass": [], "weak": [], "fail": [], "not_judged": []}
    src = {}
    for f, (k, r) in use.items():
        src[f] = k
        v = _face_verdict(r)
        out["weak" if v is True and _is_weak(r) else "pass" if v is True else "fail" if v is False else "not_judged"].append(f)
    warnings = []
    for f, (k, r) in use.items():
        if k != "faces" and f in base and _face_verdict(base[f]) != _face_verdict(r):
            warnings.append(f"{f}：faces では {_face_verdict(base[f])}、最新の回 {k} では {_face_verdict(r)}（faces が古い。回の判定を使った）")
    return {**out, "made_at": rounds[-1][0] if rounds else ks.get("made_at"), "latest_round": rounds[-1][1] if rounds else None,
            "source": src, "warnings": warnings,
            "file": str(p.relative_to(root)) if p.is_relative_to(root) else str(p)}


