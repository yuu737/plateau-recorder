"""棟 JSON の値の件数と「独立した観察」の件数（同じ状態・方法・根拠・値の種類を 1 つに束ねる）。

元は証拠ビューア（evidence.py、Rerun。PLATEAU 版で削除）の中にあった。`recorder counts` が使う。
"""

from __future__ import annotations

import re
from collections import Counter

STATUSES = ("measured", "estimated", "unknown")
STATUS_JA = {"measured": "実測", "estimated": "推定", "unknown": "不明"}


def _count(o, c: Counter) -> None:
    if isinstance(o, dict):
        if o.get("status") in STATUSES:
            c[o["status"]] += 1
        for v in o.values():
            _count(v, c)
    elif isinstance(o, list):
        for v in o:
            _count(v, c)


def status_counts(b: dict) -> dict:
    """棟 JSON の値の状態（measured／estimated／unknown）の件数。{"total", "faces": {面: Counter}, "other"}。
    other は面の外（棟全体の値・塊・部品）。撮影依頼（requests）の status（open 等）は数えない。"""
    total, other, faces = Counter(), Counter(), {}
    for k, v in b.items():
        if k in ("requests", "history"):
            continue
        if k == "faces":
            for f in v:
                c = Counter()
                _count(f, c)
                faces[f["id"]] = c
                total.update(c)
        else:
            c = Counter()
            _count(v, c)
            other.update(c)
            total.update(c)
    return {"total": total, "faces": faces, "other": other}


# --- 独立した観察の件数（11/6 追加）----------------------------------------------------------
# 値の件数は「模型をどれだけ細かく割ったか」で増える（帯を階ごと・柱型ごとに置けば、1 回見た帯の見付けが何百回も書かれる）。
# そこで、同じ (状態, 方法の文, 根拠の写真・資料, 値の種類) の値は「1 つの観察の複製」とみなして束ねる。
# 値の種類は JSON のパスから [番号] を取った物（faces[].ledges[].depth）。面をまたいだ同じ鍵も 1 つに束ねる。

EYEBALL = "目視"


def _hashable(x):
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    return x


def _walk_values(o, path: str, out: list, face: str | None, values_only: bool) -> None:
    """status 付きの値を (パス, 面, 値の dict) で集める。values_only は run.count_statuses と同じ数え方
    （status と value の両方を持つ dict だけ、その中には降りない）。既定は status_counts と同じ（status を持つ dict 全部）。"""
    if isinstance(o, dict):
        st = o.get("status")
        if values_only:
            if "status" in o and "value" in o:
                if st in STATUSES:
                    out.append((path, face, o))
                return
        elif st in STATUSES:
            out.append((path, face, o))
        for k, v in o.items():
            _walk_values(v, f"{path}.{k}" if path else k, out, face, values_only)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _walk_values(v, f"{path}[{i}]", out, face, values_only)


def value_kind(path: str) -> str:
    """値の種類：パスから [番号] を取った物。faces[3].ledges[12].depth → faces[].ledges[].depth"""
    return re.sub(r"\[\d+\]", "[]", path)


def observation_key(path: str, v: dict) -> tuple:
    """観察の鍵 (状態, 方法の文, 根拠のタプル, 値の種類)。"""
    ev = v.get("evidence") or []
    if not isinstance(ev, (list, tuple)):
        ev = [ev]
    return (v["status"], (v.get("method") or "").strip(), _hashable(list(ev)), value_kind(path))


def _tally(items: list) -> dict:
    keys = Counter(observation_key(p, v) for p, _, v in items)
    by = {s: {"values": 0, "observations": 0} for s in STATUSES}
    for (st, *_), n in keys.items():
        by[st]["values"] += n
        by[st]["observations"] += 1
    eye = [k for k in keys if k[1] == EYEBALL]
    return {"values": len(items), "observations": len(keys), "by_status": by,
            "derived": len(items) - len(keys),
            "eyeball": {"values": sum(keys[k] for k in eye), "observations": len(eye)}, "_keys": keys}


def observations(b: dict, values_only: bool = False, top: int = 10) -> dict:
    """値の個数と、独立した観察の個数を分けて数える。

    返す物：values（値の総数＝今までの数え方）、observations（観察の鍵の数）、by_status（状態ごとの値と観察）、
    derived（複製で増えた数＝values − observations）、eyeball（方法が「目視」だけの値と観察）、
    top_copies（複製の多い鍵の上位。{status, method, evidence, kind, copies}）、
    faces（面ごとの values・observations・by_status。面の中で束ねる）、other（面の外）。
    values_only=False は status_counts と同じ範囲（requests・history を除く status 付きの dict 全部）、
    True は run.count_statuses と同じ範囲（requests を除き、status と value を両方持つ値だけ）。"""
    skip = ("requests",) if values_only else ("requests", "history")
    items: list = []
    face_items: dict[str, list] = {}
    other: list = []
    for k, v in b.items():
        if k in skip:
            continue
        if k == "faces" and isinstance(v, list):
            for i, f in enumerate(v):
                got: list = []
                _walk_values(f, f"faces[{i}]", got, f.get("id") if isinstance(f, dict) else None, values_only)
                face_items[f.get("id", f"faces[{i}]") if isinstance(f, dict) else f"faces[{i}]"] = got
                items += got
        else:
            got = []
            _walk_values(v, k, got, None, values_only)
            other += got
            items += got
    res = _tally(items)
    keys = res.pop("_keys")
    res["top_copies"] = [{"status": k[0], "method": k[1], "evidence": list(k[2]), "kind": k[3], "copies": n}
                         for k, n in sorted(keys.items(), key=lambda kv: (-kv[1], kv[0][3], kv[0][1]))[:top]]
    res["faces"] = {}
    for fid, got in face_items.items():
        t = _tally(got)
        t.pop("_keys")
        res["faces"][fid] = t
    t = _tally(other)
    t.pop("_keys")
    res["other"] = t
    return res


def observations_line(o: dict) -> str:
    """1 行の表示：値 measured N / estimated N / unknown N　｜　独立した観察 measured N / … （複製 N）"""
    by = o["by_status"]
    v = " / ".join(f"{s} {by[s]['values']}" for s in STATUSES)
    ob = " / ".join(f"{s} {by[s]['observations']}" for s in STATUSES)
    return f"値 {v}　｜　独立した観察 {ob}（複製 {o['derived']}）"


OBS_NOTE = "件数は部品ごとに数えるので、同じ観察が複製される。観察の列が実際に見た回数（同じ状態・方法・根拠・値の種類を 1 つと数える）。"


def dominant_status(c: Counter) -> str:
    """面の色：一番多い状態（同数は measured > estimated > unknown の順）。何も無ければ unknown。"""
    if not c:
        return "unknown"
    return max(STATUSES, key=lambda s: (c.get(s, 0), -STATUSES.index(s)))


# --- 3D の線 ------------------------------------------------------------------------------
