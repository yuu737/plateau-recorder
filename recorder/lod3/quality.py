"""値ごとの「確かさ」の物差し。

1. 精度：棟 JSON の ±（tol）を標準偏差とみなし（控えめな読み）、作業規程の準則の地図情報レベルの標準偏差
   （PLATEAU 標準製品仕様書 6.3.3 の表）と比べて等級にする。水平（位置・幅・長さ）と標高（高さ）は別の基準。
   - 500：水平 0.25 m・標高 0.25 m 以内
   - 1000：水平 0.70 m・標高 0.33 m 以内
   - 2500：水平 1.75 m・標高 0.66 m 以内（PLATEAU の基本）
   ここでの等級は「建物の中での位置の確かさ」。地図上の位置は PLATEAU の足元（名古屋 2022 は地図情報レベル 2500、
   srcScaleLod0=1）に従う。
2. 出どころ：何をして値を得たか。「実測／推定」の言葉は使わない。PLATEAU のデータ品質のコード
   （DataQualityAttribute_geometrySrcDesc）に対応させる。
"""
from __future__ import annotations

import re

LEVELS = [("500", 0.25, 0.25), ("1000", 0.70, 0.33), ("2500", 1.75, 0.66)]
GRADE_ORDER = ["500", "1000", "2500", "2500外", "不明"]
GRADE_JA = {"500": "地図情報レベル500相当", "1000": "地図情報レベル1000相当", "2500": "地図情報レベル2500相当",
            "2500外": "2500に届かない", "不明": "±の記録なし"}
FOOTPRINT_LEVEL = "2500"       # 名古屋市 2022 の足元（srcScaleLod0=1）
FOOTPRINT_SIGMA_H = 1.75

# 出どころ：弱い順（1 つの値に複数当てはまれば弱い方を採る）
# (key, 表示, PLATEAU のコード)。弱い順。
LINEAGE = [
    ("assume", "仮に置いた", "901"),
    ("pattern", "同じ作りの所から写した", "901"),
    ("ratio", "写真の中の比率で読んだ（縮尺なし）", "901"),
    # PLATEAU 版（11/17）：外壁の読み係（Claude）が、正面化した写真の目盛りで位置を読んだ（読みの揺れ ±0.3 m 程度）
    ("claude_read", "正面化した写真の目盛りで読んだ（外壁の読み係・Claude）", "801"),
    ("photo_noscale", "正面化した写真で読んだ（縮尺の記録なし）", "801"),
    ("photo_plateau", "正面化した写真で読んだ（縮尺は PLATEAU）", "801"),
    # PLATEAU 版（11/17）：写真 1 枚ずつ面に正面化し、窓を検出して写真どうしで照合した（recorder facade）。σ は写真どうしのばらつき
    ("photo_multi", "正面化した写真を何枚も使い、窓を検出して写真どうしで照合した", "801"),
    ("public", "既存の地図データそのまま（PLATEAU・国土地理院・OSM）", "000"),
    ("photo_tape", "正面化した写真で読んだ（縮尺は巻尺）", "801"),
    ("tape", "巻尺から縮尺をつないだ", "801"),
    ("count", "写真で数えた", "801"),
]
PAT = {
    "assume": r"置いた|仮に|未測|目安|見えていない|見えない|見た目|読めない",
    "pattern": r"仮説|same_block|同じと置|を写した|写した。|と同じと(?:置|見)",
    "ratio": r"比率|に対する比|との比|の約 ?[\d.]|割合|ほど|^\s*\(b\)",
    "photo": r"px|正面化|coverage_|画素|^\s*\(a\)",
    "tape": r"鎖|巻尺|tape|柱92cm",
    "count": r"数え|枚数|段数|個数",
    "public": r"PLATEAU|plateau:|5 ?m 標高|地理院|gsi:|航空写真|OSM|許容 ?3 ?%",
}
# K-203 は「後から鎖（巻尺）の縮尺に比例で直した」という注記。値の取り方そのものではないので出どころの判定から外す
K203 = re.compile(r"（K-203[^）]*）")
LINEAGE_JA = {k: ja for k, ja, _ in LINEAGE} | {"none": "記録なし"}
LINEAGE_CODE = {k: c for k, _, c in LINEAGE} | {"none": "999"}
LINEAGE_RANK = {k: i for i, (k, *_) in enumerate(LINEAGE)} | {"none": -1}

V_KEYS = re.compile(r"(^|[._])(z|z0|z1|z_bottom|z_top|top_z|bottom_z|height|h|parapet_top|ground_z|ground_offset|floor_pitch|rise)($|[._\[])")


def axis_of(key: str) -> str:
    """値の名前から 水平 か 標高 か。"""
    return "v" if V_KEYS.search(key) else "h"


def grade(sigma: float | None, axis: str) -> str:
    if sigma is None:
        return "不明"
    for name, h, v in LEVELS:
        if sigma <= (h if axis == "h" else v) + 1e-9:
            return name
    return "2500外"


def worst(*gs: str) -> str:
    return max(gs, key=GRADE_ORDER.index) if gs else "不明"


def lineage_of(m: dict | None) -> str:
    """根拠つきの値 → 出どころの区分（弱い方を採る）。"""
    if not isinstance(m, dict):
        return "none"
    text = " ".join(str(m.get(k, "")) for k in ("method", "note", "basis"))
    text += " " + " ".join(m.get("evidence") or []) if isinstance(m.get("evidence"), list) else ""
    if m.get("basis") == "hypothesis":
        return "pattern"
    links = {c.get("link") for c in (m.get("chain") or []) if isinstance(c, dict)}
    rescaled = bool(K203.search(text))
    text = K203.sub("", text)
    hit = {k for k, pat in PAT.items() if re.search(pat, text, flags=re.M)}
    if rescaled and "photo" in hit:
        hit.add("tape")                    # 正面化で読んだ値を巻尺の鎖の縮尺に直した＝縮尺は巻尺
    if "assume" in hit:
        return "assume"
    if "pattern" in hit or "assume" in links:          # 鎖の assume＝「別の面・塊と同じと置く」輪
        return "pattern"
    if "ratio" in hit:
        return "ratio"
    tape = "tape" in hit or "tape" in links
    if "photo" in hit:
        return "photo_tape" if tape else ("photo_plateau" if "public" in hit or "登録" in text else "photo_noscale")
    if tape:
        return "tape"
    if "count" in hit or m.get("unit") in COUNT_UNITS:
        return "count"
    if "public" in hit:
        return "public"
    return "none"


# campus-recorder の ± は「物差しの許容を積んだ幅」（輪ごとの誤差の和＝最悪に近い幅。docs/手順_エージェント.md）。
# 標準偏差ではないので、正規分布の 95% の幅とみなして σ = ± × 0.5 とする（工学でふつうの読み）。
# 1.0 にすると「± をそのまま σ」＝いちばん厳しい読み。答え合わせ（正解が ± に入る割合が 95% 前後か）で確かめる。
SIGMA_PER_TOL = 0.5


def sigma_of(m: dict | None) -> float | None:
    """± から標準偏差（σ = ± × SIGMA_PER_TOL）。数えた値（個・階・段）で ± が無いものは 0。"""
    if not isinstance(m, dict) or m.get("status") == "unknown":
        return None
    t = m.get("tol")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t) * SIGMA_PER_TOL
    if m.get("unit") in COUNT_UNITS:
        return 0.0
    return None


COUNT_UNITS = ("個", "階", "段", "列", "枚", "本")


def is_length(m: dict | None) -> bool:
    return isinstance(m, dict) and (m.get("unit") in (None, "m"))
