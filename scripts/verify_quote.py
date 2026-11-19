"""引用が出どころのページの原文にあるかを確かめる（建築年などを Web で調べる係が使う）。

WebFetch の答えはページを要約したもので、引用が原文の字面と一致する保証がない。
ここではページそのもの（HTML・PDF・テキスト）を取ってきて、空白と全角・半角の違いを除いたうえで、
引用の文字列が含まれるかを見る。表の升目をまたぐ引用（「1985」「名城大学本部棟竣工」）も、
空白を除くので続けて読める。

使い方：
  python -I -X utf8 scripts/verify_quote.py URL "引用1" ["引用2" ...] [--save DIR]

結果は引用ごとに 1 行の JSON：
  {"url", "status", "kind", "encoding", "text_len", "quote", "found", "context"}
見つからないときは、引用と最も長く一致した部分（best）と一致の割合（best_ratio）も出す。
text_len が極端に短いページは、中身をスクリプトで描いている（取れていない）恐れがある。

--save DIR を付けると、取ってきた本文（正規化前のテキスト）を DIR に保存する。
ファイルの頭に URL・取った時刻・元のバイト列の sha256 を書く（後から同じ根拠を見直すため）。

終了コード：0＝全部見つかった、1＝見つからない引用がある、2＝ページを取れなかった。
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import html
import io
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (compatible; plateau-recorder verify_quote)"
MAX_BYTES = 30 * 1024 * 1024
CONTEXT = 40

_BLOCK = re.compile(r"<\s*(br|/p|/div|/li|/tr|/td|/th|/h[1-6]|/dt|/dd)\b[^>]*>", re.I)
_DROP = re.compile(r"<(script|style|noscript)\b.*?</\1\s*>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[\s​‌‍﻿]+")


def fetch(url: str) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja,en;q=0.5"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310（URL は調べ役が見つけた出どころ）
        body = r.read(MAX_BYTES + 1)[:MAX_BYTES]
        return r.status, body, r.headers.get("Content-Type", "")


def _charset(body: bytes, ctype: str) -> list[str]:
    cands = []
    m = re.search(r"charset=([\w\-]+)", ctype, re.I)
    if m:
        cands.append(m.group(1))
    m = re.search(rb"<meta[^>]+charset=[\"']?([\w\-]+)", body[:8192], re.I)
    if m:
        cands.append(m.group(1).decode("ascii", "ignore"))
    return cands + ["utf-8", "cp932", "euc_jp"]


def to_text(body: bytes, ctype: str) -> tuple[str, str, str]:
    """(本文, 種類, 文字コード) を返す。"""
    if body[:5] == b"%PDF-" or "pdf" in ctype.lower():
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(body))
        return "\n".join((p.extract_text() or "") for p in reader.pages), "pdf", "-"
    for enc in _charset(body, ctype):
        enc = {"shift_jis": "cp932", "sjis": "cp932", "x-sjis": "cp932"}.get(enc.lower(), enc)
        try:
            s = body.decode(enc)
            break
        except (LookupError, UnicodeDecodeError):
            continue
    else:
        enc, s = "utf-8(replace)", body.decode("utf-8", "replace")
    if "<" in s[:2048] and ">" in s[:2048]:
        s = _DROP.sub(" ", s)
        s = _BLOCK.sub("\n", s)
        s = html.unescape(_TAG.sub("", s))
        return s, "html", enc
    return s, "text", enc


def norm(s: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", s))


def check(text_n: str, quote: str) -> dict:
    q = norm(quote)
    i = text_n.find(q) if q else -1
    if i >= 0:
        return {"found": True, "context": text_n[max(0, i - CONTEXT): i + len(q) + CONTEXT]}
    m = difflib.SequenceMatcher(None, text_n, q, autojunk=False).find_longest_match(0, len(text_n), 0, len(q))
    best = q[m.b: m.b + m.size]
    return {
        "found": False,
        "best": best,
        "best_ratio": round(m.size / max(1, len(q)), 2),
        "context": text_n[max(0, m.a - CONTEXT): m.a + m.size + CONTEXT] if m.size else "",
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url")
    ap.add_argument("quotes", nargs="+")
    ap.add_argument("--save", type=Path, default=None)
    a = ap.parse_args(argv)

    base = {"url": a.url}
    try:
        status, body, ctype = fetch(a.url)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(json.dumps({**base, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return 2
    try:
        text, kind, enc = to_text(body, ctype)
    except Exception as e:  # PDF の壊れなど
        print(json.dumps({**base, "status": status, "error": f"本文を取り出せない: {e}"}, ensure_ascii=False))
        return 2
    text_n = norm(text)
    base.update(status=status, kind=kind, encoding=enc, text_len=len(text_n))

    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha1(a.url.encode()).hexdigest()[:12] + ".txt"
        head = (
            f"url: {a.url}\nfetched: {dt.datetime.now().isoformat(timespec='seconds')}\n"
            f"sha256: {hashlib.sha256(body).hexdigest()}\nkind: {kind} encoding: {enc}\n---\n"
        )
        (a.save / name).write_text(head + text, encoding="utf-8")
        base["saved"] = str(a.save / name)

    ok = True
    for q in a.quotes:
        r = check(text_n, q)
        ok &= r["found"]
        print(json.dumps({**base, "quote": q, **r}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
