#!/usr/bin/env bash
# GitHub Pages（gh-pages ブランチ）に置く表示画面を、書き出し済みの out/ から写す。
# 使い方: bash scripts/make_pages.sh [写し先]（既定は ../plateau-recorder_pages）
# 写すのは表示画面（viewer/）と、表示に要る生成物だけ（CityGML・質感・根拠の写真の切り抜き・3D Tiles）。
# 元の写真・棟 JSON・作業の記録は写さない。切り抜きと質感は塗りつぶし済みの公開版の写真から作った物。
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-../plateau-recorder_pages}"
mkdir -p "$DEST"
# 前回の写しを消す（.git は残す）
find "$DEST" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

mkdir -p "$DEST/viewer" "$DEST/out/lod3" "$DEST/out/lod3_converted"
cp viewer/index.html viewer/tiles.html "$DEST/viewer/"
for d in dataset_lod30 dataset_lod31 dataset_lod32; do
  cp -r "out/lod3/$d" "$DEST/out/lod3/"
done
cp out/lod3/thumbs.json "$DEST/out/lod3/"
cp -r out/lod3/thumbs "$DEST/out/lod3/"
cp -r out/lod3_converted/3dtiles "$DEST/out/lod3_converted/"
touch "$DEST/.nojekyll"

cat > "$DEST/index.html" <<'HTML'
<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLATEAU Recorder</title>
<meta http-equiv="refresh" content="0; url=viewer/index.html?lod=30">
</head><body><p><a href="viewer/index.html?lod=30">PLATEAU Recorder の表示画面へ</a></p></body></html>
HTML

# 応募フォームの文では URL の直後に全角の「）」が続き、メールの自動リンクがその先の文までリンクにする。
# そうして壊れた URL（…/plateau-recorder/）とソースコード…）で来た人を、404 から表示画面へ送る
cat > "$DEST/404.html" <<'HTML'
<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLATEAU Recorder</title>
<!-- URL の後ろに全角の括弧や文が付いて壊れたリンク（例：…/plateau-recorder/）とソースコード…）から来た人を、表示画面へ送る -->
<meta http-equiv="refresh" content="0; url=/plateau-recorder/">
<script>location.replace("/plateau-recorder/");</script>
</head><body><p>ページが見つかりませんでした。<a href="/plateau-recorder/">PLATEAU Recorder の表示画面へ</a></p></body></html>
HTML

cat > "$DEST/README.md" <<'MD'
# PLATEAU Recorder（表示画面）

https://github.com/yuu737/plateau-recorder の表示画面です。中身は `bash scripts/make_pages.sh` で書き出し済みの生成物から写した物です。

- 表示画面：`viewer/index.html?lod=30|31|32`、3D Tiles（PLATEAU 公式の変換ツール nusamai で変換）：`viewer/tiles.html`
- CityGML の建物は「3D 都市モデル（Project PLATEAU）名古屋市」（国土交通省）を引き継いで LOD を足した物です。屋根の質感は国土地理院「全国最新写真（シームレス）」。
- 写真の切り抜き・壁の質感は、名城大学（広報部門）の許可を得て公開している写真の公開版（掲示物・人名・電話番号・顔・ナンバーを塗りつぶし済み）から作りました。
- コードは MIT License、写真と生成物の条件は本体のリポジトリの `LICENSE-DATA.md` に従います。
MD
du -sh "$DEST"
