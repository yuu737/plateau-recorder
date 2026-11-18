# plateau-recorder 作業規約

スマホの写真から、PLATEAU の仕様どおりの LOD3（CityGML 2.0＋i-UR 3.1）を作る道具。窓・扉・壁の値ごとに、確かさ（1σ）・出どころ・根拠（写真と画素の範囲）を持たせ、公共測量の地図情報レベルで等級を付ける。PLATEAU AWARD 2026 の応募作品。

campus-recorder（`D:\BuiltwithClaude`、Built with Claude: Japan 2026 の提出物）を 2026-11-17 にコピーして始めた。**元のリポジトリは書き換えない。** 設計と、元から何を捨てたかは `docs/設計_PLATEAU版.md`。

## 絶対の決まり
1. **根拠の無い値は書かない。** 値には σ（1σ、m）と出どころと根拠を付ける。測れなければ値は null にして撮影・計測を頼む。決めつけない（仮の幅・なめらかな勾配・平面の当てはめをしない）。
2. **元データは棟ごとの JSON 1 つ**（`data/buildings/*.json`。設計は `scripts/design_<棟>.py` が出す）。CityGML・図面・重ね画像はそこから生成する。生成物を手で直さない。
3. **寸法は写真 1 枚ずつの正面化で読む。** 何枚もの写真を面に貼った合成（`coverage_<面>_photo.png`）は、写真ごとの位置合わせの誤差と手前の物が混ざる。表示と撮れた範囲の確認だけに使い、寸法の根拠にしない（11/17、タワー75 の東面で他の棟が貼り付いていたため）。
4. **写真の元ファイルは共有**：`data/photos/20261103`・`20261103b`・`20261104`・`20261105`・`send` は、元のリポジトリへのジャンクション。**中を書き換えない・消さない・改名しない。** 新しい写真は新しいフォルダに入れる。索引 `data/photos/index.json` と登録の台帳 `registry.json` はこのリポジトリの物（元とは別）。
5. **使ってよい資料**：自分で撮った写真、PLATEAU、OpenStreetMap、国土地理院の航空写真と標高。出典は README に書く。
   - **禁止**：Google 由来のデータ（地図・航空写真・3D・ストリートビュー）。図面をなぞって形を作ること。
   - 大学の図面・公開の設計図は、**答え合わせにだけ**使う（使うかは利用者が決める）。作る側の係（Claude）には見せない。
   - ハッカソンの期間前に撮った写真（2026-09-07〜10-12）を入力に使うかは、利用者の決定待ち。決まるまでは期間中の写真だけ。
6. **個人情報**：写真の公開版では、掲示物・人名・電話番号・顔・ナンバーを灰色の箱＋理由の文字で塗りつぶす（`recorder/privacy.py`）。研究室が分かる写真は公開しない。
7. **答え合わせの正解は、作る側から隠す。** 正解（巻尺・距離計・人が囲んだ窓）は `data/truth/` に置き、係の依頼文に入れない。判定の規則は、結果を見る前に文書に書いて固定する。

## 進め方
- 日本語で分かりやすく。専門用語は噛み砕く。
- 座標：JGD2011 平面直角座標系 VII 系（EPSG:6675）、原点 北緯35.1355° 東経136.9755° からの m（x＝東、y＝北）。棟の z は棟の `ground_z`（標高）からの m。
- Python は `.venv`（3.13、システムの site-packages を共有）。`.venv/Scripts/python -I -X utf8 -m recorder …`。
- テスト：`.venv/Scripts/python -X utf8 -m pytest -q -m "not browser and not gpu"`。
- 書き出し（11/19〜）：`.venv/Scripts/python -I -X utf8 -m recorder lod3 build --windows photos --lod 3.0 3.1 3.2 --wall-texture --roof-texture` → `out/lod3/dataset_lod30`・`_lod31`・`_lod32`（どれも LOD0・LOD1 は PLATEAU のまま、LOD2 と LOD3.x は棟 JSON から。壁・窓・扉に公開版の写真の質感、屋根に国土地理院の航空写真）。質感の元の面の画像は `recorder facade public <棟>`（`out/facade_public/`、塗りつぶしのマスク付き）。表示は `viewer/index.html?lod=30|31|32` の「見た目」。
  - 質感の合成は見た目だけに使う（寸法の根拠にしない。決まり 3）。方法と先行研究は `recorder/lod3/wall_texture.py` の冒頭と `D:\PLATEAU\lod3\research\texture_prior_art.md`。
- PLATEAU 公式の変換ツールで開けるか：`recorder lod3 build` の後に WSL で `bash /mnt/d/PLATEAU/plateau-recorder/scripts/check_plateau_converter.sh`（nusamai 0.1.14 の Linux 版。結果と限界は `docs/確認_PLATEAU変換ツール_11-17.md`）。3D Tiles の表示は `viewer/tiles.html`。

## 構成
| 場所 | 中身 |
|---|---|
| `recorder/` | Python パッケージ（写真の取り込み・正面化・登録・測量・棟 JSON の検証・CityGML の書き出し・確かさ） |
| `data/buildings/` | 棟ごとの JSON（唯一の元データ） |
| `data/photos/` | 写真（共有・git 管理外）と索引・台帳 |
| `data/reports/` | 係の報告 |
| `out/` | 生成物。git 管理外 |
| `docs/` | 設計・撮影ガイド・実験の記録 |
| `tests/` | テスト |
