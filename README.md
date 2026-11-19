# PLATEAU Recorder

スマホの写真と巻尺から、PLATEAU の仕様どおりの建築物 LOD3（CityGML 2.0＋i-UR 3.1）を作り、窓 1 枚ずつに「どこまで信じてよいか」と根拠の写真を付ける道具です。PLATEAU AWARD 2026 の応募作品です。

**デモ（表示画面）**：https://yuu737.github.io/plateau-recorder/
窓を押すと、等級・確かさ（1σ）・出どころ・根拠の写真の切り抜きが出ます。「見た目」に切り替えると写真の質感、LOD は 2・3.0・3.1・3.2 を切り替えられます。

| | 中身 |
|---|---|
| 対象 | 名城大学 天白キャンパスの 4 棟（本部棟・附属図書館・タワー75・共通講義棟南） |
| 入力 | 自分で撮ったスマホの写真（公開版 407 枚）、巻尺、PLATEAU 名古屋市の建物（LOD0・LOD1・属性） |
| 出力 | 区画 52365768 の CityGML。LOD0・LOD1 と属性は PLATEAU のまま引き継ぎ、LOD2 と LOD3.0／3.1／3.2 を足す（`out/lod3/dataset_lod30`・`_lod31`・`_lod32`） |
| 検査 | i-UR 3.1 の XSD 検査に合格。4 棟とも立体が閉じている。PLATEAU 公式の変換ツール（nusamai 0.1.14）で 3D Tiles に変換できる |
| 窓の確かさ | 窓 552 枚のうち、地図情報レベル 500 相当 60 枚（11%）、1000 以内 127 枚（23%）、2500 以内 207 枚（38%）。本部棟は 118 枚で 47%・84%・100%（LOD3.0 の書き出しの集計） |

## 何をしているか

1. **写真は貼り合わせず、1 枚ずつ測る。** 写真を 1 枚ずつ壁の面に正面化し、窓の縁を写真ごとに拾って、写真どうしで突き合わせます。写真どうしの食い違いを確かさ（1σ）にします（`recorder/facade.py`）。
2. **縮尺は巻尺からつなぐ。** 巻尺で 1 回測れる物（タイル、柱型の幅など）を物差しにして、柱間・窓へ縮尺をつなぎます（`recorder measure tape|span|chain`）。
3. **値ごとに根拠を持つ。** 窓・扉・壁の値ごとに、σ、公共測量の地図情報レベル（500／1000／2500）での等級、出どころ（PLATEAU のデータ品質のコード 801／802／000 に対応）、根拠の写真と範囲を持ちます（`recorder/lod3/`）。
4. **PLATEAU に戻せる形で書く。** 元の PLATEAU の建物を引き継いで LOD を足し、PLATEAU の標準製品仕様どおりの CityGML に書き出します。
5. **見た目は写真の質感で。** 壁・窓・扉は公開版の写真から合成した質感、屋根は国土地理院の航空写真です。質感は見た目だけに使い、寸法の根拠にはしません（`recorder/lod3/wall_texture.py`・`atlas.py`）。

## Claude の使い方

- **外壁の読み係**（`.claude/agents/gaiheki.md`）：窓はくり返すので、写真の位置合わせが柱何本分かずれても検出器は気づけません。読み係は垂れ幕などを手がかりにずれた写真を見つけて外し、外装材も読みます（`docs/実験_外壁の読み係_Sonnet対Opus_11-17.md`）。
- **保護係**（`.claude/agents/hogo.md`）：公開版の写真で、掲示物・人名・電話番号・顔・ナンバーの位置を指します。塗りつぶしはコード（`recorder/privacy.py`）が行います。
- 位置・寸法・等級の計算はコードが行い、Claude は「くり返しの中のずれを見抜く」「画像を読む」所を受け持ちます。ほかの係は前作 campus-recorder から引き継いだ物です。
- 建築年・構造種別・名称を、写真と持ち主の公開ページから根拠付きで埋める試行を 2 棟で行いました（`docs/試行_建築年_本部棟_11-19.md`・`docs/試行_建築年_図書館_11-19.md`・`docs/試行_構造と名称_11-19.md`、引用の照合は `scripts/verify_quote.py`）。CityGML にはまだ書いていません。

## 動かし方（作者の環境）

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]" opencv-python-headless shapely pyproj requests
# LOD2・LOD3.0〜3.2 の書き出し（質感つき）
.venv/Scripts/python -I -X utf8 -m recorder lod3 build --windows photos --lod 3.0 3.1 3.2 --wall-texture --roof-texture
# 表示画面
python -m http.server 8771   # http://localhost:8771/viewer/index.html?lod=30
# テスト
.venv/Scripts/python -X utf8 -m pytest -q -m "not blender and not browser and not gpu"
```

元の写真はリポジトリに入れていません（公開版だけ）。書き出しの手順は作者の環境で確かめた物で、まっさらな環境で公開版の写真から作り直せるかは確かめていません。PLATEAU 公式の変換ツールでの確認は `scripts/check_plateau_converter.sh` と `docs/確認_PLATEAU変換ツール_11-17.md` にあります。

## 正直な限界

- 申告した σ が当たっているかの答え合わせ（巻尺・距離計による独立の計測）は、まだしていません。
- 写真から拾った窓に置き換えたのは本部棟の正面（f6）だけです。ほかの面は棟の記録の窓のままで、写真は検算に使いました。
- タワー75 の縦に連なる細い窓は、検出器で取れていません。等級が 2500 に届かない窓の多くはタワー75 です。
- 公式の変換ツールで 3D Tiles にすると、窓・壁ごとの属性は建物にまとまって消えます。
- 対象は 4 棟だけです。

## データと出典・ライセンス

- **コード**：MIT License（[`LICENSE`](LICENSE)）。
- **写真・棟 JSON・生成物**：MIT ではありません。[`LICENSE-DATA.md`](LICENSE-DATA.md) の条件に従ってください。
- **自分で撮った写真**：名城大学（広報部門）の許可（2026-11-04）を得て公開しています。元の写真は入れず、縮小した公開版だけを入れています。掲示物・人名・電話番号・顔・ナンバーは灰色の箱で塗りつぶしています（記録は `data/reference/index.json`）。研究・研究室が分かる写真は公開していません。
- **3D 都市モデル（Project PLATEAU）名古屋市**（国土交通省）：建物の足元・高さ・属性。LOD0・LOD1 と属性はそのまま引き継いでいます。
- **国土地理院**：屋根の質感に「全国最新写真（シームレス）」を使っています（出典：国土地理院）。
- **OpenStreetMap**：© OpenStreetMap contributors（ODbL 1.0）。

前作 campus-recorder（Built with Claude: Japan 2026、https://github.com/yuu737/campus-recorder ）をもとに、PLATEAU 向けに作り直しました。歩く画面・Blender の模型などは外しています（`docs/設計_PLATEAU版.md`）。
