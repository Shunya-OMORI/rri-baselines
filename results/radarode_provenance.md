# radarODE-MTL 再現実装の出所の一覧 (2026-10-04)

LOSO 200 エポックの 18.48 ms(`radarode_loso200_final.md`)を出した構成の部品と設定のすべてについて、どの資料に従ったかを示す。
コードは `src/rri_baselines/radarode/`(データ・学習)と `radarODE-MTL/repro/model.py`(モデル)。
論文と公開実装の 2 者だけを比べた表は `radarode_protocol_diff.md`(10-02)。この文書は CFT-RFcardi のコードと配布重みを加え、全項目に出所を付けたもの。

## 資料の区分

| 記号 | 資料 | 中身 |
|---|---|---|
| **公** | radarODE-MTL 公開実装(GitHub ZYY0844/radarODE-MTL) | `Projects/radarODE_plus/` の Python コードと、SST を作る `MMECG_to_SST.m` |
| **論** | radarODE-MTL 論文(IEEE TIM 2025) | 本文と Table I・V |
| **C実** | CFT-RFcardi の実装(同じ著者の後継研究。GitHub の CFT-RFcardi) | `RFcardi_Transfer_Learning/` のコードと、MMECG 用のデータ生成ノートブック `utils/data_generation.ipynb` |
| **C重** | CFT-RFcardi の配布重み `Model_saved/best_Super_100_mse.pt` | 重みの形から構造を読む。backbone・squeeze・ECG 波形の枝・Anchor の枝を含み、**PPI の枝は含まない** |
| **推** | 我々の推定 | 資料に書いていないか、資料どうしが食い違うので我々が決めたもの |
| **揃** | 我々のルールに揃えた変更 | 提案手法と同じ条件で比べるために、資料と違うことを承知で変えたもの |

CFT-RFcardi の論文(Zhang et al., "From High-SNR Radar Signal to ECG: A Transfer Learning Model with Cardio-Focusing Algorithm for Scenarios with Limited Data", IEEE TMC 2025)は
"the same backbone, ECG decoder and hyperparameters as in our previous open-sourced work [8]" と書き、[8] は radarODE-MTL である。
**C実・C重を radarODE-MTL の構成の証拠として使う根拠はこの 1 文**である。

## 出所の表

### データと分割

| 項目 | 我々の実装 | 出所 | 他の資料との違い |
|---|---|---|---|
| 試行 | MMECG の 91 試行すべて | 揃 | 公はコードの `np.arange(1, 89)` で試行 89〜91 を読まない |
| 試行と参加者の対応 | 生 `.mat` の `data.id` | データ本体 | — |
| test | 1 参加者の全試行 | 論 | 公は試行番号で分けるので、test の参加者の他の試行が学習に入る(リーク) |
| val | test の次の参加者 1 名、学習は残り 9 名 | **推**・揃(提案手法と同じ) | 論は "the ten other subjects alternatively selected for training or validation" で、選び方の具体は無い。公は val が無い |
| 学習の回数 | 1 回(seed 42) | 揃 | 論は 5 回の繰り返し |

### 入力(SST)

| 項目 | 我々の実装 | 出所 | 他の資料との違い |
|---|---|---|---|
| 時間周波数変換 | MATLAB `wsst(RCG, 200, 'VoicesPerOctave', 10)` の Python 移植 | 公(`MMECG_to_SST.m`。C実にも同じファイル) | 移植は MathWorks の公式リファレンスと wsst.m の写しで照合済み(`reports/wsst_check/`) |
| 周波数の範囲 | 140 行のうち高い側の 71 行(約 0.72〜100 Hz) | 公(`.m` の `freq_len/2:freq_len`) | 09-28 までは低い側を取り違えていた(旧 24.18 ms) |
| 時間方向 | 200 Hz → 30 Hz | 公(`.m` の `resample`) | 我々は `resample_poly`。数値は一致を確認済み |
| 試行全体の正規化 | 測定点ごとに試行全体で [0, 1] | 公(`.m`) | 試行全体を見る非因果の処理 |
| 周波数の向き | 上下反転(行 0 が 100 Hz) | C実(ノートブックの `load_sst` の `np.flip`) | 公にはデータ生成のコードが無い |
| 窓 | 4 s = 120 フレーム | 論・C実 | — |
| 窓ごとの正規化 | 窓ごと・測定点ごとに [0, 1] | C実(`sst_norm_01`) | — |
| **測定点の数** | **50 点すべて** | 論("the 50 synchronous radar signals")・公(`.m` が 50 点を出す) | **C実と C重は 10 点**。C実は窓ごとに `DTM_info` から 10 点を読み、C重の最初の段の入力は 10。CFT-RFcardi 論文はこの選び方を CFT(合成テンプレートとの一致で 10 点を選ぶ)と呼ぶ。下の「backbone の再現度」も参照 |
| 学習窓の刻み | 0.8 s | C実(`step = 160` 標本) | 論は 1 s |
| 雑音の付加 | なし | 公の意図(`main.py` の `aug_snr = 100` と注記 "100 for no noise") | 公は実際には引数が渡らず 0 dB の雑音を足す |

### 正解ラベル

| 項目 | 我々の実装 | 出所 | 他の資料との違い |
|---|---|---|---|
| 対象の拍 | 窓の中央をまたぐ R 波から R 波まで | 論("the middle cardiac cycle")・揃(提案手法の目標量) | C実は中央の R 波を選ぶつもりのコードだが、R 波の絶対位置を窓内の中央 400 と比べる不具合があり、窓の開始が 400 標本以降では窓の最初の R 波を含む心周期になる |
| R 波の位置 | 提案手法と同じ R 波 | 揃 | 論は NeuroKit2 |
| ECG 波形の正解 | 対象の拍の ECG を 200 点に線形補間 | 公(`down_sample`) | — |
| PPI の正解 | 対象の拍の ECG を −10 で長さ 260 まで埋める。長さ L をクラス L−1 と読む | 公 | — |
| Anchor の正解 | 窓の中の R 波の位置に 1 を立てた 800 点 | 公・C実 | — |
| 長さ 260 を超える拍の除外 | 除外する | 公の制約(260 は PPI の出力の長さ) | 我々のデータでは 0 件 |

### モデル

| 項目 | 我々の実装 | 出所 | 他の資料との違い |
|---|---|---|---|
| backbone の構造 | 変形畳み込みの ResNet 4 段と縮小層、squeeze | 公(部品をそのまま import) | C重を strict で読み込める(10 点入力のとき、70 個の重みの名前と形がすべて一致) |
| backbone の段幅 | 32 / 128 / 256 / 512 | C実(`dcnresnet_backbone.py`。128〜1024 の行はコメントアウト)・C重(重みの形) | **公は 128 / 256 / 512 / 1024**。論は段幅を書かず、Table V の params だけ |
| ECG 波形の枝 | 転置畳み込み 3 層 + 畳み込み 4 層(+ ODE) | 公 | **C実と C重は転置畳み込み 4 層 + 畳み込み 3 層**。177 個の重みのうち 152 個だけ形が一致する。最初の畳み込みの入力を 64 にしたのは段幅 512 に合わせた**推** |
| ODE の融合 | あり | 論(報告している構成)・公のコード中の注記 "temp+ode, give the best result" | 公の既定は ODE なし |
| PPI の枝 | 畳み込み 4 層 + 転置畳み込み 3 層 + 全結合 | C実(`PPI_decoder.py`。全結合の入力は 512×4) | C重には PPI の枝が無く、重みでは確かめられない。論の記述は「転置畳み込み 2 層」でコードの 3 層と違う。公は段幅 1024 前提の直書き |
| Anchor の枝 | C実と同じ | C実(`anchor_decoder.py`)・C重 | C重を strict で読み込める(57 個すべて一致)。公は段幅 1024 でしか動かない |

### 損失・読み出し

| 項目 | 我々の実装 | 出所 | 他の資料との違い |
|---|---|---|---|
| ECG 波形の損失 | [0, 1] に正規化した後の MSE | 公 | 論は RMSE |
| PPI の損失 | 長さ 260 のクラス分類の交差エントロピー | 公・論 | — |
| Anchor の損失 | MSE | 公 | 論は交差エントロピー |
| タスクの重み | 等重み(EW) | 論(Table I の PPI 最良行 8.0 ms) | 論の提案は EGA。公の `main.py` も EGA |
| マルチタスクの枠組み | LibMTL のハード共有(HPS) | 公 | — |
| PPI の読み出し | argmax + 1 標本 | **推**(損失がクラス L−1 を正解とするので、それと整合させた) | 公の指標の式は 2 標本ずれる。その式の値 22.13 ms も併記している |

### 最適化

| 項目 | 我々の実装 | 出所 | 他の資料との違い |
|---|---|---|---|
| 最適化法 | SGD、学習率 5e-3、weight decay 5e-4、momentum 0.937 | 論・公(一致) | C実が "same hyperparameters" と書く |
| バッチ | 32 | 論 | 公の `main.py` は 22 |
| エポック | 200 | 論・公 | — |
| 学習率の周期 | コサイン、周期 100(200 エポックで下がって戻る)、下限は学習率の 1 % | 公(`main.py` の `T_max = 100`、`eta_min`) | 論に記載なし |

### 選択と評価

| 項目 | 我々の実装 | 出所 | 他の資料との違い |
|---|---|---|---|
| 最良エポックの選び方 | val の PPI 誤差が最小のエポック | 論(val を置くと書く。選ぶ指標の記述は無い)・揃 | 公は test で選ぶ |
| test の窓 | 0.2 s 刻みの全窓 | 揃(提案手法と同じ) | 論は 1 s 刻みと読める |
| 集計 | 参加者ごとの MAE を 11 名で平均 | 揃 | 窓で平均した値も `aggregation_macro_micro.md` に出した |

## backbone の再現度

論文 Table V の backbone の値と、各構成の値を並べる。パラメータは squeeze を含まない。MACs は thop、FLOPs は torch の計数。

| 構成 | 入力の測定点 | 段幅 | params | MACs | FLOPs | 配布重みとの関係 |
|---|---|---|---|---|---|---|
| 論文 Table V | 記載なし | 記載なし | 4.81 M | 1.18 G | 2.37 G | — |
| CFT-RFcardi の配布重み | 10 | 32 / 128 / 256 / 512 | 4.21 M | — | — | 基準 |
| 再現、10 点 | 10 | 32 / 128 / 256 / 512 | 4.21 M | 1.197 G | 2.390 G | 70 個の重みの名前と形がすべて一致 |
| **再現、50 点(18.48 ms の構成)** | 50 | 32 / 128 / 256 / 512 | 4.24 M | 1.460 G | 2.913 G | 最初の段の入力だけ違う |
| 公開実装の段幅、50 点 | 50 | 128 / 256 / 512 / 1024 | 16.96 M | 5.48 G | — | 形が合わない |

各段の入出力(軸は「バッチ, チャネル, 周波数, 時間」)。チャネルの意味は入力と段 0 の後で違う。
入力のチャネルは**測定点 1 つ**の SST で、段 0 の出力のチャネルは**全測定点を畳み込みで混ぜた、学習した特徴**である。

| 段 | 入力 | 出力 | params(50 点) | params(10 点) |
|---|---|---|---|---|
| stage0 | (B, 50 または 10 [測定点], 71, 120) | (B, 32 [特徴], 72, 120) | 33,950 | 3,230 |
| downsample0 | (B, 32, 72, 120) | (B, 32, 36, 61) | 6,240 | 6,240 |
| stage1 | (B, 32, 36, 61) | (B, 128, 37, 61) | 22,802 | 22,802 |
| downsample1 | (B, 128, 37, 61) | (B, 128, 19, 31) | 98,688 | 98,688 |
| stage2 | (B, 128, 19, 31) | (B, 256, 20, 31) | 234,258 | 234,258 |
| downsample2 | (B, 256, 20, 31) | (B, 256, 10, 31) | 590,592 | 590,592 |
| stage3 | (B, 256, 10, 31) | (B, 512, 11, 31) | 894,482 | 894,482 |
| downsample3 | (B, 512, 11, 31) | (B, 512, 6, 31) | 2,360,832 | 2,360,832 |
| squeeze(params は上表に含めない) | (B, 512, 6, 31) | (B, 512, 31) | 1,573,376 | 1,573,376 |

読み取れること:

- **backbone の構造は、著者が実際に学習した重み(C重)と完全に一致する。** 段幅も C実のコードと C重の両方が 32 / 128 / 256 / 512 を示す。
- **10 点入力にすると、論文 Table V の計算量に 1〜2 % で一致する。** 50 点のままだと MACs は 24 % 多い。
- **params は 10 点でも論文の 4.81 M に届かない(−12 %)。** 公開実装の 58 コミットと著者の他リポジトリを調べたが、4.81 M になる構成は見つかっていない(`radarODE-MTL/repro/README.md`)。
- 我々は論文本文の「50 点」に従った。Table V の計算量は 10 点のモデルで測った可能性がある(推定)。
  入力の点数で変わるのは stage0 だけなので、10 点と 50 点の params の差は 0.7 % である。

## 我々の推定と、資料どうしの食い違いのまとめ

我々が推定で決めたのは次の 3 点である。

1. val の選び方(次の参加者 1 名)
2. PPI の読み出し(argmax + 1)
3. ECG 波形の枝の最初の畳み込みの入力の幅(64)

資料どうしが食い違い、そのうち 1 つを選んだのは次の 4 点である。

1. 測定点の数: 論の 50 点を選んだ。C実と C重は 10 点
2. 学習窓の刻み: C実の 0.8 s を選んだ。論は 1 s
3. ECG 波形の枝の層: 公を選んだ。C実と C重は層の並びが違う
4. 重みづけ: 論 Table I の PPI 最良行の EW を選んだ。論の提案と公は EGA

## 検証に使ったコマンド

```bash
# 配布重みを再現実装へ strict で読み込む (backbone 70/70, Anchor 57/57, ECG 波形 152/177)
cd radarODE-MTL && .venv/bin/python - <<'EOF'
import sys, torch; sys.path.insert(0, '.')
from repro.model import Backbone, ShapeDecoder, WIDTH_PRESETS, build_anchor_decoder
W = torch.load('../CFT-RFcardi/RFcardi_Transfer_Learning/Model_saved/best_Super_100_mse.pt', map_location='cpu')
for p, m in {'encoder.': Backbone(10, WIDTH_PRESETS['cft']), 'decoders.Anchor.': build_anchor_decoder(dim=512, variant='cft'),
             'decoders.ECG_shape.': ShapeDecoder(dim=512, use_ode=True)}.items():
    R = {p + k: v.shape for k, v in m.state_dict().items()}
    Wk = {k: v.shape for k, v in W.items() if k.startswith(p)}
    print(p, len(Wk), sum(R.get(k) == s for k, s in Wk.items()))
EOF
```
