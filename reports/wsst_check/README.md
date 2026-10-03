# SST の移植(rri_baselines/radarode/wsst.py)と MATLAB の照合(2026-10-03)

著者の生成: `radarODE-MTL/MMECG_to_SST.m` = 50 点それぞれ `wsst(RCG, 200, 'VoicesPerOctave', 10)` → 後半 71 行の絶対値 → `resample(·, 30, 200)` → 点ごとに試行全体で min-max.
MATLAB は手元・s7・nodame のどこにも無いため, MathWorks の公式リファレンス(wsst / cwtfreqbounds / resample, Wayback の 2019〜2026 年版)と,
公開されている wsst.m の写し 3 本(本体は 3 本で一致), resample.m・firls.m の写しを行単位で Python に写した参照実装と照合した.

| 段 | MATLAB(根拠) | 我々 | 判定 |
|---|---|---|---|
| 版 | R2024a から cwtfilterbank 方式(行数 127 になり `freq_len/2` が添字エラー)→ 71 行が出る著者の環境は R2023b 以前(推論) | 旧方式 | 一致 |
| ウェーブレット | 'amor' 既定(公式), exp(-(sω-6)^2/2)(写し) | 2 倍の定数 | min-max で消える |
| 端の拡張 | 'ExtendSignal' false が既定(公式 2019〜2025 年版) | 埋めなし | 一致 |
| 行・周波数 | floor(log2 N)-1 オクターブ × 10(公式), 昇順の対数等間隔(写し) | 同じ | f の差 0 |
| 瞬時周波数・再割り当て | iω 微分, floor の行割り当て(写し) | 同じ | 複素 SST の相対差 5e-10 |
| しきい値 | 位相 1e-8 cycles/sample と振幅 1e-8(写し) | 位相 1e-8 Hz のみ | 影響 1e-9 |
| 振幅の係数 | 1/nv(写し) | log2/nv | 定数倍. min-max で消える |
| resample | fir1/firls × kaiser β=5, 次数 2·10·max(p,q), p/q は約分(公式) | resample_poly(3, 20) | 係数の相対差 1.8e-16, 出力 5,326 点で一致 |

最終入力(窓ごとの min-max の後, 試行 1 点 1 の 217 窓): 同じ倍精度入力で最大差 4.7e-7. 学習キャッシュ(float32 の rcg_z から計算)とは
試行 1・79 の 5 点で最大 1.1e-3(1 画素), 平均 1e-7, 窓ごとの相関 0.99999998 以上. → 学習のやり直しは不要.

決着の手順: MATLAB(R2023b 以前, Wavelet + Signal Processing Toolbox)で `matlab_dump.m` を MMECG の `1.mat` のあるフォルダで実行 →
`python compare_with_matlab_mat.py wsst_check_trial1_p1.mat`. 行数 140・複素 SST の相対差 1e-9 程度なら一致.
参照実装(MathWorks のコードの写しを含む)はリポジトリに入れず, スクラッチに置いた.

## 関連して確かめたこと
- 窓ごとに 10 点を選ぶ手順(DTM, `DTM_info['point'][i]` = 例 [36, 31, 21, 48, 45, 39, 28, 32, 35, 16])と 10 チャネル入力の配布重み
  (`CFT-RFcardi/.../best_Super_100_mse.pt`, stage0 の入力 10)は後継研究 CFT-RFcardi のもの. radarODE-MTL の論文(TIM 2025)は
  "the 50 synchronous radar signals will be preprocessed into spectrograms by SST" で, 我々の 50 点入力と一致.
- ただし我々の段幅(cft = 32/128/256/512)はこの CFT-RFcardi の配布重みから取った. radarODE-MTL の公開コードは 128/256/512/1024.
- 窓ごとの正規化: 著者の sst_norm_01 は最初の軸ごとの min-max. 点を含む 3 次元配列に当てるので「点ごと」で, 我々と同じ.
