"""radarODE(SCEG → 長期再構成)を, 提案手法と同じ分割・4 s 窓・採点で学習・評価する. 1 分割ぶん.

段取り(論文は段ごとに別の評価をしており, 長期モジュールは SCEG の出力を入力に取るので, 順に学習する):
  1. sceg : SCEG を 1 拍 = 1 標本で学習(損失は公開実装の shapeLoss と同じ MSE. 正解の断片は [0, 1])
  2. refs : 学習した SCEG を固定し, 窓ごとに形態参照(断片を KDE の PPI に伸縮して並べた 800 点)を作ってキャッシュ
            PPI・位相は ppi.py のキャッシュ(設定は val で選んだもの)から. 学習側でも推定した PPI を使う(論文 p.8 "previous PPI estimation")
  3. long : 長期モジュールを (RCG, 形態参照) → ECG で学習(損失 RMSE, Fig. 2(c))
  4. eval : test の 0.2 s 刻みの全窓で 4 s の ECG を出し, ecg_rri.window_rr(NeuroKit2)で窓中央の RR を取る
最適化(p.9): SGD, 学習率 1e-3, コサイン, バッチ 32, 200 エポック, 早期終了. momentum 0.937・weight decay 5e-4 は論文に無いので
radarODE-MTL 公開実装(同じ著者)の値. 早期終了の待ちエポック数も論文に無いので 20. エポックは val で選ぶ
(SCEG は val の MSE, 長期は val の窓中央 RR の誤差 = 相手に有利な側).

  PYTHONPATH=src RADARODE_ROOT=../radarODE-MTL ../radarODE-MTL/.venv/bin/python -m rri_baselines.radarode_long.train all <分割>
  (動作確認: ... train smoke 0 --max-steps 20)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rri_baselines.ecg_rri import FALLBACK_CENTER, FALLBACK_MEAN, FALLBACK_MEDIAN, train_mean_rr, window_rr
from rri_baselines.protocol import FS, STEP, WINDOW_LEN, eval_windows, loso_folds, score, trials_of
from rri_baselines.radarode_long import ppi as kde
from rri_baselines.radarode_long.data import SCEGBeats, LongWindows, Trials, anchors, mechanical_delay, place
from rri_baselines.radarode_long.model import LongTerm, build_sceg, count_params
from rri_baselines.radarode_long.sst import N_FRAMES, WindowSST, crop_centered

REPO = Path(__file__).resolve().parents[3]
DEFAULT_KDE = "paper/lseg4/neurokit/scott"        # ppi_kde.json が無いときの設定(test 以外の 10 名で選ぶと 11 分割すべてでこれ)
# KDE の設定の選び方: by_ten(test 以外の 10 名, 既定)/ by_val(val の 1 名). val 1 名だと参加者 17 で 187 ms に崩れる(ppi_kde.json)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sceg", "refs", "long", "eval", "all", "smoke"])
    ap.add_argument("fold", type=int)
    ap.add_argument("--width", default="table1", choices=["table1", "cft"])
    ap.add_argument("--band", default="paper", choices=["paper", "author"], help="SCEG の入力 SST の帯域")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-stride", type=int, default=160, help="長期の学習窓の間隔[標本](0.8 s)")
    ap.add_argument("--causal-tcn", action="store_true")
    ap.add_argument("--tcn", default="plain", choices=["residual", "plain"], help="長期の融合 TCN(model.py)")
    ap.add_argument("--max-steps", type=int, default=0, help="動作確認用: 1 エポックの最大ステップ数")
    ap.add_argument("--limit-trials", type=int, default=0, help="動作確認用: 使う試行の数を各分けで絞る")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RADARODE_WORKERS", "4")))
    return ap.parse_args()


class Run:
    def __init__(self, a: argparse.Namespace):
        self.a, self.dev = a, torch.device(a.device)
        self.fold = loso_folds()[a.fold]
        lim = (lambda ids: ids[:a.limit_trials]) if a.limit_trials else (lambda ids: ids)
        self.train_ids, self.val_ids, self.test_ids = (lim(trials_of(self.fold.train)), lim(trials_of(self.fold.val)),
                                                       lim(trials_of(self.fold.test)))
        variant = f"{a.width}_{a.band}" + ("_causal" if a.causal_tcn else "") + ("_residual" if a.tcn == "residual" else "") + ("_smoke" if a.stage == "smoke" else "")
        self.dir = REPO / "runs/radarode_long" / variant / f"fold{self.fold.index:02d}_test{self.fold.test[0]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sst = WindowSST(a.band, self.dev)
        self.fallback = train_mean_rr(trials_of(self.fold.train))
        self.kde_cfg = self._kde_config()
        self.log(f"分割 {self.fold.index}: test {self.fold.test} / val {self.fold.val} / 学習 {len(self.train_ids)} 試行. "
                 f"KDE の設定 {self.kde_cfg}, 平均 RR(代わりの値){self.fallback:.1f} ms")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with (self.dir / "log.txt").open("a") as f:
            f.write(line + "\n")

    def _kde_config(self) -> str:
        res = REPO / "results/radarode_long/ppi_kde.json"
        if res.exists():
            return json.loads(res.read_text())["chosen"][self.fold.index][os.environ.get("RADARODE_KDE_SELECT", "by_ten")]
        return DEFAULT_KDE

    # ------------------------------------------------------------ PPI と位相 ------------------------------------------------------------
    def kde_lookup(self, t: int, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = kde.parse_label(self.kde_cfg)
        path = kde.CACHE / f"trial{t}.npz"
        if path.exists():
            z = kde.load_cache(t)
            idx = (starts // STEP).astype(int)
            return z["ppi_ms"][(idx,) + cfg], z["phase"][(idx,) + cfg]
        # キャッシュがまだ無い試行(動作確認)はその場で計算する
        band = WindowSST(kde.BANDS[cfg[0]], self.dev)
        rcg = Trials([t])[t][0]
        ppi_ms, phase = np.full(len(starts), np.nan), np.full(len(starts), np.nan)
        for i in range(0, len(starts), 8):
            x = torch.stack([torch.from_numpy(rcg[:, s:s + WINDOW_LEN]) for s in starts[i:i + 8]])
            en = band.energy(x).cpu().numpy()
            for j, e in enumerate(en):
                p, ph, _ = kde.window_estimates(e)
                ppi_ms[i + j], phase[i + j] = p[cfg[1:]], ph[cfg[1:]]
        return ppi_ms, phase

    def delay(self) -> float:
        """δ: エネルギー波形の位相と R 波のずれ[標本]の中央値(学習側の試行, 0.8 s 刻みの窓)."""
        path = self.dir / "delay.json"
        if path.exists():
            return json.loads(path.read_text())["delta_samples"]
        tr = Trials(self.train_ids)
        d = []
        for t in self.train_ids:
            rcg, _, peaks = tr[t]
            starts = np.arange(0, rcg.shape[1] - WINDOW_LEN + 1, 160)
            p, ph = self.kde_lookup(t, starts)
            d.append(mechanical_delay(ph, p, starts, peaks))
        d = np.concatenate(d)
        delta = float(np.median(d))
        path.write_text(json.dumps({"delta_samples": delta, "delta_ms": delta / FS * 1000, "n": int(len(d)),
                                    "iqr_ms": [float(np.percentile(d, 25)) / FS * 1000, float(np.percentile(d, 75)) / FS * 1000]}))
        self.log(f"δ(位相 − R 波)= {delta / FS * 1000:.1f} ms(窓 {len(d)} 個の中央値)")
        return delta

    # ------------------------------------------------------------ 学習の共通部 ------------------------------------------------------------
    def fit(self, name: str, model: torch.nn.Module, step, validate, loader) -> None:
        a = self.a
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.937, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.01)
        ckpt, best_path = self.dir / f"{name}_last.pt", self.dir / f"{name}_best.pt"
        start, best, best_ep, hist = 0, float("inf"), -1, []
        if ckpt.exists():                                          # 途中から続ける
            st = torch.load(ckpt, map_location=self.dev, weights_only=False)
            model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
            start, best, best_ep, hist = st["epoch"] + 1, st["best"], st["best_ep"], st["hist"]
        for ep in range(start, a.epochs):
            if ep - best_ep > a.patience and best_ep >= 0:
                self.log(f"{name}: 早期終了(最良はエポック {best_ep})")
                break
            model.train()
            t0, losses = time.time(), []
            for k, batch in enumerate(loader):
                if a.max_steps and k >= a.max_steps:
                    break
                loss = step(model, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()
            t_train = time.time() - t0
            model.eval()
            v = validate(model)
            hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val": v, "sec_train": t_train,
                         "sec_val": time.time() - t0 - t_train, "steps": len(losses)})
            if v < best:
                best, best_ep = v, ep
                torch.save(model.state_dict(), best_path)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                        "best": best, "best_ep": best_ep, "hist": hist}, ckpt)
            self.log(f"{name} エポック {ep}: 学習 {np.mean(losses):.4f}({len(losses)} 歩, {t_train:.0f} s, "
                     f"{1000 * t_train / max(len(losses), 1):.0f} ms/歩)/ val {v:.4f}")
        (self.dir / f"{name}_history.json").write_text(json.dumps(hist, indent=1))

    # ------------------------------------------------------------ 1. SCEG ------------------------------------------------------------
    def sceg_model(self) -> torch.nn.Module:
        if self.dev.type == "cuda" and os.environ.get("RADARODE_FAST_OPS", "1") == "1":
            m = build_sceg(self.a.width)               # radarODE-MTL を sys.path に足すのもここ
            from repro.fast_ops import apply           # 変形畳み込みの基準座標を使い回すだけ(数値は変わらない)
            apply()
            return m.to(self.dev)
        return build_sceg(self.a.width).to(self.dev)

    def sceg_input(self, x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        img, _ = self.sst(x.to(self.dev), chunk=8)
        return crop_centered(img, center.to(self.dev))

    def train_sceg(self) -> None:
        a = self.a
        tr, va = SCEGBeats(Trials(self.train_ids), True), SCEGBeats(Trials(self.val_ids), False)
        model = self.sceg_model()
        self.log(f"SCEG: 学習 {len(tr)} 拍 / val {len(va)} 拍, {count_params(model):,} パラメータ")
        loader = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                            generator=torch.Generator().manual_seed(42))
        vloader = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

        def step(m, batch):
            x, c, y = batch
            return F.mse_loss(m(self.sceg_input(x, c)), y.to(self.dev))

        @torch.no_grad()
        def validate(m):
            errs = []
            for k, (x, c, y) in enumerate(vloader):
                if a.max_steps and k >= a.max_steps:
                    break
                errs.append(F.mse_loss(m(self.sceg_input(x, c)), y.to(self.dev), reduction="sum").item() / y.numel())
            return float(np.mean(errs))

        torch.manual_seed(42)
        self.fit("sceg", model, step, validate, loader)

    # ------------------------------------------------------------ 2. 形態参照 ------------------------------------------------------------
    @torch.no_grad()
    def build_refs(self, sceg, trial_ids: list[int], stride: int, tag: str) -> dict:
        out = {}
        trials = Trials(trial_ids)
        delta = self.delay()
        for t in trial_ids:
            path = self.dir / "refs" / f"{tag}_trial{t}.npz"
            if path.exists():
                z = np.load(path)
                out[t] = (z["starts"], z["ref"])
                continue
            rcg = trials[t][0]
            starts = np.arange(0, rcg.shape[1] - WINDOW_LEN + 1, stride)
            if self.a.max_steps:
                starts = starts[:self.a.max_steps * 4]
            ppi_ms, phase = self.kde_lookup(t, starts)
            ppi_ms = np.where(np.isfinite(ppi_ms), ppi_ms, self.fallback)
            refs = np.zeros((len(starts), WINDOW_LEN), dtype=np.float16)
            for i in range(0, len(starts), 16):
                sl = slice(i, i + 16)
                x = torch.stack([torch.from_numpy(rcg[:, s:s + WINDOW_LEN]) for s in starts[sl]])
                img, _ = self.sst(x.to(self.dev), chunk=8)
                r_hat = [anchors(p, ph, delta) for p, ph in zip(ppi_ms[sl], phase[sl])]
                w_idx = torch.tensor([w for w, r in enumerate(r_hat) for _ in r], dtype=torch.long, device=self.dev)
                if len(w_idx) == 0:
                    continue
                centers = torch.tensor(np.round(np.concatenate(r_hat) * N_FRAMES / WINDOW_LEN), device=self.dev)
                crops = crop_centered(img[w_idx], centers)
                pieces = torch.cat([sceg(c)[:, 0] for c in crops.split(32)])
                k = 0
                for w, r in enumerate(r_hat):
                    refs[i + w] = place(pieces[k:k + len(r)], r, ppi_ms[i + w]).cpu().numpy()
                    k += len(r)
            path.parent.mkdir(exist_ok=True)
            np.savez(path, starts=starts, ref=refs, ppi_ms=ppi_ms, phase=phase)
            out[t] = (starts, refs)
        return out

    def load_sceg(self):
        m = self.sceg_model()
        m.load_state_dict(torch.load(self.dir / "sceg_best.pt", map_location=self.dev))
        return m.eval()

    def refs(self) -> None:
        sceg = self.load_sceg()
        t0 = time.time()
        for ids, stride, tag in ((self.train_ids, self.a.train_stride, "train"), (self.val_ids, 160, "val"),
                                 (self.test_ids, STEP, "test")):
            r = self.build_refs(sceg, ids, stride, tag)
            self.log(f"形態参照 {tag}: {sum(len(v[0]) for v in r.values())} 窓({time.time() - t0:.0f} s まで)")

    # ------------------------------------------------------------ 3. 長期 ------------------------------------------------------------
    def window_errors(self, model, refs: dict, trials: Trials) -> tuple[dict, dict]:
        """窓ごとの 4 s の ECG → window_rr. 戻り値 {試行: 予測[ms]}, {試行: 規則の内訳}."""
        preds, rules = {}, {}
        for t, (starts, ref) in refs.items():
            rcg = trials[t][0]
            outs = []
            with torch.no_grad():
                for i in range(0, len(starts), 64):
                    x = torch.stack([torch.from_numpy(rcg[:, s:s + WINDOW_LEN]) for s in starts[i:i + 64]]).to(self.dev)
                    r = torch.from_numpy(ref[i:i + 64].astype(np.float32))[:, None].to(self.dev)
                    outs.append(model(x, r)[:, 0].cpu().numpy())
            ecg = np.concatenate(outs)
            got = [window_rr(e, self.fallback) for e in ecg]
            preds[t], rules[t] = np.array([v for v, _ in got]), np.array([c for _, c in got])
        return preds, rules

    def train_long(self) -> None:
        a = self.a
        sceg = self.load_sceg()
        tr_trials, va_trials = Trials(self.train_ids), Trials(self.val_ids)
        tr = LongWindows(tr_trials, self.build_refs(sceg, self.train_ids, a.train_stride, "train"))
        va_refs = self.build_refs(sceg, self.val_ids, 160, "val")
        del sceg
        torch.cuda.empty_cache() if self.dev.type == "cuda" else None
        model = LongTerm(causal_tcn=a.causal_tcn, tcn=a.tcn).to(self.dev)
        self.log(f"長期: 学習 {len(tr)} 窓 / val {sum(len(v[0]) for v in va_refs.values())} 窓, {count_params(model):,} パラメータ")
        loader = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                            generator=torch.Generator().manual_seed(42))
        truth = {}
        for t, (starts, _) in va_refs.items():
            rcg, _, peaks = va_trials[t]
            all_starts, true_ms = eval_windows(rcg.shape[1], peaks)
            truth[t] = true_ms[(starts // STEP).astype(int)]

        def step(m, batch):
            x, r, y = (b.to(self.dev) for b in batch)
            return torch.sqrt(F.mse_loss(m(x, r), y))

        def validate(m):                 # val の窓中央 RR の誤差(試行平均 → 参加者 1 名)
            preds, _ = self.window_errors(m, va_refs, va_trials)
            return float(np.mean([np.nanmean(np.where(np.isfinite(truth[t]), np.abs(preds[t] - truth[t]), np.nan)) for t in preds]))

        torch.manual_seed(42)
        self.fit("long", model, step, validate, loader)

    # ------------------------------------------------------------ 4. 評価 ------------------------------------------------------------
    def evaluate(self) -> dict:
        sceg = self.load_sceg()
        refs = self.build_refs(sceg, self.test_ids, STEP, "test")
        del sceg
        model = LongTerm(causal_tcn=self.a.causal_tcn, tcn=self.a.tcn).to(self.dev)
        model.load_state_dict(torch.load(self.dir / "long_best.pt", map_location=self.dev))
        model.eval()
        trials = Trials(self.test_ids)
        preds, rules = self.window_errors(model, refs, trials)
        ref_rr = {t: np.array([window_rr(r.astype(np.float32), self.fallback)[0] for r in ref]) for t, (_, ref) in refs.items()}
        out = {}
        for t in self.test_ids:
            out[f"trial{t}_pred_ms"] = preds[t]
            out[f"trial{t}_ref_ms"] = ref_rr[t]
            out[f"trial{t}_rule"] = rules[t]
        np.savez(self.dir / "predictions.npz", **out)
        # 近道の確認(shortcut_check.py): val の窓でレーダ・参照を他の窓と入れ替えたときの RMSE
        from rri_baselines.radarode_long.data import minmax01
        from rri_baselines.radarode_long.shortcut_check import swap_losses
        va_trials = Trials(self.val_ids)
        xs, rs, ys = [], [], []
        for t, (starts, ref) in self.build_refs(self.load_sceg(), self.val_ids, 160, "val").items():
            rcg, ecg, _ = va_trials[t]
            xs += [rcg[:, s:s + WINDOW_LEN] for s in starts]
            rs += [r[None].astype(np.float32) for r in ref]
            ys += [minmax01(ecg[s:s + WINDOW_LEN])[None] for s in starts]
        to_t = lambda v: torch.from_numpy(np.stack(v).astype(np.float32)).to(self.dev)  # noqa: E731
        swap = swap_losses(model, to_t(xs), to_t(rs), to_t(ys))
        allr = np.concatenate(list(rules.values()))
        res = {"rule_share": {n: float(np.mean(allr == c)) for n, c in
                              (("center", FALLBACK_CENTER), ("median", FALLBACK_MEDIAN), ("train_mean", FALLBACK_MEAN))}, "swap_val_rmse": swap}
        if not self.a.limit_trials and not self.a.max_steps:
            res["mae_ms"] = score(preds)["mae_ms"]
            res["mae_ms_reference_only"] = score(ref_rr)["mae_ms"]
        (self.dir / "evaluation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        self.log(f"評価: {res}")
        return res


def main() -> None:
    a = parse()
    if a.stage == "smoke":                     # 動作確認: 各段を数十歩だけ
        a.epochs, a.patience = 2, 5
        a.max_steps = a.max_steps or 20
        a.limit_trials = a.limit_trials or 2
    run = Run(a)
    stages = {"sceg": ["sceg"], "refs": ["refs"], "long": ["long"], "eval": ["eval"],
              "all": ["sceg", "refs", "long", "eval"], "smoke": ["sceg", "long", "eval"]}[a.stage]
    for s in stages:
        t0 = time.time()
        {"sceg": run.train_sceg, "refs": run.refs, "long": run.train_long, "eval": run.evaluate}[s]()
        run.log(f"段 {s}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
