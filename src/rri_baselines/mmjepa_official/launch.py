"""公式スクリプトを, ファイルを書き換えずに実行する起動器.

    python -m rri_baselines.mmjepa_official.launch <mmJEPA.py|end2end_train.py|inference_from_train.py> [公式の引数...]

公式スクリプトのソースをそのまま読み, __main__ として実行する. 引数では変えられない「実行環境の固定値」だけを,
計算の中身に触れない形で合わせる(何をしたかは MMJEPA_PATCH_LOG の JSON に残す):

  1. DataLoader の num_workers: 3 本とも DataLoader(num_workers=8, persistent_workers=True) が直書きで,
     引数では下げられない(mmJEPA.py の -num_workers 引数は読まれるが dl_kwargs に渡っていない). 手元の RAM 16 GB を守るため
     MMJEPA_MAX_WORKERS(既定 1, 最小 1)で上限を掛ける. 0 にはしない: persistent_workers=True のままなら反復子は 1 回だけ
     作られ, 親プロセスの乱数(既定の生成器)の消費が 8 本のときと同じになる(0 本だと persistent が切れてエポックごとに
     base_seed を 1 つ引き, 以後のシャッフル・拡散の t・ノイズの乱数列がずれる). 窓の取り出しは添字だけの決定的な処理なので,
     ワーカー数 1 と 8 で計算結果は同じ
  2. end2end_train.py の device: torch.device('cuda:6' if ...) が直書き(GPU 1 枚の機械では存在しない番号).
     この 1 か所の文字列 'cuda:6' だけを MMJEPA_DEVICE(既定 cuda:0)に置き換えて実行する(ファイルは変えない).
     mmJEPA.py は -device, inference_from_train.py は --gpu-id の引数で渡す
  3. random / numpy の乱数種: mmJEPA.py は torch.manual_seed(42) だけで, マスク(block/block_mask.py)は
     種を与えない Python の random を使う. 再現のため実行前に random.seed(42)・np.random.seed(42) を置く
  4. CPU で動かすときだけ: mmJEPA.py の timer() と end2end_train.py のログが torch.cuda.synchronize() を無条件に呼ぶので
     CUDA が無いときは何もしない関数にする(GPU では何もしない)
"""
import json
import os
import random
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils.data as tud  # noqa: E402

from rri_baselines.mmjepa_official.common import OFFICIAL, SEED, official_on_path  # noqa: E402


def main() -> None:
    script, args = sys.argv[1], sys.argv[2:]
    path = OFFICIAL / script
    src = path.read_text(encoding="utf-8")
    max_workers = max(1, int(os.environ.get("MMJEPA_MAX_WORKERS", "1")))
    device = os.environ.get("MMJEPA_DEVICE", "cuda:0")
    patches = {"script": str(path), "args": args, "max_workers": max_workers, "device": device,
               "torchdynamo_disable": os.environ.get("TORCHDYNAMO_DISABLE"), "applied": []}

    # 1. DataLoader の作業プロセス数に上限
    base = tud.DataLoader

    class CappedDataLoader(base):
        def __init__(self, *a, **k):
            asked = k.get("num_workers", 0)
            if asked > max_workers:
                k["num_workers"] = max_workers
            super().__init__(*a, **k)

    tud.DataLoader = CappedDataLoader
    patches["applied"].append(f"DataLoader num_workers <= {max_workers}")

    # 2. end2end_train.py の直書きの GPU 番号
    if script == "end2end_train.py":
        old = "torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')"
        assert src.count(old) == 1, "end2end_train.py の device 行が想定と違う"
        src = src.replace(old, f"torch.device({device!r} if torch.cuda.is_available() else 'cpu')")
        patches["applied"].append(f"end2end_train.py: 'cuda:6' -> {device!r}")

    # 3. 乱数種
    random.seed(SEED)
    np.random.seed(SEED)
    patches["applied"].append(f"random.seed({SEED}), np.random.seed({SEED})")

    # 4. CPU だけ
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
        patches["applied"].append("CPU: torch.cuda.synchronize -> no-op")

    log = os.environ.get("MMJEPA_PATCH_LOG")
    if log:
        with open(log, "w", encoding="utf-8") as f:
            json.dump(patches, f, indent=1, ensure_ascii=False)

    official_on_path()
    sys.argv = [str(path), *args]
    glb = {"__name__": "__main__", "__file__": str(path), "__builtins__": __builtins__}
    exec(compile(src, str(path), "exec"), glb)


if __name__ == "__main__":
    main()
