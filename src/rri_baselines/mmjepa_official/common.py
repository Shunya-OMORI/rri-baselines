"""公式コードの場所・実行環境・再現性の記録(manifest).

公式リポジトリ external/mmJEPA-ECG は一切書き換えない. __pycache__ が追跡されているので, 公式モジュールを読む
プロセスはすべて PYTHONDONTWRITEBYTECODE=1(sys.dont_write_bytecode)で動かし, 実行の前後に git status が空かを確かめる.
"""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[3]
OFFICIAL = Path(os.environ.get("MMJEPA_OFFICIAL", "/home/shunya/research/external/mmJEPA-ECG"))
OFFICIAL_COMMIT = "4423ef66e0fb9f1a469f078eee8c0c335d12e94e"
RUNS = Path(os.environ.get("MMJEPA_OFF_RUNS", REPO / "runs/mmjepa_official"))
CACHE = Path(os.environ.get("MMJEPA_OFF_CACHE", REPO / "cache/mmjepa_official"))
SEED = 42
SCRIPTS = ("mmJEPA.py", "end2end_train.py", "inference_from_train.py")


def official_on_path() -> None:
    """公式モジュール(block.*, custom_model, diffusion)を import できるようにする(バイトコードは書かない)."""
    sys.dont_write_bytecode = True
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))


def official_state() -> dict:
    """公式リポジトリのコミットと, 作業ツリーが無改変か(git status が空か)."""
    def git(*a):
        return subprocess.run(["git", "-C", str(OFFICIAL), *a], capture_output=True, text=True).stdout.strip()

    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"path": str(OFFICIAL), "commit": head, "expected_commit": OFFICIAL_COMMIT,
            "clean": status == "", "status": status,
            "sha256": {s: hashlib.sha256((OFFICIAL / s).read_bytes()).hexdigest() for s in SCRIPTS}}


def assert_official_clean() -> dict:
    st = official_state()
    if st["commit"] != OFFICIAL_COMMIT or not st["clean"]:
        raise RuntimeError(f"公式リポジトリが想定と違う(コミット {st['commit']}, 変更 {st['status']!r})")
    return st


def pip_freeze() -> list[str]:
    """venv に入っている配布物と版(pip freeze 相当. venv に pip が無くても取れる)."""
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())


def run_env(device: str) -> dict:
    """公式スクリプトを動かすサブプロセスの環境変数.

    - TORCHDYNAMO_DISABLE=1: 公式は torch.compile を無条件にかける. 研究室の規約(比べる学習は compile なし)に合わせ,
      かつ compile したままだと mmJEPA.py の保存した重み(接頭辞なし)を end2end_train.py が読めない(下の注)ので切る.
      MMJEPA_COMPILE=1 で公式どおり compile を残す(そのときは run.py が重みの名前に _orig_mod. を付けた写しを渡す)
    - CUDA_VISIBLE_DEVICES: device が cpu のときは空にして GPU に触れない
    """
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled", WANDB_SILENT="true", MPLBACKEND="Agg",
               MMJEPA_DEVICE=device,
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]))
    if os.environ.get("MMJEPA_COMPILE") != "1":
        env["TORCHDYNAMO_DISABLE"] = "1"
    else:
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")       # compile の作業プロセスを増やさない(メモリ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def gpu_name() -> str:
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        q = ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"]
        names = subprocess.run(q, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return f"CUDA_VISIBLE_DEVICES={vis!r}: " + "; ".join(names)
    except Exception as e:      # noqa: BLE001
        return f"不明 ({e})"


def write_manifest(out: Path, extra: dict | None = None) -> dict:
    """分割の出力ディレクトリに, 公式コミット・依存の版・実行環境を残す(段ごとのコマンドは stages/ に追記)."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
           "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda_build": torch.version.cuda, "gpu": gpu_name(), "seed": SEED,
           "official": official_state(),
           "env": {k: v for k, v in os.environ.items() if k.startswith(("MMJEPA", "CUDA_VISIBLE", "TORCHDYNAMO",
                                                                        "OMP_NUM", "RRI_DATA"))},
           **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    (out / "pip_freeze.txt").write_text("\n".join(pip_freeze()) + "\n")
    return man


def run_stage(name: str, cmd: list[str], cwd: Path, log: Path, device: str, record_dir: Path) -> float:
    """公式スクリプト(launch 経由)を 1 段走らせ, コマンド・時間・戻り値を record_dir/stages/<name>.json に残す."""
    assert_official_clean()
    record_dir.joinpath("stages").mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "cmd": cmd, "cwd": str(cwd), "device": device, "log": str(log),
           "start": time.strftime("%Y-%m-%d %H:%M:%S"),
           "patch_log": str(log.with_suffix(".patches.json"))}
    env = run_env(device)
    env["MMJEPA_PATCH_LOG"] = rec["patch_log"]
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n==== {rec['start']} {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    rec.update(end=time.strftime("%Y-%m-%d %H:%M:%S"), sec=time.time() - t0, returncode=p.returncode,
               official_after=official_state()["clean"])
    (record_dir / "stages" / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    if p.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"{name} が失敗(戻り値 {p.returncode}). ログ {log} の末尾:\n" + "\n".join(tail))
    if not rec["official_after"]:
        raise RuntimeError("公式リポジトリの作業ツリーが汚れた")
    return rec["sec"]
