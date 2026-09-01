"""処理エンジンの登録と切り替え

各処理段(分離 / 拍 / 採譜 / ドラム / コード)に複数のバックエンドを用意し、
config.json で切り替えられるようにする。

tier の意味:
  cpu      … CPUだけで実用的に動く。既定で使う
  cpu-slow … CPUでも動くが目に見えて遅い
  gpu      … NVIDIA GPU がないと現実的でない
"""
import json
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "separator": "auto",
    "beat": "auto",
    "pitch": "auto",
    "drums": "auto",
    "chords": "builtin",
    "device": "auto",
    # 採譜するパート。減らすとその分だけ速くなる
    "parts": ["melody", "guitar", "bass", "piano", "drums"],
    # 保存する音源の形式と、どのステムを残すか
    "audio_format": "opus",   # opus (小さい) / mp3 (互換性重視)
    "keep_stems": "used",     # used(既定) / all / none
    # 外部採譜スクリプト用のコマンドテンプレート (pitch=external のとき使う)
    # {input} と {output} が実際のパスに置き換わる。出力はMIDIファイル。
    "pitch_external_cmd": "",
}


# ------------------------------------------------------------------ 判定
def _has(mod):
    try:
        return find_spec(mod) is not None
    except Exception:
        return False


_import_cache = {}


def _importable(mod):
    """実際に import してみる。インストール済みでも壊れている場合を弾くため。

    (basic-pitch は TensorFlow を連れてくることがあり、numpy の版が噛み合わないと
     パッケージは存在するのに import で落ちる。find_spec だけでは検出できない)
    結果はキャッシュする — 読み込みに数秒かかるものがあるため。
    """
    if mod not in _import_cache:
        try:
            __import__(mod)
            _import_cache[mod] = True
        except Exception as e:
            print(f"[tabmaker] {mod} を読み込めません: {type(e).__name__}: "
                  f"{str(e)[:160]}", flush=True)
            _import_cache[mod] = False
    return _import_cache[mod]


def has_cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device(cfg=None):
    """使用するデバイスを決める"""
    want = (cfg or load_config()).get("device", "auto")
    if want == "cpu":
        return "cpu"
    if want == "cuda":
        return "cuda" if has_cuda() else "cpu"
    return "cuda" if has_cuda() else "cpu"


# ------------------------------------------------------------------ 登録
def _cmd_exists(name):
    return shutil.which(name) is not None


ENGINES = {
    "separator": [
        {
            "id": "htdemucs_6s", "label": "Demucs 6stem (ギター/ピアノを分離)",
            "tier": "cpu", "recommended": True,
            "desc": "ギターとピアノを専用ステムとして取り出す。ギター譜の精度が上がる。"
                    "ピアノステムの品質は本家でも発展途上",
            "check": lambda: _has("demucs"),
            "hint": "setup.bat で入ります",
        },
        {
            "id": "htdemucs", "label": "Demucs 4stem (最速)",
            "tier": "cpu",
            "desc": "vocals/drums/bass/other の4つ。ギターとピアノは other に混ざる",
            "check": lambda: _has("demucs"),
            "hint": "setup.bat で入ります",
        },
        {
            "id": "htdemucs_ft", "label": "Demucs 4stem 微調整版 (高品質・4倍遅い)",
            "tier": "cpu-slow",
            "desc": "4つのモデルを順に走らせる。分離は良くなるがギター/ピアノは分かれない",
            "check": lambda: _has("demucs"),
            "hint": "setup.bat で入ります",
        },
        {
            "id": "roformer_hybrid", "label": "RoFormer + Demucs 6stem (GPU推奨)",
            "tier": "gpu",
            "desc": "ボーカルを BS-RoFormer で抜いてから、残りを Demucs 6stem に通す。"
                    "分離品質は最高だがCPUだと数倍の時間がかかる",
            "check": lambda: _has("audio_separator") and _has("demucs"),
            "hint": "setup-gpu.bat で入ります",
        },
    ],
    "beat": [
        {
            "id": "beat_this", "label": "Beat This! (ISMIR 2024)",
            "tier": "cpu", "recommended": True,
            "desc": "拍と小節頭(ダウンビート)を同時に推定するTransformer。"
                    "拍子も自動判定できるようになる",
            "check": lambda: _importable("beat_this.inference"),
            "hint": "setup.bat で入ります (モデル78MB)",
        },
        {
            "id": "librosa", "label": "librosa (内蔵・追加DL不要)",
            "tier": "cpu",
            "desc": "自作のヒューリスティックで小節頭を推定する。4/4固定",
            "check": lambda: True,
        },
    ],
    "pitch": [
        {
            "id": "basic_pitch", "label": "basic-pitch (Spotify)",
            "tier": "cpu", "recommended": True,
            "desc": "和音も拾える音符推定AI",
            "check": lambda: _importable("basic_pitch.inference"),
            "hint": "setup.bat で入ります (check.bat で読み込みエラーを確認できます)",
        },
        {
            "id": "librosa", "label": "librosa pyin (軽量)",
            "tier": "cpu",
            "desc": "単音のみ。追加DLなしで動くが和音は取れない",
            "check": lambda: True,
        },
        {
            "id": "external", "label": "外部スクリプト (YourMT3+ など)",
            "tier": "gpu",
            "desc": "config.json の pitch_external_cmd に設定したコマンドを呼び、"
                    "出力されたMIDIを読み込む。YourMT3+ 等を繋ぐための拡張点",
            "check": lambda: bool(load_config().get("pitch_external_cmd")),
            "hint": "config.json の pitch_external_cmd を設定してください",
        },
    ],
    "drums": [
        {
            "id": "adtof", "label": "ADTOF (学習済みAI)",
            "tier": "cpu", "recommended": True,
            "desc": "リズムゲーム譜面359時間で学習したドラム採譜モデル。"
                    "キック/スネア/ハイハット/タム/シンバルの5クラス。非商用ライセンス",
            "check": lambda: _importable("adtof_pytorch"),
            "hint": "setup.bat で入ります (Gitが必要)",
        },
        {
            "id": "builtin", "label": "内蔵ヒューリスティック (追加DL不要)",
            "tier": "cpu",
            "desc": "オンセットの帯域パワーで打楽器を見分ける。タムとスネアを間違えやすい",
            "check": lambda: True,
        },
    ],
    "chords": [
        {
            "id": "builtin", "label": "コード認識あり (内蔵)",
            "tier": "cpu", "recommended": True,
            "desc": "クロマ特徴とビタビ探索でコード進行を推定する。追加DLなし",
            "check": lambda: True,
        },
        {"id": "off", "label": "コード認識なし", "tier": "cpu",
         "desc": "処理時間を少し短縮できる", "check": lambda: True},
    ],
}


def _entry(stage, engine_id):
    for e in ENGINES[stage]:
        if e["id"] == engine_id:
            return e
    return None


def available(stage, engine_id):
    e = _entry(stage, engine_id)
    if not e:
        return False
    try:
        return bool(e["check"]())
    except Exception:
        return False


def resolve(stage, cfg=None):
    """設定値から実際に使うエンジンIDを決める。auto なら使える中で一番上のもの"""
    cfg = cfg or load_config()
    want = cfg.get(stage, "auto")
    if want != "auto" and available(stage, want):
        return want
    order = [e for e in ENGINES[stage] if e.get("recommended")] + \
            [e for e in ENGINES[stage] if not e.get("recommended")]
    for e in order:
        if e["tier"] == "gpu" and not has_cuda() and want == "auto":
            continue  # auto ではGPU前提のものを勝手に選ばない
        try:
            if e["check"]():
                return e["id"]
        except Exception:
            pass
    # どれも使えないときは、GPU前提でないものを既定にする
    for e in ENGINES[stage]:
        if e["tier"] != "gpu":
            return e["id"]
    return ENGINES[stage][0]["id"]


# ------------------------------------------------------------------ 設定
def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_PATH.exists():
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg


def save_config(patch):
    cfg = load_config()
    for k, v in (patch or {}).items():
        if k in DEFAULT_CONFIG:
            cfg[k] = v
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return cfg


ALL_PARTS = [("melody", "メロディ"), ("guitar", "ギター"), ("bass", "ベース"),
             ("piano", "ピアノ"), ("drums", "ドラム")]


def wanted_parts(cfg=None):
    """採譜するパート。不正な値が入っていても壊れないようにする"""
    cfg = cfg or load_config()
    want = cfg.get("parts")
    valid = [k for k, _ in ALL_PARTS]
    if not isinstance(want, list):
        return valid
    got = [p for p in valid if p in want]
    return got or valid


def describe():
    """UIに渡す一覧"""
    cfg = load_config()
    out = {"config": cfg, "cuda": has_cuda(), "device": device(cfg), "stages": {},
           "all_parts": [{"id": k, "label": v} for k, v in ALL_PARTS],
           "parts": wanted_parts(cfg)}
    labels = {"separator": "パート分離", "beat": "拍・小節の推定",
              "pitch": "音符の推定", "drums": "ドラムの採譜", "chords": "コード進行"}
    for stage, entries in ENGINES.items():
        out["stages"][stage] = {
            "label": labels.get(stage, stage),
            "active": resolve(stage, cfg),
            "options": [{
                "id": e["id"], "label": e["label"], "tier": e["tier"],
                "desc": e.get("desc", ""), "hint": e.get("hint", ""),
                "recommended": bool(e.get("recommended")),
                "available": available(stage, e["id"]),
            } for e in entries],
        }
    return out


def summary_line():
    cfg = load_config()
    parts = [f"{s}={resolve(s, cfg)}" for s in ENGINES]
    return f"device={device(cfg)}  " + "  ".join(parts)


def pip_install(args, timeout=1800):
    """任意ライブラリの追加インストール (setupスクリプトから使う)"""
    cmd = [sys.executable, "-m", "pip", "install", *args]
    return subprocess.run(cmd, timeout=timeout).returncode == 0
