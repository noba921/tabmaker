"""パート分離のバックエンド

どのエンジンを使っても、最終的に {ステム名: wavのパス} を返すところまで揃える。
返るステム名は vocals / drums / bass / other と、モデルによって guitar / piano。
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEMUCS_MODELS = {"htdemucs", "htdemucs_6s", "htdemucs_ft", "hdemucs_mmi"}
STEM_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano")

# BS-RoFormer: ボーカル分離のSDRが最も高い部類のモデル
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def _run_demucs(src, out_dir, model, device, on_progress, extra=()):
    cmd = [sys.executable, "-m", "demucs", "-n", model,
           "-o", str(out_dir), "-d", device, *extra, str(src)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env=env)
    tail, logbuf = b"", []
    while True:
        chunk = p.stdout.read(256)
        if not chunk:
            break
        tail = (tail + chunk)[-4096:]
        text = tail.decode("utf-8", errors="replace")
        logbuf.append(chunk.decode("utf-8", errors="replace"))
        m = re.findall(r"(\d{1,3}(?:\.\d)?)%", text)
        if m:
            try:
                on_progress(min(100.0, float(m[-1])))
            except ValueError:
                pass
        if "Downloading" in text:
            on_progress(None, "分離AIモデルをダウンロード中 (初回のみ)")
    p.wait()
    if p.returncode != 0:
        raise RuntimeError("パート分離に失敗:\n" + "".join(logbuf)[-800:])


def _collect(sep_dir):
    found = {}
    for name in STEM_NAMES:
        hit = next(sep_dir.rglob(f"{name}.wav"), None)
        if hit:
            found[name] = hit
    return found


def _separate_demucs(src, work_dir, model, device, on_progress):
    sep_dir = work_dir / "demucs"
    extra = ()
    if model == "htdemucs_ft":
        on_progress(None, "微調整版は4つのモデルを順に実行します (時間がかかります)")
    _run_demucs(src, sep_dir, model, device, on_progress, extra)
    stems = _collect(sep_dir)
    missing = {"vocals", "drums", "bass", "other"} - set(stems)
    if missing:
        raise RuntimeError(f"分離結果が足りません: {sorted(missing)}")
    return stems


def _separate_roformer_hybrid(src, work_dir, device, on_progress):
    """BS-RoFormer でボーカルを抜き、残りを Demucs 6stem に通す

    ボーカル分離は RoFormer 系が明確に上、楽器の細分化は Demucs 6stem という
    それぞれの得意分野を組み合わせる。
    """
    from audio_separator.separator import Separator

    on_progress(None, "RoFormer でボーカルを分離中 (初回はモデルDLあり)")
    ro_dir = work_dir / "roformer"
    ro_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(ro_dir), output_format="WAV",
                    log_level=40, use_autocast=(device == "cuda"))
    sep.load_model(model_filename=ROFORMER_MODEL)
    sep.separate(str(src), {"Vocals": "vocals", "Instrumental": "instrumental"})

    vocals = ro_dir / "vocals.wav"
    inst = ro_dir / "instrumental.wav"
    if not vocals.exists() or not inst.exists():
        # 出力名の指定が効かなかった場合は中身から拾う
        for f in ro_dir.glob("*.wav"):
            low = f.name.lower()
            if "vocal" in low and not vocals.exists():
                vocals = f
            elif "instrumental" in low and not inst.exists():
                inst = f
    if not vocals.exists() or not inst.exists():
        raise RuntimeError("RoFormer の出力が見つかりませんでした")

    on_progress(0, "Demucs 6stem で楽器を分けています")
    stems = _separate_demucs(inst, work_dir, "htdemucs_6s", device, on_progress)
    stems["vocals"] = vocals  # ボーカルは RoFormer の方を採用
    return stems


def separate(src_wav, work_dir, engine, device="cpu", on_progress=None):
    """src_wav をステムに分ける。{名前: Path} を返す"""
    def prog(pct, detail=""):
        if on_progress:
            on_progress(pct, detail)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if engine == "roformer_hybrid":
        try:
            return _separate_roformer_hybrid(src_wav, work_dir, device, prog)
        except Exception as e:
            print(f"[tabmaker] RoFormer 失敗 ({e}) -> Demucs 6stem に切替", flush=True)
            prog(0, "RoFormer が使えないため Demucs 6stem で続行します")
            return _separate_demucs(src_wav, work_dir, "htdemucs_6s", device, prog)
    model = engine if engine in DEMUCS_MODELS else "htdemucs_6s"
    return _separate_demucs(src_wav, work_dir, model, device, prog)


def cleanup(work_dir):
    shutil.rmtree(work_dir, ignore_errors=True)
