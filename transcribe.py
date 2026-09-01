"""採譜エンジン — 分離済みステムから音符を取り出す

- 音程のあるパート: basic-pitch (Spotify の音符推定AI)。無ければ librosa の pyin で代用。
- ドラム: オンセット検出 + 帯域エネルギーによる打楽器の分類。
"""
import numpy as np
from pathlib import Path

SR = 22050

# パートごとの採譜パラメータ
# poly: 和音を許すか / lo,hi: 想定音域(MIDIノート番号)
PART_CONFIG = {
    "guitar": dict(stem="other", poly=True, lo=40, hi=88,
                   onset=0.55, frame=0.32, min_ms=110, label="ギター"),
    "bass":   dict(stem="bass", poly=False, lo=24, hi=60,
                   onset=0.40, frame=0.28, min_ms=90, label="ベース"),
    "melody": dict(stem="vocals", poly=False, lo=45, hi=88,
                   onset=0.50, frame=0.30, min_ms=100, label="メロディ"),
    "piano":  dict(stem="other", poly=True, lo=36, hi=96,
                   onset=0.50, frame=0.30, min_ms=110, label="ピアノ"),
}

def _stem_configs():
    """同じステムを使うパートは採譜を1回で済ませる (basic-pitch は重いので)"""
    merged = {}
    for part, c in PART_CONFIG.items():
        m = merged.setdefault(c["stem"], dict(c))
        m["lo"] = min(m["lo"], c["lo"])
        m["hi"] = max(m["hi"], c["hi"])
        m["onset"] = min(m["onset"], c["onset"])
        m["frame"] = min(m["frame"], c["frame"])
        m["min_ms"] = min(m["min_ms"], c["min_ms"])
        m["poly"] = m["poly"] or c["poly"]
    return merged


STEM_CONFIG = _stem_configs()

DRUM_LABELS = {
    "kick": "バスドラム", "snare": "スネア", "hihat": "ハイハット",
    "open_hihat": "オープンハイハット", "tom": "タム",
    "crash": "クラッシュ", "ride": "ライド",
}
# 表示順(上から) と GM のノート番号
DRUM_ORDER = ["crash", "ride", "open_hihat", "hihat", "tom", "snare", "kick"]
DRUM_MIDI = {"kick": 36, "snare": 38, "hihat": 42, "open_hihat": 46,
             "tom": 45, "crash": 49, "ride": 51}


# ------------------------------------------------------------------ 高精度モード
def basic_pitch_available():
    try:
        import basic_pitch.inference  # noqa: F401
        return True
    except Exception:
        return False


def _bp_predict(path, cfg):
    """basic-pitch を呼ぶ。バージョン差でシグネチャが違うので inspect で吸収する"""
    import inspect
    import librosa
    from basic_pitch.inference import predict
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH as model_path
    except Exception:
        model_path = None

    sig = inspect.signature(predict)
    kw = {}
    if model_path is not None:
        for name in ("model_or_model_path", "model_path"):
            if name in sig.parameters:
                kw[name] = model_path
                break
    wanted = {
        "onset_threshold": cfg["onset"],
        "frame_threshold": cfg["frame"],
        "minimum_note_length": cfg["min_ms"],
        "minimum_frequency": float(librosa.midi_to_hz(cfg["lo"] - 1)),
        "maximum_frequency": float(librosa.midi_to_hz(cfg["hi"] + 1)),
        "multiple_pitch_bends": False,
        "melodia_trick": True,
    }
    for k, v in wanted.items():
        if k in sig.parameters:
            kw[k] = v
    out = predict(str(path), **kw)
    note_events = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else out
    notes = []
    for ev in note_events:
        start, end, pitch, amp = ev[0], ev[1], int(ev[2]), float(ev[3])
        if end - start < 0.03 or not (cfg["lo"] <= pitch <= cfg["hi"]):
            continue
        notes.append({"start": float(start), "end": float(end), "midi": pitch,
                      "vel": int(np.clip(amp * 160, 30, 127)), "amp": amp})
    notes.sort(key=lambda n: (n["start"], -n["midi"]))
    return notes


# ------------------------------------------------------------------ 軽量モード
def _pyin_notes(path, cfg):
    """単音前提のピッチ追跡 → 同じ音高が続く区間を1音符にまとめる"""
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    if len(y) < sr // 2:
        return []
    hop = 256
    f0, voiced, _ = librosa.pyin(
        y, fmin=float(librosa.midi_to_hz(cfg["lo"])),
        fmax=float(librosa.midi_to_hz(cfg["hi"])), sr=sr,
        frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n], rms[:n]
    midi = np.full(n, np.nan)
    ok = ~np.isnan(f0)
    midi[ok] = librosa.hz_to_midi(f0[ok])

    # 3フレームのメディアンでピッチのゆれを均してから半音に丸める
    sm = np.copy(midi)
    for i in range(1, n - 1):
        w = midi[i - 1:i + 2]
        if not np.isnan(w).any():
            sm[i] = np.median(w)
    rounded = np.where(np.isnan(sm), np.nan, np.round(sm))

    thr = float(np.percentile(rms, 60)) * 0.25
    notes, cur, start = [], None, 0
    min_frames = max(2, int(cfg["min_ms"] / 1000 * sr / hop))

    def flush(end_i):
        if cur is None:
            return
        if end_i - start >= min_frames:
            amp = float(np.mean(rms[start:end_i]) / (rms.max() + 1e-9))
            notes.append({"start": start * hop / sr, "end": end_i * hop / sr,
                          "midi": int(cur), "vel": int(np.clip(amp * 200, 40, 127)),
                          "amp": amp})

    for i in range(n):
        p = rounded[i]
        alive = (not np.isnan(p)) and voiced[i] and rms[i] > thr
        val = int(p) if alive else None
        if val != cur:
            flush(i)
            cur, start = val, i
    flush(n)
    return notes


# ------------------------------------------------------------------ 後処理
def force_monophonic(notes, prefer="low"):
    """重なりを解消して単音の並びにする (ベース・メロディ用)"""
    out = []
    for n in sorted(notes, key=lambda x: (x["start"], -x.get("amp", 0))):
        if not out:
            out.append(dict(n))
            continue
        last = out[-1]
        if n["start"] < last["end"] - 0.02:  # 重なっている
            if n["start"] - last["start"] < 0.06:
                # ほぼ同時 → 片方を捨てる (ベースは低い方、メロディは強い方)
                keep_new = (n["midi"] < last["midi"]) if prefer == "low" \
                    else (n.get("amp", 0) > last.get("amp", 0))
                if keep_new:
                    out[-1] = dict(n)
                continue
            last["end"] = n["start"]  # 前の音を切る
        if n["end"] - max(n["start"], last["end"]) < 0.05:
            continue
        m = dict(n)
        m["start"] = max(m["start"], last["end"])
        out.append(m)
    return [n for n in out if n["end"] - n["start"] > 0.04]


def thin_chords(notes, max_voices=6, window=0.05):
    """同時発音を max_voices 個までに絞る (弦の数を超える和音を防ぐ)"""
    notes = sorted(notes, key=lambda n: n["start"])
    out, i = [], 0
    while i < len(notes):
        j = i
        while j < len(notes) and notes[j]["start"] - notes[i]["start"] < window:
            j += 1
        group = sorted(notes[i:j], key=lambda n: -n.get("amp", 0))[:max_voices]
        base = notes[i]["start"]
        for g in group:
            g = dict(g)
            g["start"] = base
            out.append(g)
        i = j
    out.sort(key=lambda n: (n["start"], -n["midi"]))
    return out


# ------------------------------------------------------------------ 公開API
def _external_notes(path, cmd_template, cfg):
    """外部の採譜スクリプトを呼び、出力されたMIDIを読み込む

    YourMT3+ のような、pipで入らない研究用モデルを繋ぐための拡張点。
    config.json の pitch_external_cmd に {input} と {output} を含むコマンドを書く。
    """
    import subprocess
    import tempfile
    import export

    with tempfile.TemporaryDirectory() as td:
        out_mid = str(Path(td) / "out.mid")
        cmd = cmd_template.format(input=str(path), output=out_mid)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=3600)
        if r.returncode != 0 or not Path(out_mid).exists():
            raise RuntimeError((r.stderr or r.stdout or "")[-400:] or
                               "外部スクリプトがMIDIを出力しませんでした")
        notes = export.read_midi(out_mid)
    return [n for n in notes if n["channel"] != 9
            and cfg["lo"] - 12 <= n["midi"] <= cfg["hi"] + 12]


def transcribe_pitched(path, part, engine="basic_pitch", cache=None,
                       external_cmd=""):
    """音程のあるパートを採譜して 秒単位の音符リストを返す

    cache に dict を渡すと、同じステムの推論結果を使い回して時間を半分にする。
    """
    cfg = PART_CONFIG[part]
    stem_cfg = STEM_CONFIG.get(cfg["stem"], cfg)
    key = str(path)
    notes = None
    if cache is not None and key in cache:
        notes = cache[key]
    if notes is None:
        notes = []
        if engine == "external" and external_cmd:
            try:
                notes = _external_notes(path, external_cmd, stem_cfg)
            except Exception as e:
                print(f"[tabmaker] 外部採譜スクリプト失敗 ({part}): {e} -> basic-pitch に切替",
                      flush=True)
        if not notes and engine != "librosa" and basic_pitch_available():
            try:
                notes = _bp_predict(path, stem_cfg)
            except Exception as e:
                print(f"[tabmaker] basic-pitch 失敗 ({part}): {e} -> 軽量モードに切替",
                      flush=True)
        if not notes:
            notes = _pyin_notes(path, stem_cfg)
        if cache is not None:
            cache[key] = notes

    # このパートの音域に絞る
    notes = [dict(n) for n in notes if cfg["lo"] <= n["midi"] <= cfg["hi"]]
    if cfg["poly"]:
        return thin_chords(notes, max_voices=6)
    return force_monophonic(notes, prefer="low" if part == "bass" else "amp")


DRUM_BANDS = {
    "low": (20, 130),       # バスドラム
    "body": (130, 400),     # タム / スネアの胴鳴り
    "noise": (1500, 8000),  # スネアのスナッピー(ざらつき)
    "high": (8000, 17000),  # シンバル・ハイハット
}
# 打楽器ごとに「音量の目安にする帯域」
BAND_OF = {"kick": "low", "snare": "noise", "tom": "body", "hihat": "high",
           "open_hihat": "high", "crash": "high", "ride": "high"}


def _drum_envs(path):
    """帯域ごとの音量の時系列を作る。A特性重み付けで聴感に近づける"""
    import librosa
    y, sr = librosa.load(str(path), sr=44100, mono=True)
    if len(y) < sr // 2:
        return None
    hop = 256
    P = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    try:  # 等ラウドネス補正 (A特性)
        P = P * (10 ** (librosa.A_weighting(np.maximum(freqs, 1.0)) / 10.0))[:, None]
    except Exception:
        pass
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop,
                                             aggregate=np.median)
    envs = {}
    for k, (a, b) in DRUM_BANDS.items():
        m = (freqs >= a) & (freqs < b)
        e = P[m].mean(axis=0) if m.any() else np.zeros(P.shape[1])
        # 帯域どうしを直接比べると帯域幅の違いだけで勝敗が決まるので、
        # 各帯域を「その曲の中でのその帯域の最大値」で正規化してから判定する
        ref = np.percentile(e, 99.5) + 1e-12
        envs[k] = np.clip(e / ref, 0, 1.5)
    return {"y": y, "sr": sr, "hop": hop, "envs": envs,
            "onset_env": onset_env, "n_frames": P.shape[1]}


def assign_velocities(events, ctx):
    """打点の強さを、その打楽器の帯域のRMS(dB)から決める

    ADTOF等の採譜モデルは音符の位置は当てられるが強弱を返さないことが多い。
    分離したステムのラウドネス曲線から推定し、打楽器の種類ごとに正規化する
    (arXiv:2509.24853 の後処理と同じ考え方)。
    """
    if not events or not ctx:
        return events
    envs, sr, hop, n_frames = ctx["envs"], ctx["sr"], ctx["hop"], ctx["n_frames"]

    def frame_of(t):
        return int(np.clip(round(t * sr / hop), 0, n_frames - 1))

    levels = []
    for e in events:
        band = envs[BAND_OF.get(e["inst"], "noise")]
        f0, f1 = frame_of(e["start"]), frame_of(e["start"] + 0.055)
        levels.append(float(band[f0:max(f1, f0 + 1)].max()))
    db = 20 * np.log10(np.asarray(levels) + 1e-5)

    for inst in {e["inst"] for e in events}:
        idx = [i for i, e in enumerate(events) if e["inst"] == inst]
        vals = db[idx]
        lo, hi = np.percentile(vals, 10), np.percentile(vals, 95)
        span = max(hi - lo, 1e-3)
        for i in idx:
            v = 45 + 82 * (db[i] - lo) / span
            events[i]["vel"] = int(np.clip(v, 35, 127))
    return events


def _drums_builtin(ctx, sensitivity=1.0):
    """オンセットの帯域パワーで打楽器の種類を見分ける (追加DL不要)"""
    import librosa
    envs, sr, hop = ctx["envs"], ctx["sr"], ctx["hop"]
    onset_env, n_frames = ctx["onset_env"], ctx["n_frames"]
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units="time",
        backtrack=True, delta=0.07 / max(sensitivity, 0.2), wait=2)
    if len(onsets) == 0:
        return []

    def frame_of(t):
        return int(np.clip(round(t * sr / hop), 0, n_frames - 1))

    events = []
    for oi, t in enumerate(onsets):
        f0, f1 = frame_of(t), frame_of(t + 0.055)
        sl = slice(f0, max(f1, f0 + 1))
        n = {k: float(v[sl].max()) for k, v in envs.items()}

        # シンバルの減衰: 次の打点の手前までを見る (次の音を拾わないように)
        nxt = onsets[oi + 1] if oi + 1 < len(onsets) else t + 1.0
        d0, d1 = frame_of(t + 0.12), frame_of(min(t + 0.28, nxt - 0.02))
        tail = envs["high"][d0:d1] if d1 > d0 else np.zeros(1)
        decay = float(np.sqrt(max(tail.mean(), 0) / (n["high"] + 1e-12)))

        hits = []
        if n["low"] > 0.30:
            hits.append("kick")
        if n["noise"] > 0.30:
            hits.append("snare")
        elif n["body"] > 0.35 and n["low"] < 0.30:
            hits.append("tom")
        if n["high"] > 0.28 and n["high"] > 0.55 * n["noise"]:
            hits.append("crash" if decay > 0.55 else
                        ("open_hihat" if decay > 0.38 else "hihat"))
        if not hits:  # どの条件にも当てはまらなければ一番強い帯域で決める
            fallback = {"low": "kick", "body": "tom",
                        "noise": "snare", "high": "hihat"}
            hits.append(fallback[max(n.items(), key=lambda x: x[1])[0]])
        for h in hits:
            events.append({"start": float(t), "end": float(t) + 0.12,
                           "inst": h, "vel": 90})
    return events


# ADTOF が出すGMノート番号 -> このアプリの打楽器名
ADTOF_MAP = {35: "kick", 36: "kick",
             37: "snare", 38: "snare", 39: "snare", 40: "snare",
             42: "hihat", 44: "hihat", 46: "open_hihat",
             41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
             49: "crash", 52: "crash", 55: "crash", 57: "crash",
             51: "ride", 53: "ride", 59: "ride"}


def _drums_adtof(path, device="cpu"):
    """ADTOF (学習済みドラム採譜モデル) を使う

    ADTOF-pytorch は torch だけで動くPyTorch移植版で、本家との性能差は約0.2%F値。
    出力はMIDIなので、自前のMIDIリーダーで読み戻す。
    """
    import tempfile
    import export
    from adtof_pytorch import transcribe_to_midi

    with tempfile.TemporaryDirectory() as td:
        out_mid = str(Path(td) / "drums.mid")
        try:
            transcribe_to_midi(str(path), out_mid, device=device)
        except TypeError:      # 古い版は device 引数なし
            transcribe_to_midi(str(path), out_mid)
        notes = export.read_midi(out_mid)

    events = []
    for n in notes:
        inst = ADTOF_MAP.get(n["midi"])
        if not inst:
            continue
        events.append({"start": n["start"], "end": n["start"] + 0.12,
                       "inst": inst, "vel": n["vel"] or 90})
    return events


def transcribe_drums(path, engine="builtin", device="cpu", sensitivity=1.0):
    """ドラムのステムから打点と種類を推定する"""
    ctx = _drum_envs(path)
    if ctx is None:
        return []
    events = []
    if engine == "adtof":
        try:
            events = _drums_adtof(path, device=device)
        except Exception as e:
            print(f"[tabmaker] ADTOF 失敗 ({e}) -> 内蔵ヒューリスティックに切替",
                  flush=True)
    if not events:
        events = _drums_builtin(ctx, sensitivity)
    events = assign_velocities(events, ctx)
    events.sort(key=lambda e: (e["start"], DRUM_ORDER.index(e["inst"])))
    return events
