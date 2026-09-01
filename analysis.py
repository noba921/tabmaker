"""テンポ・拍グリッド・調の推定と、秒 <-> 拍 の変換

譜面にするには「何秒に鳴ったか」ではなく「何拍目に鳴ったか」が要る。
このモジュールは音源から拍の位置を割り出し、時刻を拍に変換する Grid を作る。
"""
import numpy as np

# Krumhansl-Kessler の調プロファイル (長調・短調)
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 調号 (シャープ数を正・フラット数を負で表す MusicXML の fifths)
_FIFTHS_MAJOR = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}


class Grid:
    """時刻(秒) <-> 拍 の相互変換。拍0 = 最初の小節頭。"""

    def __init__(self, beat_times, tempo, beats_per_bar=4, downbeat_phase=0):
        self.beat_times = np.asarray(beat_times, dtype=float)
        self.tempo = float(tempo)
        self.beats_per_bar = int(beats_per_bar)
        self.phase = int(downbeat_phase)
        self.sec_per_beat = 60.0 / max(self.tempo, 1e-6)

    # -- 変換 ---------------------------------------------------------
    def time_to_beat(self, t):
        t = np.asarray(t, dtype=float)
        bt = self.beat_times
        if len(bt) < 2:
            return t / self.sec_per_beat - self.phase
        idx = np.arange(len(bt), dtype=float)
        b = np.interp(t, bt, idx)
        b = np.where(t < bt[0], (t - bt[0]) / self.sec_per_beat, b)
        b = np.where(t > bt[-1], (len(bt) - 1) + (t - bt[-1]) / self.sec_per_beat, b)
        return b - self.phase

    def beat_to_time(self, b):
        b = np.asarray(b, dtype=float) + self.phase
        bt = self.beat_times
        if len(bt) < 2:
            return b * self.sec_per_beat
        idx = np.arange(len(bt), dtype=float)
        t = np.interp(b, idx, bt)
        t = np.where(b < 0, bt[0] + b * self.sec_per_beat, t)
        t = np.where(b > len(bt) - 1, bt[-1] + (b - (len(bt) - 1)) * self.sec_per_beat, t)
        return t

    def to_dict(self):
        return {"tempo": round(self.tempo, 2), "beats_per_bar": self.beats_per_bar,
                "sec_per_beat": self.sec_per_beat,
                "beat_times": [round(float(x), 4) for x in self.beat_times],
                "phase": self.phase,
                "bar0_sec": round(float(self.beat_to_time(0)), 4)}


def estimate_grid_beat_this(audio_path, device="cpu", model="final0"):
    """Beat This! (ISMIR 2024) で拍とダウンビートを取る

    自作ヒューリスティックと違い、小節頭そのものをモデルが出力するので位相を当てにいかなくて済む。
    拍子(1小節が何拍か)もダウンビート間隔から数えられる。
    """
    from beat_this.inference import File2Beats
    f2b = File2Beats(checkpoint_path=model, device=device, dbn=False)
    beats, downbeats = f2b(str(audio_path))
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) < 4:
        raise RuntimeError("Beat This! が拍を検出できませんでした")

    tempo = 60.0 / float(np.median(np.diff(beats)))

    # ダウンビートを拍列の添字に対応づける
    phase, beats_per_bar = 0, 4
    if len(downbeats):
        idx = np.unique(np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1))
        # 実際にダウンビートの近くにある拍だけ採用する
        idx = np.array([i for i in idx
                        if abs(beats[i] - downbeats[np.abs(downbeats - beats[i]).argmin()])
                        < 0.5 * 60.0 / tempo], dtype=int)
        if len(idx):
            phase = int(idx[0])
            if len(idx) > 1:
                step = float(np.median(np.diff(idx)))
                if 1.5 <= step <= 12.5:
                    beats_per_bar = int(round(step))
    beats_per_bar = int(np.clip(beats_per_bar, 2, 12))
    return Grid(beats, tempo, beats_per_bar, phase)


def estimate_grid_auto(audio_path, y, sr, engine="librosa", device="cpu",
                       beats_per_bar=4):
    """設定に応じて拍推定のバックエンドを選ぶ。失敗したら librosa に落ちる"""
    if engine == "beat_this":
        try:
            g = estimate_grid_beat_this(audio_path, device=device)
            print(f"[tabmaker] Beat This!: {g.tempo:.1f} BPM / "
                  f"{g.beats_per_bar}拍子 / 小節頭 {g.beat_to_time(0):.2f}s", flush=True)
            return g, "beat_this"
        except Exception as e:
            print(f"[tabmaker] Beat This! 失敗 ({e}) -> librosa に切替", flush=True)
    return estimate_grid(y, sr, beats_per_bar=beats_per_bar), "librosa"


def estimate_grid(y, sr, beats_per_bar=4, tempo_hint=None):
    """全体ミックスから拍と小節頭を推定する (librosa 版・追加DL不要)"""
    import librosa
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    kw = {"onset_envelope": onset_env, "sr": sr, "units": "frames", "trim": False}
    if tempo_hint:
        kw["start_bpm"] = float(tempo_hint)
    tempo, beat_frames = librosa.beat.beat_track(**kw)
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 4:  # 拍が取れないほど静かな曲へのフォールバック
        beat_times = np.arange(0.0, len(y) / sr, 60.0 / tempo)
        return Grid(beat_times, tempo, beats_per_bar, 0)

    # 実測の拍間隔からテンポを取り直す (beat_track の倍/半テンポずれ対策)
    med = float(np.median(np.diff(beat_times)))
    if med > 0:
        tempo = 60.0 / med

    # 小節頭の位相を決める。
    # 手がかりは「その拍のオンセットが強いか」と「低音の打点(バスドラム)が来ているか」。
    # 低音は帯域を絞ったオンセット強度で見る (総エネルギーだと高域のビン数に負ける)。
    fi = np.clip(beat_frames, 0, len(onset_env) - 1)
    strength = onset_env[fi]
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=16, fmax=200)
        low_env = librosa.onset.onset_strength(
            S=librosa.power_to_db(mel, ref=np.max), sr=sr)
        li = np.clip(beat_frames, 0, len(low_env) - 1)
        bass = low_env[li]
        bass = np.clip(bass, 0, None)
        bass = bass / (bass.max() + 1e-9)
    except Exception:
        bass = np.zeros_like(strength)
    # オンセット強度はスネア(広帯域)に引っぱられやすいので、重みは控えめにする
    score = 0.25 * strength / (strength.max() + 1e-9) + bass

    phase, best = 0, -1.0
    for p in range(beats_per_bar):
        s = float(np.mean(score[p::beats_per_bar])) if len(score) > p else 0.0
        if s > best * 1.05:  # 僅差なら小さい位相 (曲の頭) を優先する
            best, phase = s, p
    return Grid(beat_times, tempo, beats_per_bar, phase)


def estimate_key(y, sr):
    """調を推定して (表示名, MusicXMLのfifths, is_minor) を返す"""
    import librosa
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    except Exception:
        return "C major", 0, False
    chroma = chroma / (chroma.sum() + 1e-9)
    best = (None, -2.0)
    for root in range(12):
        for prof, minor in ((_MAJ, False), (_MIN, True)):
            rolled = np.roll(prof, root)
            c = float(np.corrcoef(chroma, rolled)[0, 1])
            if c > best[1]:
                best = ((root, minor), c)
    root, minor = best[0]
    if minor:  # 平行長調の調号を使う
        fifths = _FIFTHS_MAJOR.get((root + 3) % 12, 0)
        return f"{_PITCH_NAMES[root]} minor", fifths, True
    return f"{_PITCH_NAMES[root]} major", _FIFTHS_MAJOR.get(root, 0), False


def quantize(beats, div=4):
    """拍を 1/div 拍単位に丸める (div=4 なら16分音符グリッド)"""
    return np.round(np.asarray(beats, dtype=float) * div) / div


def quantize_notes(notes, grid, div=4, min_dur=0.25):
    """秒単位の音符リストを拍単位に変換して量子化する。

    notes: [{"start": 秒, "end": 秒, "midi": int, "vel": 0-127}, ...]
    返り値: [{"b": 開始拍, "d": 長さ拍, "midi": int, "vel": int}, ...]
    """
    if not notes:
        return []
    starts = grid.time_to_beat([n["start"] for n in notes])
    ends = grid.time_to_beat([n["end"] for n in notes])
    qs = quantize(starts, div)
    qe = quantize(ends, div)
    out = []
    for n, b, e in zip(notes, qs, qe):
        d = max(float(e - b), min_dur)
        if b < -min_dur:  # グリッドより前の音は捨てる
            continue
        out.append({"b": round(max(float(b), 0.0), 4), "d": round(d, 4),
                    "midi": int(n["midi"]), "vel": int(n.get("vel", 90))})
    out.sort(key=lambda n: (n["b"], -n["midi"]))
    return out
