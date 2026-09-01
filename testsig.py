"""テスト用の合成音源を作る

selftest.py と apitest.py の両方から使う。実際の曲を用意しなくても、
「正解が分かっている音源」で採譜の精度を機械的に確認できるようにするためのもの。
"""
import numpy as np
import soundfile as sf

SR = 44100
TEMPO = 120.0
SPB = 60.0 / TEMPO          # 1拍の秒数
BARS, BPB = 8, 4            # 既定は8小節 / 4拍子
DUR = BARS * BPB * SPB

# 生成中の音の長さ(サンプル数)。build() が小節数に応じて設定する。
# テストによっては短い音源で十分なので、小節数を変えられるようにしてある。
_N = int(DUR * SR)

BASS_SEQ = [40, 43, 45, 47]                  # E2 G2 A2 B2 を4分音符で
MELODY = [64, 66, 67, 69]
GUITAR_CHORD = (52, 56, 59)                  # E major
# I - vi - IV - V (キーC)
PROG = [(60, 64, 67, "C"), (57, 60, 64, "Am"), (53, 57, 60, "F"), (55, 59, 62, "G")]


def tone(midi, start, length, amp=0.5, harm=(1, .45, .2)):
    """減衰する倍音つきの音"""
    f = 440 * 2 ** ((midi - 69) / 12)
    n0, n1 = int(start * SR), min(int((start + length) * SR), _N)
    if n1 <= n0:
        return np.zeros(_N)
    seg = np.arange(n1 - n0) / SR
    env = np.exp(-3.0 * seg / max(length, 1e-3)) * np.minimum(1, seg * 400)
    w = sum(h * np.sin(2 * np.pi * f * k * seg) for k, h in enumerate(harm, 1))
    y = np.zeros(_N)
    y[n0:n1] = amp * env * w
    return y


def noise(start, length, lo, hi, amp=0.5, decay=8.0):
    """帯域ノイズ (sos形式でないと低域で数値的に壊れる)"""
    from scipy.signal import butter, sosfilt
    n0, n1 = int(start * SR), min(int((start + length) * SR), _N)
    if n1 <= n0:
        return np.zeros(_N)
    seg = np.arange(n1 - n0) / SR
    sos = butter(4, [lo / (SR / 2), min(hi, SR / 2 - 1) / (SR / 2)],
                 btype="band", output="sos")
    x = sosfilt(sos, np.random.randn(n1 - n0))
    y = np.zeros(_N)
    y[n0:n1] = amp * x / (np.abs(x).max() + 1e-9) * np.exp(-decay * seg)
    return y


def kick(start, amp=1.0):
    """実物に近いバスドラム: 130Hz -> 45Hz へ落ちるサイン + クリック"""
    length = 0.22
    n0, n1 = int(start * SR), min(int((start + length) * SR), _N)
    if n1 <= n0:
        return np.zeros(_N)
    seg = np.arange(n1 - n0) / SR
    f = 45 + 85 * np.exp(-38 * seg)
    phase = 2 * np.pi * np.cumsum(f) / SR
    y = np.zeros(_N)
    y[n0:n1] = amp * np.sin(phase) * np.exp(-11 * seg)
    return y + noise(start, 0.02, 1500, 6000, amp=amp * 0.12, decay=180)


def snare(start, amp=1.0):
    """スネア: 190Hzの胴鳴り + 広帯域ノイズ"""
    length = 0.18
    n0, n1 = int(start * SR), min(int((start + length) * SR), _N)
    seg = np.arange(max(n1 - n0, 0)) / SR
    y = np.zeros(_N)
    if n1 > n0:
        y[n0:n1] = amp * 0.5 * np.sin(2 * np.pi * 190 * seg) * np.exp(-24 * seg)
    return y + noise(start, length, 1200, 7000, amp=amp * 0.8, decay=26)


def _norm(y, peak=0.9):
    return y / (np.abs(y).max() + 1e-9) * peak


def build(out_dir, bars=BARS):
    """テスト用のステム一式を書き出し、正解データを返す

    bars を小さくすると短い音源になり、テストが速く終わる。
    """
    global _N
    _N = int(bars * BPB * SPB * SR)
    out_dir.mkdir(parents=True, exist_ok=True)

    bass = np.zeros(_N)
    expected_bass = []
    for bar in range(bars):
        for k in range(BPB):
            m = BASS_SEQ[k]
            bass += tone(m, (bar * BPB + k) * SPB, SPB * 0.9, amp=0.6,
                         harm=(1, .5, .25))
            expected_bass.append((bar * BPB + k, m))

    drums = np.zeros(_N)
    expected_drums = []
    for bar in range(bars):
        for k in range(BPB):
            st = (bar * BPB + k) * SPB
            if k in (0, 2):
                drums += kick(st, amp=1.0)
                expected_drums.append((bar * BPB + k, "kick"))
            if k in (1, 3):
                drums += snare(st, amp=0.8)
                expected_drums.append((bar * BPB + k, "snare"))
            for h in (0, 0.5):
                drums += noise(st + h * SPB, .05, 7000, 16000, amp=0.35, decay=90)

    other = np.zeros(_N)
    for bar in range(bars):
        for m in GUITAR_CHORD:
            other += tone(m, bar * BPB * SPB, SPB * 3.5, amp=0.22)

    vocals = np.zeros(_N)
    for bar in range(bars):
        for k in range(BPB):
            vocals += tone(MELODY[(bar + k) % 4], (bar * BPB + k) * SPB,
                           SPB * .8, amp=0.5, harm=(1, .3, .12))

    # コード認識用: I-vi-IV-V の和音とそのルート
    chord_h, chord_b, expected_chords = np.zeros(_N), np.zeros(_N), []
    for bar in range(bars):
        root, third, fifth, name = PROG[bar % 4]
        st = bar * BPB * SPB
        for m in (root, third, fifth):
            chord_h += tone(m, st, SPB * 3.8, amp=0.30, harm=(1, .4, .18))
        chord_b += tone(root - 24, st, SPB * 3.8, amp=0.6, harm=(1, .5, .2))
        expected_chords.append(name)

    # 強弱の差をつけたドラム (ベロシティ推定の検証用)
    dyn = np.zeros(_N)
    for bar in range(bars):
        for k in range(BPB):
            dyn += kick((bar * BPB + k) * SPB, amp=1.0 if k == 0 else 0.35)

    mix = bass + drums + other + vocals
    files = {}
    for name, y in [("mix", _norm(mix, 0.9)), ("bass", bass), ("drums", drums),
                    ("other", other), ("vocals", vocals),
                    ("chord_harm", _norm(chord_h, 0.8)),
                    ("chord_bass", _norm(chord_b, 0.8)), ("dyn", _norm(dyn))]:
        p = out_dir / f"{name}.wav"
        sf.write(p, y, SR)
        files[name] = p

    return {"files": files, "expected_bass": expected_bass,
            "expected_drums": expected_drums, "expected_chords": expected_chords,
            "tempo": TEMPO, "bars": bars, "beats_per_bar": BPB,
            "duration": bars * BPB * SPB, "sr": SR, "spb": SPB}
