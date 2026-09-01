"""コード進行の推定

外部ライブラリを増やさずに済ませるため、クロマ特徴とテンプレート照合 + ビタビ探索で実装する。
音符単位の採譜が多少荒くても、コード進行が合っていればギター譜としては十分に使えるので、
実用性への寄与が大きい。

流れ:
  1. ドラム以外のステムを足して和声成分だけの音を作る
  2. CQTクロマを拍ごとに集約する (コードは拍の頭で変わることが多い)
  3. 24〜48種のコードテンプレートとの一致度を出す
  4. ベース音のクロマでルートを補正する
  5. ビタビ探索で「コロコロ変わらない」進行に均す
"""
import numpy as np

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# 品質ごとの (構成音からの半音, 重み) と、出やすさの補正
QUALITIES = [
    ("maj",  [(0, 1.0), (4, 0.9), (7, 0.8)],              0.00, "major"),
    ("min",  [(0, 1.0), (3, 0.9), (7, 0.8)],              0.00, "minor"),
    ("7",    [(0, 1.0), (4, 0.85), (7, 0.65), (10, 0.6)], -0.05, "dominant"),
    ("min7", [(0, 1.0), (3, 0.85), (7, 0.65), (10, 0.6)], -0.05, "minor-seventh"),
]
SUFFIX = {"maj": "", "min": "m", "7": "7", "min7": "m7"}


def _build_templates():
    """(状態数, 12) のテンプレート行列と、状態のメタ情報を作る"""
    mats, meta = [], []
    for root in range(12):
        for name, tones, bias, xml_kind in QUALITIES:
            v = np.zeros(12)
            for semi, w in tones:
                v[(root + semi) % 12] = w
            mats.append(v / np.linalg.norm(v))
            meta.append({"root": root, "quality": name, "bias": bias,
                         "kind": xml_kind})
    mats.append(np.full(12, 1 / np.sqrt(12)))       # N.C.
    meta.append({"root": None, "quality": "N", "bias": -0.02, "kind": "none"})
    return np.array(mats), meta


TEMPLATES, META = _build_templates()


def _load_sum(paths, sr):
    import librosa
    total = None
    for p in paths:
        try:
            y, _ = librosa.load(str(p), sr=sr, mono=True)
        except Exception:
            continue
        if total is None:
            total = y
        else:
            n = min(len(total), len(y))
            total = total[:n] + y[:n]
    return total


def recognize(harmonic_paths, bass_path, grid, sr=22050, hop=2048,
              flats=False, min_beats=1):
    """コード進行を推定して [{b, d, name, root, quality, kind}] を返す"""
    import librosa

    y = _load_sum(harmonic_paths, sr)
    if y is None or len(y) < sr:
        return []

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop,
                                        bins_per_octave=36)
    energy = chroma.sum(axis=0)
    n_frames = chroma.shape[1]

    # 拍の境界でフレームを区切る
    bt = np.asarray(grid.beat_times, dtype=float)
    if len(bt) < 2:
        return []
    edges = np.clip(librosa.time_to_frames(bt, sr=sr, hop_length=hop),
                    0, n_frames)
    edges = np.unique(edges)
    if len(edges) < 3:
        return []
    n_beats = len(edges) - 1

    def _sync(mat):
        out = np.zeros((mat.shape[0], n_beats))
        for i in range(n_beats):
            a, b = edges[i], max(edges[i + 1], edges[i] + 1)
            out[:, i] = np.median(mat[:, a:b], axis=1)
        return out

    beat_chroma = _sync(chroma)
    beat_energy = np.array([np.median(energy[edges[i]:max(edges[i+1], edges[i]+1)])
                            for i in range(n_beats)])

    # ベースのクロマ (ルート補正用)
    bass_chroma = None
    if bass_path:
        try:
            yb, _ = librosa.load(str(bass_path), sr=sr, mono=True)
            cb = librosa.feature.chroma_cqt(y=yb, sr=sr, hop_length=hop,
                                            bins_per_octave=36,
                                            fmin=librosa.note_to_hz("C1"),
                                            n_octaves=4)
            cb = cb[:, :n_frames] if cb.shape[1] >= n_frames else np.pad(
                cb, ((0, 0), (0, n_frames - cb.shape[1])))
            bass_chroma = _sync(cb)
        except Exception:
            bass_chroma = None

    # 一致度 -> 確率
    norms = np.linalg.norm(beat_chroma, axis=0) + 1e-9
    sim = TEMPLATES @ (beat_chroma / norms)          # (状態, 拍)
    sim += np.array([m["bias"] for m in META])[:, None]

    if bass_chroma is not None:
        bn = bass_chroma / (np.linalg.norm(bass_chroma, axis=0) + 1e-9)
        bonus = np.zeros_like(sim)
        for si, m in enumerate(META):
            if m["root"] is not None:
                bonus[si] = 0.18 * bn[m["root"]]
        sim += bonus

    # 音が薄い拍は N.C. に寄せる
    quiet = beat_energy < np.percentile(beat_energy, 12)
    sim[-1, quiet] += 0.25

    prob = np.exp((sim - sim.max(axis=0)) / 0.08)
    prob /= prob.sum(axis=0, keepdims=True)

    # ビタビ: コードは頻繁には変わらない
    n_states = len(META)
    stay = 0.90
    trans = np.full((n_states, n_states), (1 - stay) / (n_states - 1))
    np.fill_diagonal(trans, stay)
    try:
        path = librosa.sequence.viterbi(prob, trans)
    except Exception:
        path = prob.argmax(axis=0)

    # 同じコードが続く区間をまとめる
    segs, start = [], 0
    for i in range(1, n_beats + 1):
        if i == n_beats or path[i] != path[start]:
            m = META[path[start]]
            b0 = start - grid.phase
            length = i - start
            segs.append({"b": round(float(b0), 4), "d": float(length),
                         "state": int(path[start]), **m})
            start = i

    # 1拍だけの飛び道具を前後に吸収させる
    merged = []
    for s in segs:
        if merged and s["d"] < min_beats and s["state"] != merged[-1]["state"]:
            merged[-1]["d"] += s["d"]
            continue
        if merged and s["state"] == merged[-1]["state"]:
            merged[-1]["d"] += s["d"]
            continue
        merged.append(s)

    out = []
    names = FLAT if flats else SHARP
    for s in merged:
        if s["b"] + s["d"] <= 0:
            continue
        b = max(s["b"], 0.0)
        d = s["d"] - (b - s["b"])
        if d <= 0:
            continue
        if s["root"] is None:
            name = "N.C."
        else:
            name = names[s["root"]] + SUFFIX[s["quality"]]
        out.append({"b": round(b, 4), "d": round(d, 4), "name": name,
                    "root": s["root"], "quality": s["quality"],
                    "kind": s["kind"]})
    return out


def to_musicxml_harmony(chord, flats=False):
    """MusicXML の <harmony> 要素を作る"""
    if chord["root"] is None:
        return ""
    names = FLAT if flats else SHARP
    label = names[chord["root"]]
    step = label[0]
    alter = 1 if label.endswith("#") else (-1 if label.endswith("b") else 0)
    s = "<harmony><root><root-step>" + step + "</root-step>"
    if alter:
        s += f"<root-alter>{alter}</root-alter>"
    s += "</root>"
    s += f'<kind text="{_display(chord)}">{chord["kind"]}</kind>'
    return s + "</harmony>"


def _display(chord):
    return SUFFIX.get(chord["quality"], "")
