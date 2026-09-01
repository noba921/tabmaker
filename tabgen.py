"""TAB譜の生成 — 音高の並びを「どの弦の何フレットで押さえるか」に変換する

同じ音でも押さえ方は複数ある。動的計画法で「左手の移動が少ない運指」を選ぶ。
"""
import numpy as np

# 弦は 1弦(高い方) から順に並べる。値は開放弦の MIDI ノート番号。
TUNINGS = {
    "guitar": {
        "standard":  {"label": "レギュラー (E A D G B E)", "strings": [64, 59, 55, 50, 45, 40]},
        "half_down": {"label": "半音下げ (E♭)",           "strings": [63, 58, 54, 49, 44, 39]},
        "whole_down": {"label": "全音下げ (D)",            "strings": [62, 57, 53, 48, 43, 38]},
        "drop_d":    {"label": "ドロップD",                "strings": [64, 59, 55, 50, 45, 38]},
        "dadgad":    {"label": "DADGAD",                   "strings": [62, 57, 55, 50, 45, 38]},
    },
    "bass": {
        "standard":  {"label": "レギュラー (E A D G)", "strings": [43, 38, 33, 28]},
        "half_down": {"label": "半音下げ (E♭)",        "strings": [42, 37, 32, 27]},
        "drop_d":    {"label": "ドロップD",            "strings": [43, 38, 33, 26]},
        "five":      {"label": "5弦 (B E A D G)",      "strings": [43, 38, 33, 28, 23]},
    },
}
DEFAULT_MAX_FRET = {"guitar": 17, "bass": 20}


def fold_into_range(notes, strings, capo=0, max_fret=17):
    """音域外の音をオクターブ単位で移して鳴らせるようにする"""
    lo = min(strings) + capo
    hi = max(strings) + capo + max_fret
    out = []
    for n in notes:
        m = n["midi"]
        while m < lo:
            m += 12
        while m > hi:
            m -= 12
        if lo <= m <= hi:
            q = dict(n)
            q["midi"] = m
            if m != n["midi"]:      # オクターブを動かした音だけ印を付ける
                q["moved"] = True   # (全音符に付けると score.json が無駄に膨らむ)
            out.append(q)
    return out


def _assignments(midis, strings, capo, max_fret, limit=40):
    """同時に鳴る音に弦を割り当てる候補を列挙する (高音ほど細い弦)"""
    res = []
    n = len(midis)

    def rec(i, min_s, cur):
        if len(res) >= limit:
            return
        if i == n:
            res.append(list(cur))
            return
        for s in range(min_s, len(strings)):
            fret = midis[i] - strings[s] - capo
            if 0 <= fret <= max_fret:
                cur.append((s, fret))
                rec(i + 1, s + 1, cur)
                cur.pop()
    rec(0, 0, [])
    return res


def _shape_cost(shape):
    """1つの押さえ方の弾きにくさ"""
    frets = [f for _, f in shape]
    used = [f for f in frets if f > 0]
    span = (max(used) - min(used)) if used else 0
    open_bonus = 0.35 * sum(1 for f in frets if f == 0)
    high_pen = 0.03 * (np.mean(used) if used else 0)
    strings = [s for s, _ in shape]
    gap_pen = 0.15 * ((max(strings) - min(strings)) - (len(strings) - 1)) if len(strings) > 1 else 0
    return span * 0.9 + high_pen + gap_pen - open_bonus, span


def _anchor(shape):
    used = [f for _, f in shape if f > 0]
    return min(used) if used else None


def make_tab(notes, instrument="guitar", tuning="standard", capo=0, max_fret=None):
    """量子化済みの音符に string / fret を付ける

    notes: [{"b": 開始拍, "d": 長さ拍, "midi": int, "vel": int}, ...]
    """
    tun = TUNINGS[instrument].get(tuning) or TUNINGS[instrument]["standard"]
    strings = tun["strings"]
    max_fret = max_fret or DEFAULT_MAX_FRET[instrument]
    notes = fold_into_range(notes, strings, capo, max_fret)
    if not notes:
        return [], strings

    # 同時発音でグループ化
    groups, cur = [], []
    for n in sorted(notes, key=lambda x: (x["b"], -x["midi"])):
        if cur and abs(n["b"] - cur[0]["b"]) > 1e-6:
            groups.append(cur)
            cur = []
        cur.append(n)
    if cur:
        groups.append(cur)

    # 各グループの候補を数個に絞る
    cand_per_group = []
    for g in groups:
        midis = [x["midi"] for x in g]
        shapes = _assignments(midis, strings, capo, max_fret)
        scored = []
        for sh in shapes:
            cost, span = _shape_cost(sh)
            if len(sh) > 1 and span > 5:  # 手が届かない押さえ方は除外
                continue
            scored.append((cost, sh))
        if not scored:  # 届かないものしか無ければ制限を外して最良を採用
            for sh in shapes:
                scored.append((_shape_cost(sh)[0] + 3.0, sh))
        if not scored:
            cand_per_group.append([])
            continue
        scored.sort(key=lambda x: x[0])
        cand_per_group.append(scored[:6])

    # 動的計画法: 左手のポジション移動が少ない経路を選ぶ
    prev = None  # [(累積コスト, anchor, 選んだshape, 親index)]
    table = []
    for gi, cands in enumerate(cand_per_group):
        if not cands:
            table.append([])
            continue
        layer = []
        for cost, sh in cands:
            a = _anchor(sh)
            if prev is None:
                layer.append([cost, a, sh, -1])
                continue
            best_i, best_c = 0, float("inf")
            for pi, p in enumerate(prev):
                move = 0.0 if (a is None or p[1] is None) else abs(a - p[1]) * 0.45
                c = p[0] + cost + move
                if c < best_c:
                    best_c, best_i = c, pi
            layer.append([best_c, a if a is not None else
                          (prev[best_i][1] if prev else None), sh, best_i])
        table.append(layer)
        prev = layer

    # 経路の復元
    out_groups = [None] * len(groups)
    li = len(table) - 1
    while li >= 0 and not table[li]:
        li -= 1
    if li < 0:
        return [], strings
    k = int(np.argmin([r[0] for r in table[li]]))
    while li >= 0:
        if not table[li]:
            li -= 1
            continue
        row = table[li][k]
        out_groups[li] = row[2]
        k = row[3]
        li -= 1
        if k < 0:
            break

    result = []
    for g, sh in zip(groups, out_groups):
        if sh is None:
            continue
        for note, (s, f) in zip(g, sh):
            q = dict(note)
            q["string"] = int(s) + 1        # 1弦 = 1
            q["fret"] = int(f)
            result.append(q)
    result.sort(key=lambda n: (n["b"], n["string"]))
    return result, strings


def tab_stats(notes):
    """譜面の難易度の目安"""
    if not notes:
        return {}
    frets = [n["fret"] for n in notes if n.get("fret", 0) > 0]
    return {
        "notes": len(notes),
        "max_fret": max(frets) if frets else 0,
        "min_fret": min(frets) if frets else 0,
        "open_ratio": round(sum(1 for n in notes if n.get("fret") == 0) / len(notes), 2),
    }
