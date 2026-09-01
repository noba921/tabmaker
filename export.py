"""楽譜ファイルの書き出し — MusicXML と MIDI

MusicXML は MuseScore / Guitar Pro / Dorico などで開いて清書・編集できる。
外部ライブラリを使わずに自前で書き出しているので、依存が増えない。
"""
import struct
from xml.sax.saxutils import escape

DIVISIONS = 4  # 4分音符あたりの分解能 = 16分音符グリッド

# (tick数, 音符の種類, 付点)
_DUR_TABLE = [(16, "whole", 0), (12, "half", 1), (8, "half", 0),
              (6, "quarter", 1), (4, "quarter", 0), (3, "eighth", 1),
              (2, "eighth", 0), (1, "16th", 0)]

_SHARP_NAMES = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
                ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]
_FLAT_NAMES = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
               ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0)]

# ドラムの五線上の位置と符頭
DRUM_NOTATION = {
    "kick":       ("F", 4, "normal"),
    "snare":      ("C", 5, "normal"),
    "tom":        ("E", 5, "normal"),
    "hihat":      ("G", 5, "x"),
    "open_hihat": ("G", 5, "circle-x"),
    "ride":       ("F", 5, "x"),
    "crash":      ("A", 5, "x"),
}
DRUM_MIDI = {"kick": 36, "snare": 38, "hihat": 42, "open_hihat": 46,
             "tom": 45, "crash": 49, "ride": 51}


def _snap(ticks, limit=None):
    """表記できる音価に丸める"""
    t = int(max(1, round(ticks)))
    if limit is not None:
        t = min(t, int(limit))
    t = max(1, t)
    for val, _, _ in _DUR_TABLE:
        if val <= t:
            return val
    return 1


def _type_of(ticks):
    for val, name, dots in _DUR_TABLE:
        if val == ticks:
            return name, dots
    return "16th", 0


def _fill_rests(gap):
    out = []
    remain = int(gap)
    while remain > 0:
        d = _snap(remain)
        out.append(d)
        remain -= d
    return out


def _pitch_xml(midi, flats=False):
    table = _FLAT_NAMES if flats else _SHARP_NAMES
    step, alter = table[midi % 12]
    octave = midi // 12 - 1
    s = f"<pitch><step>{step}</step>"
    if alter:
        s += f"<alter>{alter}</alter>"
    return s + f"<octave>{octave}</octave></pitch>"


def _group_events(notes, key="b"):
    """同じ開始拍の音符をまとめ、[(tick, [notes])] を返す"""
    buckets = {}
    for n in notes:
        t = int(round(n[key] * DIVISIONS))
        buckets.setdefault(max(t, 0), []).append(n)
    return [(t, buckets[t]) for t in sorted(buckets)]


# ------------------------------------------------------------------ MusicXML
def _measure_body(events, m_start, m_len, render_note, flats, harmonies=()):
    """1小節分の <note> 群を組み立てる

    harmonies: [(小節内のtick, <harmony>のXML)] — コードネームを音符の前に差し込む
    """
    xml, pos = [], 0
    pending = sorted(harmonies)
    hi = 0

    def flush_harmony(upto):
        nonlocal hi
        while hi < len(pending) and pending[hi][0] <= upto:
            xml.append(pending[hi][1])
            hi += 1

    def add_rests(n_ticks):
        nonlocal pos
        for d in _fill_rests(n_ticks):
            flush_harmony(pos)
            ntype, dots = _type_of(d)
            xml.append(f'<note><rest/><duration>{d}</duration>'
                       f'<voice>1</voice><type>{ntype}</type>'
                       + "<dot/>" * dots + "</note>")
            pos += d

    inside = [(t - m_start, ns) for t, ns in events if m_start <= t < m_start + m_len]
    for i, (off, ns) in enumerate(inside):
        if off < pos:
            continue
        if off > pos:
            add_rests(off - pos)
            pos = off
        nxt = inside[i + 1][0] if i + 1 < len(inside) else m_len
        want = max(n.get("d", 1.0) for n in ns) * DIVISIONS
        dur = _snap(min(want, nxt - off, m_len - off))
        flush_harmony(off)
        for j, n in enumerate(ns):
            xml.append(render_note(n, dur, chord=(j > 0), flats=flats))
        pos = off + dur
    if pos < m_len:
        add_rests(m_len - pos)
    flush_harmony(m_len)
    return "".join(xml)


def _render_pitched(n, dur, chord, flats, tab=None):
    ntype, dots = _type_of(dur)
    s = "<note>"
    if chord:
        s += "<chord/>"
    s += _pitch_xml(n["midi"], flats)
    s += f"<duration>{dur}</duration><voice>1</voice><type>{ntype}</type>"
    s += "<dot/>" * dots
    if tab and "string" in n and "fret" in n:
        s += ("<notations><technical>"
              f"<string>{n['string']}</string><fret>{n['fret']}</fret>"
              "</technical></notations>")
    s += "</note>"
    return s


def _render_drum(n, dur, chord, flats):
    ntype, dots = _type_of(dur)
    step, octv, head = DRUM_NOTATION.get(n.get("inst", "snare"), ("C", 5, "normal"))
    s = "<note>"
    if chord:
        s += "<chord/>"
    s += (f"<unpitched><display-step>{step}</display-step>"
          f"<display-octave>{octv}</display-octave></unpitched>"
          f"<duration>{dur}</duration>"
          f"<instrument id=\"DRUM-{n.get('inst','snare')}\"/>"
          f"<voice>1</voice><type>{ntype}</type>")
    s += "<dot/>" * dots
    if head != "normal":
        s += f"<notehead>{head}</notehead>"
    s += "</note>"
    return s


def build_musicxml(parts, tempo=120.0, beats_per_bar=4, fifths=0,
                   title="Untitled", software="TAB Maker", chords=None):
    """parts: [{"id","name","kind","tuning","capo","notes"}] -> MusicXML文字列

    chords を渡すと、先頭パートの上にコードネームを載せる。
    """
    flats = fifths < 0
    m_len = int(beats_per_bar * DIVISIONS)

    # コードネームを小節ごとに振り分ける
    harmony_by_measure = {}
    if chords:
        import chords as chordmod
        for c in chords:
            tick = int(round(c["b"] * DIVISIONS))
            if tick < 0:
                continue
            xml = chordmod.to_musicxml_harmony(c, flats=flats)
            if xml:
                m = tick // m_len
                harmony_by_measure.setdefault(m, []).append((tick % m_len, xml))

    # 小節数
    last = 0
    for p in parts:
        for n in p["notes"]:
            last = max(last, (n["b"] + n.get("d", 1.0)) * DIVISIONS)
    n_measures = max(1, int(-(-last // m_len)))

    head = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">',
            '<score-partwise version="3.1">',
            f'<work><work-title>{escape(title)}</work-title></work>',
            f'<identification><encoding><software>{escape(software)}</software>'
            '</encoding></identification>',
            '<part-list>']
    for p in parts:
        head.append(f'<score-part id="{p["id"]}">'
                    f'<part-name>{escape(p["name"])}</part-name>')
        if p["kind"] == "drums":
            used = sorted({n["inst"] for n in p["notes"]})
            for u in used:
                head.append(f'<score-instrument id="DRUM-{u}">'
                            f'<instrument-name>{u}</instrument-name></score-instrument>')
            for u in used:
                head.append(f'<midi-instrument id="DRUM-{u}"><midi-channel>10</midi-channel>'
                            f'<midi-unpitched>{DRUM_MIDI.get(u,38)+1}</midi-unpitched>'
                            '</midi-instrument>')
        else:
            head.append(f'<score-instrument id="{p["id"]}-I1">'
                        f'<instrument-name>{escape(p["name"])}</instrument-name></score-instrument>'
                        f'<midi-instrument id="{p["id"]}-I1"><midi-channel>{p.get("channel",1)}</midi-channel>'
                        f'<midi-program>{p.get("program",1)}</midi-program></midi-instrument>')
        head.append('</score-part>')
    head.append('</part-list>')
    out = head

    for pi, p in enumerate(parts):
        out.append(f'<part id="{p["id"]}">')
        events = _group_events(p["notes"])
        for m in range(n_measures):
            out.append(f'<measure number="{m+1}">')
            if m == 0:
                attrs = [f'<divisions>{DIVISIONS}</divisions>',
                         f'<key><fifths>{fifths}</fifths></key>',
                         f'<time><beats>{beats_per_bar}</beats>'
                         f'<beat-type>4</beat-type></time>']
                if p["kind"] == "tab":
                    tun = p.get("tuning") or []
                    lines = len(tun)
                    det = [f'<staff-details><staff-lines>{lines}</staff-lines>']
                    # MusicXML の弦番号は 1 = 一番高い弦
                    for i, midi in enumerate(tun):
                        step, alter = _SHARP_NAMES[midi % 12]
                        det.append(f'<staff-tuning line="{lines-i}">'
                                   f'<tuning-step>{step}</tuning-step>'
                                   + (f'<tuning-alter>{alter}</tuning-alter>' if alter else '')
                                   + f'<tuning-octave>{midi//12-1}</tuning-octave></staff-tuning>')
                    det.append('</staff-details>')
                    if p.get("capo"):
                        det.insert(1, f'<capo>{p["capo"]}</capo>')
                    attrs.append("".join(det))
                    attrs.append('<clef><sign>TAB</sign><line>5</line></clef>')
                elif p["kind"] == "drums":
                    attrs.append('<clef><sign>percussion</sign><line>2</line></clef>')
                else:
                    sign, line = p.get("clef", ("G", 2))
                    attrs.append(f'<clef><sign>{sign}</sign><line>{line}</line></clef>')
                out.append('<attributes>' + "".join(attrs) + '</attributes>')
                out.append('<direction placement="above"><direction-type>'
                           f'<metronome><beat-unit>quarter</beat-unit>'
                           f'<per-minute>{int(round(tempo))}</per-minute></metronome>'
                           '</direction-type>'
                           f'<sound tempo="{int(round(tempo))}"/></direction>')
            if p["kind"] == "drums":
                render = _render_drum
            elif p["kind"] == "tab":
                def render(n, d, chord, flats, _p=p):
                    return _render_pitched(n, d, chord, flats, tab=True)
            else:
                def render(n, d, chord, flats):
                    return _render_pitched(n, d, chord, flats, tab=False)
            harm = harmony_by_measure.get(m, []) if pi == 0 else []
            out.append(_measure_body(events, m * m_len, m_len, render, flats, harm))
            out.append('</measure>')
        out.append('</part>')
    out.append('</score-partwise>')
    return "\n".join(out)


# ------------------------------------------------------------------ MIDI
def _vlq(n):
    n = int(n)
    buf = bytearray([n & 0x7F])
    n >>= 7
    while n:
        buf.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(buf)


def _chunk(tag, data):
    return tag + struct.pack(">I", len(data)) + data


def build_midi(tracks, tempo=120.0, beats_per_bar=4, tpq=480, title="Untitled"):
    """tracks: [{"name","program","channel","notes":[{"b","d","midi","vel"}]}] -> bytes"""
    header = struct.pack(">HHH", 1, len(tracks) + 1, tpq)
    chunks = [_chunk(b"MThd", header)]

    # テンポ・拍子トラック
    mpqn = int(round(60_000_000 / max(tempo, 1e-6)))
    meta = bytearray()
    meta += _vlq(0) + b"\xFF\x03" + _vlq(len(title.encode())) + title.encode()
    meta += _vlq(0) + b"\xFF\x51\x03" + mpqn.to_bytes(3, "big")
    meta += _vlq(0) + b"\xFF\x58\x04" + bytes([beats_per_bar, 2, 24, 8])
    meta += _vlq(0) + b"\xFF\x2F\x00"
    chunks.append(_chunk(b"MTrk", bytes(meta)))

    for tr in tracks:
        ch = int(tr.get("channel", 0)) & 0x0F
        evs = []
        for n in tr["notes"]:
            on = int(round(n["b"] * tpq))
            off = on + max(1, int(round(n.get("d", 0.25) * tpq)))
            midi = int(n.get("midi", DRUM_MIDI.get(n.get("inst", "snare"), 38)))
            vel = int(min(127, max(1, n.get("vel", 90))))
            evs.append((on, 1, 0x90 | ch, midi, vel))
            evs.append((off, 0, 0x80 | ch, midi, 0))
        evs.sort(key=lambda e: (e[0], e[1]))

        data = bytearray()
        name = str(tr.get("name", "Track")).encode("utf-8")[:120]
        data += _vlq(0) + b"\xFF\x03" + _vlq(len(name)) + name
        if ch != 9:
            data += _vlq(0) + bytes([0xC0 | ch, int(tr.get("program", 0)) & 0x7F])
        prev = 0
        for t, _, status, a, b in evs:
            data += _vlq(max(0, t - prev)) + bytes([status, a & 0x7F, b & 0x7F])
            prev = t
        data += _vlq(0) + b"\xFF\x2F\x00"
        chunks.append(_chunk(b"MTrk", bytes(data)))
    return b"".join(chunks)


# ------------------------------------------------------------------ MIDI読み込み
def read_midi(path):
    """MIDIファイルを読んで [{start, end, midi, vel, channel}] を返す (秒単位)

    ADTOF や外部の採譜スクリプトの出力を取り込むために使う。
    pretty_midi に依存しないよう自前で解析する。
    """
    with open(str(path), "rb") as f:
        data = f.read()
    if data[:4] != b"MThd":
        raise ValueError("MIDIファイルではありません")
    hdr_len = int.from_bytes(data[4:8], "big")
    _, ntrks, division = struct.unpack(">HHH", data[8:14])
    pos = 8 + hdr_len
    if division & 0x8000:  # SMPTEタイムコードは非対応
        raise ValueError("SMPTE形式のMIDIには対応していません")
    tpq = division or 480

    tempos = [(0, 500000)]   # (tick, マイクロ秒/4分音符)
    raw_notes = []           # (tick_on, tick_off, midi, vel, ch)

    for _ in range(ntrks):
        if pos + 8 > len(data) or data[pos:pos + 4] != b"MTrk":
            break
        length = int.from_bytes(data[pos + 4:pos + 8], "big")
        p, end = pos + 8, pos + 8 + length
        pos = end
        tick, status, active = 0, 0, {}
        while p < end:
            # デルタタイム (可変長)
            dt = 0
            while p < end:
                b = data[p]
                p += 1
                dt = (dt << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            tick += dt
            if p >= end:
                break
            b = data[p]
            if b & 0x80:
                status = b
                p += 1
            if status == 0xFF:                      # メタイベント
                mtype = data[p]
                p += 1
                ln = 0
                while p < end:
                    c = data[p]
                    p += 1
                    ln = (ln << 7) | (c & 0x7F)
                    if not (c & 0x80):
                        break
                if mtype == 0x51 and ln == 3:
                    tempos.append((tick, int.from_bytes(data[p:p + 3], "big")))
                p += ln
            elif status in (0xF0, 0xF7):            # SysEx
                ln = 0
                while p < end:
                    c = data[p]
                    p += 1
                    ln = (ln << 7) | (c & 0x7F)
                    if not (c & 0x80):
                        break
                p += ln
            else:
                kind, ch = status & 0xF0, status & 0x0F
                nbytes = 1 if kind in (0xC0, 0xD0) else 2
                args = data[p:p + nbytes]
                p += nbytes
                if kind == 0x90 and len(args) == 2 and args[1] > 0:
                    active.setdefault((ch, args[0]), []).append((tick, args[1]))
                elif (kind == 0x80 or (kind == 0x90 and len(args) == 2)) and len(args) == 2:
                    stack = active.get((ch, args[0]))
                    if stack:
                        t0, v = stack.pop(0)
                        raw_notes.append((t0, tick, args[0], v, ch))
        for (ch, note), stack in active.items():    # 閉じ忘れは短い音として拾う
            for t0, v in stack:
                raw_notes.append((t0, t0 + tpq // 8, note, v, ch))

    # ティック -> 秒 (テンポ変化に対応)
    tempos = sorted(set(tempos))

    def to_sec(t):
        sec, last_tick, mpqn = 0.0, 0, tempos[0][1]
        for tt, mm in tempos:
            if tt >= t:
                break
            sec += (tt - last_tick) / tpq * (mpqn / 1e6)
            last_tick, mpqn = tt, mm
        return sec + (t - last_tick) / tpq * (mpqn / 1e6)

    notes = []
    for t0, t1, note, vel, ch in sorted(raw_notes):
        s, e = to_sec(t0), to_sec(t1)
        notes.append({"start": s, "end": max(e, s + 0.02), "midi": int(note),
                      "vel": int(vel), "channel": int(ch),
                      "amp": min(vel / 127.0, 1.0)})
    notes.sort(key=lambda n: (n["start"], n["midi"]))
    return notes


# ------------------------------------------------------------------ テキストTAB
def build_text_tab(notes, tuning, beats_per_bar=4, bars_per_line=4, div=4):
    """e|--3--5--| 形式のASCII TAB (おまけ)"""
    names = []
    for m in tuning:
        step, alter = _SHARP_NAMES[m % 12]
        names.append((step + ("#" if alter else "")).ljust(2))
    cell = 1  # 1グリッド = 1カラム(数字は最大2桁なので実際は3文字幅)
    steps_per_bar = beats_per_bar * div
    total = max([int(round((n["b"] + n["d"]) * div)) for n in notes] + [steps_per_bar])
    n_bars = -(-total // steps_per_bar)
    grid = {}
    for n in notes:
        col = int(round(n["b"] * div))
        grid.setdefault(col, {})[n["string"] - 1] = str(n["fret"])

    lines = []
    for start_bar in range(0, n_bars, bars_per_line):
        rows = ["" for _ in tuning]
        for bar in range(start_bar, min(start_bar + bars_per_line, n_bars)):
            for s in range(len(tuning)):
                rows[s] += "|"
            for step in range(steps_per_bar):
                col = bar * steps_per_bar + step
                cellmap = grid.get(col, {})
                for s in range(len(tuning)):
                    v = cellmap.get(s)
                    rows[s] += (v.rjust(2, "-") if v else "--") + "-"
        for s in range(len(tuning)):
            rows[s] += "|"
            lines.append(names[s] + rows[s])
        lines.append("")
    return "\n".join(lines)
