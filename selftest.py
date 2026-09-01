"""自己診断 — 合成した音源で「採譜 → TAB化 → 楽譜出力」まで一通り動くか確かめる

selftest.bat から実行する。曲をダウンロードせずに環境の健全性を確認できる。
テスト用の音源は testsig.py が作る(正解が分かっているので精度を機械的に測れる)。
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis, chords as chordmod, engines, testsig, transcribe, tabgen, export  # noqa: E402
from testsig import BARS, BPB, SPB, TEMPO  # noqa: E402

OUT = Path(__file__).resolve().parent / "tmp" / "selftest"
_sig = testsig.build(OUT)
expected_bass = _sig["expected_bass"]
expected_drums = _sig["expected_drums"]
expected_chords = _sig["expected_chords"]

fails = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ================================================== 1. テンポと拍
print("\n[1] テンポ・拍グリッド")
import librosa
y22, sr22 = librosa.load(str(OUT / "mix.wav"), sr=22050, mono=True)
grid = analysis.estimate_grid(y22, sr22, beats_per_bar=BPB)
print(f"       推定テンポ = {grid.tempo:.1f} BPM (正解 {TEMPO})  位相={grid.phase}")
check(abs(grid.tempo - TEMPO) < 3, "テンポが正しく取れている")
b0 = float(grid.beat_to_time(0))
print(f"       1小節目の頭 = {b0:.3f}s")
check(abs(b0 % (BPB * SPB)) < 0.12 or abs(b0 % (BPB * SPB) - BPB * SPB) < 0.12,
      "小節頭が実際の小節線に合っている")
key, fifths, _ = analysis.estimate_key(y22, sr22)
print(f"       推定キー = {key} (fifths={fifths})")

# ================================================== 2. ベースの採譜
print("\n[2] ベース採譜 (軽量モード)")
bn = transcribe.transcribe_pitched(OUT / "bass.wav", "bass", engine="librosa")
qn = analysis.quantize_notes(bn, grid, div=4)
print(f"       検出 {len(qn)} 音 / 正解 {len(expected_bass)} 音")
check(abs(len(qn) - len(expected_bass)) <= 4, "音符の数がだいたい合っている")
# 拍位置ごとに音高を突き合わせる
hit = 0
for beat, midi in expected_bass:
    near = [n for n in qn if abs(n["b"] - beat) <= 0.5]
    if any(n["midi"] == midi for n in near):
        hit += 1
print(f"       音高一致 {hit}/{len(expected_bass)}")
check(hit / len(expected_bass) > 0.8, "ベースの音高が8割以上一致")
check(all(abs(n["b"] * 4 - round(n["b"] * 4)) < 1e-6 for n in qn), "16分グリッドに量子化されている")

# ================================================== 3. TAB化
print("\n[3] TAB譜への変換")
tab, strings = tabgen.make_tab(qn, "bass", "standard", 0)
bad = [n for n in tab if strings[n["string"] - 1] + n["fret"] != n["midi"]]
check(not bad, "弦とフレットから元の音高が再現できる")
check(all(0 <= n["fret"] <= 20 for n in tab), "フレット番号が範囲内")
frets = [n["fret"] for n in tab]
print(f"       使用フレット {min(frets)}〜{max(frets)} / 音符 {len(tab)}")
check(max(frets) - min(frets) <= 7, "ポジション移動が抑えられている")

gtab, gstr = tabgen.make_tab(
    analysis.quantize_notes(
        transcribe.transcribe_pitched(OUT / "other.wav", "guitar", engine="librosa"),
        grid), "guitar", "standard", 0)
bad = [n for n in gtab if gstr[n["string"] - 1] + n["fret"] != n["midi"]]
check(not bad, "ギターも弦/フレットが正しい")
# 同時発音で弦が重複していないこと
by_beat = {}
for n in gtab:
    by_beat.setdefault(n["b"], []).append(n["string"])
check(all(len(v) == len(set(v)) for v in by_beat.values()), "同時に同じ弦を使っていない")

# カポとチューニング変更
ctab, cstr = tabgen.make_tab(qn, "bass", "drop_d", 2)
bad = [n for n in ctab if cstr[n["string"] - 1] + 2 + n["fret"] != n["midi"]]
check(not bad, "カポ/変則チューニングでも整合している")

# ================================================== 4. ドラム
print("\n[4] ドラム採譜")
dev = transcribe.transcribe_drums(OUT / "drums.wav", engine="builtin")
dbeats = grid.time_to_beat([e["start"] for e in dev]) if dev else []
det = [(round(float(analysis.quantize(b, 4)) * 2) / 2, e["inst"])
       for e, b in zip(dev, dbeats)]
print(f"       打点 {len(dev)} 個")
kick_hit = sum(1 for beat, inst in expected_drums if inst == "kick" and
               any(abs(b - beat) <= 0.3 and i == "kick" for b, i in det))
snare_hit = sum(1 for beat, inst in expected_drums if inst == "snare" and
                any(abs(b - beat) <= 0.3 and i == "snare" for b, i in det))
n_kick = sum(1 for _, i in expected_drums if i == "kick")
n_snare = sum(1 for _, i in expected_drums if i == "snare")
print(f"       キック {kick_hit}/{n_kick}  スネア {snare_hit}/{n_snare}")
check(kick_hit / n_kick > 0.7, "キックの検出率 > 70%")
check(snare_hit / n_snare > 0.7, "スネアの検出率 > 70%")
hats = sum(1 for _, i in det if i in ("hihat", "open_hihat"))
print(f"       ハイハット {hats} 個 (正解 {BARS*BPB*2})")
check(hats > BARS * BPB, "ハイハットが検出できている")

drum_notes = [{"b": max(round(float(analysis.quantize(b, 4)), 4), 0.0),
               "d": 0.25, "inst": e["inst"], "vel": e["vel"]}
              for e, b in zip(dev, dbeats)]

# ================================================== 5. 書き出し
print("\n[5] MusicXML / MIDI の書き出し")
parts = [
    {"id": "P1", "name": "Bass", "kind": "tab", "tuning": strings, "capo": 0,
     "program": 34, "channel": 2, "notes": tab},
    {"id": "P2", "name": "Guitar", "kind": "tab", "tuning": gstr, "capo": 0,
     "program": 30, "channel": 1, "notes": gtab},
    {"id": "P3", "name": "Melody", "kind": "staff", "clef": ("G", 2),
     "program": 74, "channel": 3,
     "notes": analysis.quantize_notes(
         transcribe.transcribe_pitched(OUT / "vocals.wav", "melody", engine="librosa"), grid)},
    {"id": "P4", "name": "Drums", "kind": "drums", "notes": drum_notes},
]
xml = export.build_musicxml(parts, tempo=grid.tempo, beats_per_bar=BPB,
                            fifths=fifths, title="テスト曲 <&>")
(OUT / "test.musicxml").write_text(xml, encoding="utf-8")

import xml.etree.ElementTree as ET
root = ET.fromstring(xml)
check(root.tag == "score-partwise", "MusicXMLがパースできる")
xparts = root.findall("part")
check(len(xparts) == 4, f"パート数が4 (実際 {len(xparts)})")
for xp in xparts:
    ms = xp.findall("measure")
    check(len(ms) >= BARS - 1, f"{xp.get('id')} の小節数 {len(ms)}")
    # 各小節の音価の合計が拍子と一致するか (和音は数えない)
    bad_m = []
    for m in ms:
        total = 0
        for n in m.findall("note"):
            if n.find("chord") is not None:
                continue
            total += int(n.find("duration").text)
        if total != BPB * export.DIVISIONS:
            bad_m.append((m.get("number"), total))
    check(not bad_m, f"{xp.get('id')} 全小節の音価合計が {BPB*export.DIVISIONS} "
                     + (f"(異常: {bad_m[:3]})" if bad_m else ""))
# TABの弦/フレット情報
tech = xparts[0].findall(".//technical")
check(len(tech) == len(tab), f"TABに弦/フレットが全音符分ある ({len(tech)}/{len(tab)})")
check(xparts[0].find(".//clef/sign").text == "TAB", "TABクレフが出ている")
check(xparts[3].find(".//clef/sign").text == "percussion", "打楽器クレフが出ている")
check(len(xparts[3].findall(".//unpitched")) > 0, "ドラムがunpitchedで書かれている")

mid = export.build_midi(
    [{"name": "Bass", "program": 33, "channel": 1, "notes": tab},
     {"name": "Drums", "program": 0, "channel": 9, "notes": drum_notes}],
    tempo=grid.tempo, beats_per_bar=BPB, title="テスト曲")
(OUT / "test.mid").write_bytes(mid)
check(mid[:4] == b"MThd", "MIDIヘッダが正しい")
n_trk = mid.count(b"MTrk")
check(n_trk == 3, f"MIDIトラック数 {n_trk} (テンポ+2パート)")
# 中身をざっくり検証
try:
    import struct
    ntracks = struct.unpack(">H", mid[10:12])[0]
    check(ntracks == 3, "ヘッダのトラック数が一致")
except Exception as e:
    check(False, f"MIDI解析に失敗: {e}")

txt = export.build_text_tab(tab, strings, BPB)
(OUT / "test_tab.txt").write_text(txt, encoding="utf-8")
check(txt.count("|") > BARS, "テキストTABに小節線がある")
print("\n---- テキストTAB (先頭) ----")
print("\n".join(txt.splitlines()[:6]))

# ================================================== 6. コード認識
print("\n[6] コード進行の認識")
cy, _ = librosa.load(str(OUT / "chord_harm.wav"), sr=22050, mono=True)
cgrid = analysis.estimate_grid(cy + librosa.load(str(OUT / "drums.wav"), sr=22050,
                                                 mono=True)[0][:len(cy)], 22050, BPB)
found = chordmod.recognize([OUT / "chord_harm.wav"], OUT / "chord_bass.wav", cgrid)
print(f"       検出 {len(found)} 区間: " +
      " ".join(f"{c['name']}@{c['b']:.0f}" for c in found[:8]))
check(len(found) >= 4, "コード区間が取れている")
hit = 0
for bar, name in enumerate(expected_chords):
    b0 = bar * BPB
    at = [c for c in found if c["b"] <= b0 + 0.5 and c["b"] + c["d"] > b0 + 0.5]
    if at and at[0]["name"] == name:
        hit += 1
print(f"       コード名一致 {hit}/{len(expected_chords)}")
check(hit / len(expected_chords) >= 0.75, "コード名が3/4以上一致")
check(all(c["b"] >= 0 and c["d"] > 0 for c in found), "コード区間の拍位置が妥当")

# ================================================== 7. MIDIの読み書き往復
print("\n[7] MIDIの読み書き")
ref_notes = [{"b": 0.0, "d": 1.0, "midi": 60, "vel": 100},
             {"b": 1.0, "d": 0.5, "midi": 64, "vel": 80},
             {"b": 2.5, "d": 1.5, "midi": 67, "vel": 120}]
mid2 = export.build_midi([{"name": "T", "program": 0, "channel": 0,
                           "notes": ref_notes}], tempo=120.0, beats_per_bar=4)
(OUT / "roundtrip.mid").write_bytes(mid2)
back = export.read_midi(OUT / "roundtrip.mid")
check(len(back) == len(ref_notes), f"音符数が一致 ({len(back)}/{len(ref_notes)})")
ok_pitch = all(b["midi"] == r["midi"] for b, r in zip(back, ref_notes))
ok_vel = all(b["vel"] == r["vel"] for b, r in zip(back, ref_notes))
ok_time = all(abs(b["start"] - r["b"] * 0.5) < 0.01 for b, r in zip(back, ref_notes))
ok_dur = all(abs((b["end"] - b["start"]) - r["d"] * 0.5) < 0.02
             for b, r in zip(back, ref_notes))
check(ok_pitch, "音高が往復しても一致")
check(ok_vel, "ベロシティが往復しても一致")
check(ok_time, "発音時刻が往復しても一致")
check(ok_dur, "音の長さが往復しても一致")

# ================================================== 8. ドラムの強弱
print("\n[8] ドラムのベロシティ推定")
dyn = transcribe.transcribe_drums(OUT / "dyn.wav", engine="builtin")  # 1拍目だけ強打
strong = [e["vel"] for e in dyn if abs(e["start"] % (BPB * SPB)) < 0.08]
weak = [e["vel"] for e in dyn if abs(e["start"] % (BPB * SPB)) > 0.3]
print(f"       強打 {np.mean(strong) if strong else 0:.0f} / "
      f"弱打 {np.mean(weak) if weak else 0:.0f}")
check(bool(strong) and bool(weak) and np.mean(strong) > np.mean(weak) + 10,
      "強く叩いた打点のベロシティが高い")
check(len({e["vel"] for e in dyn}) > 1, "ベロシティが一定値になっていない")

# ================================================== 9. MusicXMLのコードネーム
print("\n[9] MusicXML にコードネームが載る")
xml2 = export.build_musicxml(
    [{"id": "P1", "name": "Guitar", "kind": "tab", "tuning": gstr, "capo": 0,
      "program": 30, "channel": 1, "notes": gtab}],
    tempo=grid.tempo, beats_per_bar=BPB, fifths=0, title="コードつき",
    chords=[{"b": 0.0, "d": 4.0, "name": "C", "root": 0, "quality": "maj",
             "kind": "major"},
            {"b": 4.0, "d": 4.0, "name": "Am", "root": 9, "quality": "min",
             "kind": "minor"}])
r2 = ET.fromstring(xml2)
harms = r2.findall(".//harmony")
check(len(harms) == 2, f"<harmony> が2つ書かれている ({len(harms)})")
if len(harms) == 2:
    check(harms[0].find("root/root-step").text == "C"
          and harms[1].find("root/root-step").text == "A", "コードのルートが正しい")
    check(harms[1].find("kind").text == "minor", "マイナーとして書かれている")
bad_m = []
for m in r2.findall("part/measure"):
    total = sum(int(n.find("duration").text) for n in m.findall("note")
                if n.find("chord") is None)
    if total != BPB * export.DIVISIONS:
        bad_m.append(m.get("number"))
check(not bad_m, "コードを入れても各小節の音価合計が崩れない")

# ================================================== 10. エンジン切り替え
print("\n[10] エンジンの切り替え")
import tempfile as _tf
_bak = engines.CONFIG_PATH
engines.CONFIG_PATH = Path(_tf.mkdtemp()) / "config.json"
try:
    d = engines.describe()
    check(set(d["stages"]) == {"separator", "beat", "pitch", "drums", "chords"},
          "5つの処理段が登録されている")
    for stage, s in d["stages"].items():
        avail = [o["id"] for o in s["options"] if o["available"]]
        print(f"       {stage}: 使用={s['active']} / 選択可={avail}")
        tier = {o["id"]: o["tier"] for o in s["options"]}
        if avail:
            check(s["active"] in avail, f"{stage} は使えるエンジンが選ばれている")
        else:
            # 何も入っていない環境でも、GPU前提のエンジンを既定にしてはいけない
            check(tier.get(s["active"]) != "gpu",
              f"{stage} は未インストールでもGPU前提に落ちない")
    engines.save_config({"separator": "htdemucs", "chords": "off"})
    cfg2 = engines.load_config()
    check(cfg2["separator"] == "htdemucs" and cfg2["chords"] == "off", "設定が保存される")
    check(engines.resolve("chords", cfg2) == "off", "設定した値が使われる")
    engines.save_config({"separator": "存在しないエンジン"})
    check(engines.resolve("separator") in
          [e["id"] for e in engines.ENGINES["separator"]],
          "壊れた設定でも既知のエンジンに落ちる")
    check(isinstance(engines.has_cuda(), bool), "CUDAの検出が例外を出さない")
    check(engines.device({"device": "cpu"}) == "cpu", "デバイス指定が効く")
finally:
    engines.CONFIG_PATH = _bak

print("\n" + "=" * 50)
if fails:
    print(f"NG が {len(fails)} 件:")
    for f in fails:
        print("  -", f)
else:
    print("すべてのチェックを通過しました")
print(f"生成物: {OUT}")
input("\nEnterキーで閉じます...")
sys.exit(1 if fails else 0)
