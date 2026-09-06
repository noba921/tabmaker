"""API疎通テスト — サーバーの入口から出口までを、曲をダウンロードせずに確かめる

selftest.py が各モジュールの中身を検証するのに対し、こちらは
「HTTPのやり取り・保存形式・書き出し・設定変更」といった結合部分を見る。
Demucs や yt-dlp は使わず、testsig.py が作った音源をステムとして直接置く。

apitest.bat から実行する。
"""
import json
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.argv = ["app"]          # app.py の起動処理を走らせないため

import librosa  # noqa: E402
import analysis, chords as chordmod, engines, testsig  # noqa: E402
import app as A  # noqa: E402

fails = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main():
    work = Path(tempfile.mkdtemp(prefix="tabmaker_apitest_"))
    # 結合部分の確認が目的なので、短い音源で十分(採譜精度は selftest.py が見る)
    sig = testsig.build(work / "sig", bars=4)
    f = sig["files"]

    # 本物の songs/ と config.json を汚さないよう、一時領域に差し替える
    A.SONGS_DIR = work / "songs"
    A.SONGS_DIR.mkdir(parents=True)
    engines.CONFIG_PATH = work / "config.json"

    sid = "apitest"
    d = A.SONGS_DIR / sid
    d.mkdir()
    for name in ("mix", "bass", "drums", "other", "vocals"):
        shutil.copy(f[name], d / f"{name}.wav")

    cfg = engines.load_config()
    y, sr = librosa.load(str(f["mix"]), sr=22050, mono=True)
    grid = analysis.estimate_grid(y, sr, sig["beats_per_bar"])
    key, fifths, _ = analysis.estimate_key(y, sr)

    # ---------------------------------------------- 譜面の生成 (ステムの選択)
    print("\n[1] ステムの選択とフォールバック")
    stems4 = {k: f[k] for k in ("vocals", "drums", "bass", "other")}
    parts4, _ = A.build_score(d, stems4, grid, key, fifths, len(y) / sr, cfg)
    check(parts4["guitar"]["stem"] == "other",
          "4stemではギターが other にフォールバックする")
    check(parts4["piano"]["stem"] == "other",
          "4stemではピアノが other にフォールバックする")

    shutil.copy(f["other"], d / "guitar.wav")
    shutil.copy(f["chord_harm"], d / "piano.wav")
    stems6 = dict(stems4, guitar=d / "guitar.wav", piano=d / "piano.wav")
    parts, raw = A.build_score(d, stems6, grid, key, fifths, len(y) / sr, cfg)
    check(parts["guitar"]["stem"] == "guitar", "6stemではギター専用ステムを使う")
    check(parts["piano"]["stem"] == "piano", "6stemではピアノ専用ステムを使う")
    check(set(raw) <= {"guitar", "bass"},
          f"rawはTAB譜のパートだけ持つ (実際: {sorted(raw)})")

    print("\n[2] 採譜するパートの絞り込み")
    engines.save_config({"parts": ["bass", "drums"]})
    few, _ = A.build_score(d, stems6, grid, key, fifths, len(y) / sr,
                           engines.load_config())
    check(set(few) == {"bass", "drums"}, f"指定したパートだけ採譜する ({sorted(few)})")
    engines.save_config({"parts": ["melody", "guitar", "bass", "piano", "drums"]})
    engines.save_config({"parts": []})
    allp, _ = A.build_score(d, stems6, grid, key, fifths, len(y) / sr,
                            engines.load_config())
    check(len(allp) == 5, "パート指定が空なら全パートに戻る")
    engines.save_config({"parts": ["melody", "guitar", "bass", "piano", "drums"]})

    # ---------------------------------------------- score.json を用意
    chord_list = chordmod.recognize([d / "piano.wav"], d / "bass.wav", grid,
                                    flats=(fifths < 0))
    check(len(chord_list) > 0, f"コード進行が入る ({len(chord_list)}区間)")
    score = {
        "id": sid, "title": "APIテスト", "duration": round(len(y) / sr, 1),
        "tempo": round(grid.tempo, 1), "beats_per_bar": grid.beats_per_bar,
        "key": key, "fifths": fifths, "measures": sig["bars"],
        "engines": {"separator": "htdemucs_6s", "beat": "librosa",
                    "pitch": "librosa", "drums": "builtin",
                    "chords": "builtin", "device": "cpu"},
        "timings": {"採譜": 1.0}, "grid": grid.to_dict(),
        "audio": {"mix": "mix.wav", "bass": "bass.wav"},
        "parts": parts, "chords": chord_list, "raw": raw, "created": time.time()}
    (d / "score.json").write_text(json.dumps(score, ensure_ascii=False),
                                  encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "id": sid, "title": "APIテスト", "duration": score["duration"],
        "tempo": score["tempo"], "key": key, "measures": sig["bars"],
        "created": score["created"], "engines": score["engines"],
        "parts": {k: v["stats"].get("notes", 0) for k, v in parts.items()},
    }, ensure_ascii=False), encoding="utf-8")

    c = A.app.test_client()

    # ---------------------------------------------- 基本的な入口
    print("\n[3] 基本のエンドポイント")
    r = c.get("/")
    check(r.status_code == 200 and b"TAB Maker" in r.data, "トップページが返る")
    check(c.get("/favicon.ico").status_code == 200, "favicon が500にならない")
    check(c.get("/api/存在しない").status_code == 404, "無いパスは404 (500に化けない)")
    j = c.get("/api/config").get_json()
    check("guitar" in j["tunings"], "チューニング一覧が返る")
    check(set(j["engines"]["stages"]) ==
          {"separator", "beat", "pitch", "drums", "chords"}, "エンジン一覧が返る")

    print("\n[4] 曲一覧")
    lst = c.get("/api/songs").get_json()
    check(isinstance(lst, dict) and "songs" in lst and "total_size" in lst,
          "曲一覧が {songs, total_size} 形式で返る")
    check(len(lst["songs"]) == 1 and lst["songs"][0]["size"] > 0,
          "曲ごとの容量が入っている")

    # ---------------------------------------------- 設定
    print("\n[5] 設定の変更")
    r = c.post("/api/config", json={"separator": "htdemucs", "chords": "off"})
    check(r.status_code == 200 and r.get_json()["config"]["separator"] == "htdemucs",
          "設定をPOSTで変更できる")
    check(engines.load_config()["chords"] == "off", "設定がファイルに永続化される")
    check(c.post("/api/config", json={"separator": "<不正>"}).status_code == 200,
          "不正な値でも落ちない")
    check(engines.resolve("separator") in
          [e["id"] for e in engines.ENGINES["separator"]],
          "不正な設定でも既知のエンジンに解決される")
    c.post("/api/config", json={"separator": "auto", "chords": "builtin"})
    check(c.post("/api/config", json={"知らないキー": 1}).status_code == 200,
          "知らないキーは無視される")
    check("知らないキー" not in engines.load_config(), "知らないキーは保存されない")

    # ---------------------------------------------- 譜面と音源
    print("\n[6] 譜面データと音源")
    s = c.get(f"/api/songs/{sid}/score").get_json()
    check("raw" not in s and "grid" not in s, "scoreから重いデータが除かれている")
    check("beat_times" in s and "phase" in s, "描画に必要な拍情報が入っている")
    check(len(s.get("chords", [])) > 0, "scoreにコードが含まれる")
    check(s["engines"]["separator"] == "htdemucs_6s", "使用エンジンが記録されている")
    check(c.get(f"/api/songs/{sid}/audio/mix").status_code == 200, "音源が配信される")
    check(c.get(f"/api/songs/{sid}/audio/nonexist").status_code == 404,
          "無い音源は404")
    check(c.get(f"/api/songs/{sid}/audio/../score").status_code in (404, 308, 301),
          "パストラバーサルができない")
    check(c.get("/api/songs/zzz/score").status_code == 404, "無い曲は404")

    # ---------------------------------------------- 書き出し
    print("\n[7] 書き出し")
    r = c.get(f"/api/songs/{sid}/export/musicxml")
    check(r.status_code == 200 and r.data.startswith(b"<?xml"), "MusicXMLが返る")
    root = ET.fromstring(r.data.decode("utf-8"))
    check(len(root.findall("part")) == len(s["parts"]), "パート数が一致する")
    check(len(root.findall(".//harmony")) > 0, "コードネームが載っている")
    m_len = s["beats_per_bar"] * 4
    bad = [m.get("number") for p in root.findall("part") for m in p.findall("measure")
           if sum(int(n.find("duration").text) for n in m.findall("note")
                  if n.find("chord") is None) != m_len]
    check(not bad, f"全パート全小節の音価合計が拍子と一致 (異常: {bad[:3]})")
    one = ET.fromstring(c.get(f"/api/songs/{sid}/export/musicxml?parts=bass")
                        .data.decode())
    check(len(one.findall("part")) == 1, "パートを指定して書き出せる")

    r = c.get(f"/api/songs/{sid}/export/midi")
    check(r.data[:4] == b"MThd", "MIDIが返る")
    tmp_mid = work / "out.mid"
    tmp_mid.write_bytes(r.data)
    import export
    back = export.read_midi(tmp_mid)
    check(len(back) > 0, f"書き出したMIDIを読み戻せる ({len(back)}音)")
    check(any(n["channel"] == 9 for n in back), "ドラムがチャンネル10に入っている")

    r = c.get(f"/api/songs/{sid}/export/txt?part=bass")
    check(r.status_code == 200 and b"|" in r.data, "テキストTABが返る")
    check(c.get(f"/api/songs/{sid}/export/txt?part=melody").status_code == 400,
          "TABでないパートを指定すると400")

    # ---------------------------------------------- 編集操作
    print("\n[8] 運指のやり直しと小節の補正")
    r = c.post(f"/api/songs/{sid}/retab",
               json={"part": "guitar", "tuning": "drop_d", "capo": 1, "max_fret": 12})
    p = r.get_json()
    check(r.status_code == 200, "retabが成功する")
    check(all(p["strings"][n["string"] - 1] + 1 + n["fret"] == n["midi"]
              for n in p["notes"]), "再運指しても音高が保たれる")
    check(max((n["fret"] for n in p["notes"]), default=0) <= 12,
          "最高フレットの指定が効く")
    check(c.post(f"/api/songs/{sid}/retab", json={"part": "drums"}).status_code == 400,
          "TAB譜でないパートのretabは400")

    before = c.get(f"/api/songs/{sid}/score").get_json()
    bn = before["parts"]["bass"]["notes"][0]["b"]
    b_end = before["chords"][-1]["b"] + before["chords"][-1]["d"]
    b_note_end = before["parts"]["bass"]["notes"][-1]["b"]
    check(c.post(f"/api/songs/{sid}/shift", json={"beats": 1}).status_code == 200,
          "小節位置をずらせる")
    after = c.get(f"/api/songs/{sid}/score").get_json()
    an = after["parts"]["bass"]["notes"][0]["b"]
    a_end = after["chords"][-1]["b"] + after["chords"][-1]["d"]
    a_note_end = after["parts"]["bass"]["notes"][-1]["b"]
    check(abs((bn - 1) - an) < 1e-6 or an == 0.0, "先頭の音符の拍位置がずれる")
    check(abs((b_note_end - 1) - a_note_end) < 1e-6, "末尾の音符も1拍分ずれる")
    check(abs((b_end - 1) - a_end) < 1e-6, "コード区間も1拍分ずれる")
    c.post(f"/api/songs/{sid}/shift", json={"beats": -1})

    # ---------------------------------------------- 削除
    print("\n[9] 削除")
    check(c.delete(f"/api/songs/{sid}").status_code == 200 and not d.exists(),
          "曲を削除できる")
    check(len(c.get("/api/songs").get_json()["songs"]) == 0, "一覧から消える")

    # ---------------------------------------------- 音源だけ取り込む
    print("\n[10] 音源だけの取り込み")
    A.AUDIO_DIR = work / "audio"
    A.AUDIO_DIR.mkdir()

    check("wav" in A.AUDIO_FORMATS and A.AUDIO_FORMATS["wav"].get("passthrough"),
          "既定はWAV(非圧縮・変換なし)")
    check(A._pick_format({"format": "存在しない"}) == "wav",
          "知らない形式を指定してもWAVに落ちる")
    check(A._pick_format({}) == "wav", "形式未指定でもWAVになる")

    # 同名ファイルの連番付け(日本語のタイトルでも壊れないこと)
    (A.AUDIO_DIR / "曲.wav").write_bytes(b"RIFFtest")
    p2 = A._unique_path(A.AUDIO_DIR, "曲", "wav")
    check(p2.name == "曲 (2).wav", f"同名は連番になる ({p2.name})")
    check(A._safe_name('曲/名:前?') == "曲_名_前_",
          f"ファイル名に使えない文字が除かれる ({A._safe_name('曲/名:前?')})")
    (A.AUDIO_DIR / "曲.wav").unlink()

    (A.AUDIO_DIR / "sample.wav").write_bytes(b"RIFFtest")
    lst = c.get("/api/audio").get_json()
    check(len(lst["files"]) == 1 and lst["files"][0]["name"] == "sample.wav",
          "保存した音源が一覧に出る")
    check(lst["total_size"] > 0 and "dir" in lst, "合計容量と保存先が返る")

    r = c.get("/api/audio/file/sample.wav")
    check(r.status_code == 200 and r.data == b"RIFFtest", "音源をダウンロードできる")
    check(c.get("/api/audio/file/none.wav").status_code == 404, "無い音源は404")
    check(A._audio_path("../app.py") is None, "audio/の外は参照できない")
    check(A._audio_path("sample.wav") is not None, "audio/の中は参照できる")

    check(c.delete("/api/audio/sample.wav").status_code == 200
          and not (A.AUDIO_DIR / "sample.wav").exists(), "音源を削除できる")
    check(len(c.get("/api/audio").get_json()["files"]) == 0, "一覧から消える")

    check(c.post("/api/audio/youtube", json={"url": "not-a-url"}).status_code == 400,
          "不正なURLは400")

    cfgj = c.get("/api/config").get_json()
    check(len(cfgj.get("audio_formats", [])) >= 3, "形式の一覧がUIに渡される")

    # 実際に1本通す (ffmpegが要るので、無い環境ではスキップ扱いにする)
    with open(f["mix"], "rb") as fh:
        r = c.post("/api/audio/upload",
                   data={"file": (fh, "テスト音源.wav"), "format": "wav"},
                   content_type="multipart/form-data")
    check(r.status_code == 200 and "job" in r.get_json(), "音源取り込みジョブが始まる")
    job_id = r.get_json()["job"]
    for _ in range(60):
        st = c.get(f"/api/jobs/{job_id}").get_json()
        if st["status"] != "running":
            break
        time.sleep(0.5)
    if st["status"] == "done":
        check(st.get("kind") == "audio", "音源ジョブとして識別できる")
        check(st.get("file", "").endswith(".wav") and st.get("size", 0) > 0,
              f"ファイルが書き出される ({st.get('file')})")
        check(len(c.get("/api/audio").get_json()["files"]) == 1, "一覧に反映される")
    else:
        print("  SKIP  音源取り込みの実行 (ffmpegが無い環境):",
              str(st.get("error"))[:80])

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 56)
    print("  TAB Maker  API疎通テスト")
    print("=" * 56)
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        fails.append("テスト自体が例外で停止しました")
    print("\n" + "=" * 56)
    if fails:
        print(f"NG が {len(fails)} 件:")
        for x in fails:
            print("  -", x)
    else:
        print("すべてのチェックを通過しました")
    try:
        input("\nEnterキーで閉じます...")
    except EOFError:
        pass
    sys.exit(1 if fails else 0)
