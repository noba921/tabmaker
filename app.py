"""TAB Maker — YouTube URL から楽器ごとの TAB譜・楽譜を自動生成するローカルアプリ

処理の流れ:
  音源取得 (yt-dlp) → パート分離 (Demucs) → テンポ/拍の解析 (librosa)
  → 音符の推定 (basic-pitch) → 運指の最適化 → 譜面表示 / MusicXML・MIDI 出力

起動: python app.py  →  ブラウザで http://127.0.0.1:8766
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

try:
    import numpy as np
    from flask import Flask, jsonify, request, send_file, send_from_directory, Response
    import analysis
    import chords as chordmod
    import engines
    import export
    import separate
    import tabgen
    import transcribe
except ImportError as e:
    print("=" * 50)
    print("必要なライブラリが見つかりません:", e)
    print("setup.bat を実行(再実行)してください。")
    print("=" * 50)
    input("Enterキーで閉じます...")
    raise SystemExit(1)

BASE_DIR = Path(__file__).resolve().parent
SONGS_DIR = BASE_DIR / "songs"
TMP_DIR = BASE_DIR / "tmp"
SONGS_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

PORT = 8766
ANALYSIS_SR = 22050
PHASES = ["音源を取得", "パートを分離 (AI)", "拍・小節を解析",
          "音符を推定 (採譜)", "譜面を生成"]

# 譜面にするパート
#   stem     … 使いたいステム名
#   fallback … そのステムが無い分離モデルのときに代わりに使うステム
PART_SPEC = [
    ("melody", dict(label="メロディ", kind="staff", clef=("G", 2),
                    program=73, channel=2, stem="vocals")),
    ("guitar", dict(label="ギター", kind="tab", instrument="guitar",
                    program=29, channel=0, stem="guitar", fallback="other")),
    ("bass",   dict(label="ベース", kind="tab", instrument="bass",
                    program=33, channel=1, stem="bass")),
    ("piano",  dict(label="ピアノ", kind="staff", clef=("G", 2),
                    program=0, channel=3, stem="piano", fallback="other")),
    ("drums",  dict(label="ドラム", kind="drums", channel=9, stem="drums")),
]
PART_SPEC_MAP = dict(PART_SPEC)


def pick_stem(spec, stems):
    """そのパートに使うステムを決める (6stemなら専用、4stemなら other に落とす)"""
    for key in (spec["stem"], spec.get("fallback")):
        if key and key in stems and Path(stems[key]).exists():
            return key, stems[key]
    return None, None

app = Flask(__name__, static_folder="static")
jobs = {}


def log(*args):
    print("[tabmaker]", *args, flush=True)


@app.errorhandler(Exception)
def on_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e          # 404などは500に化けさせない
    traceback.print_exc()
    return jsonify({"error": f"サーバー内部エラー: {type(e).__name__}: {e}"}), 500


@app.get("/favicon.ico")
def favicon():
    return Response(b"", mimetype="image/x-icon")


# ---------------------------------------------------------------- ffmpeg
_FFMPEG = None


def ffmpeg_exe():
    """ffmpegを多段フォールバックで探す。見つからなければ自動インストールを試みる"""
    global _FFMPEG
    if _FFMPEG and Path(_FFMPEG).exists():
        return _FFMPEG

    def _try_imageio():
        try:
            import importlib
            import imageio_ffmpeg
            importlib.reload(imageio_ffmpeg)
            p = imageio_ffmpeg.get_ffmpeg_exe()
            if p and Path(p).exists():
                return p
        except Exception as e:
            log("imageio-ffmpeg 利用不可:", e)
        return None

    p = _try_imageio() or shutil.which("ffmpeg")
    if not p:
        import glob
        for pat in [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\**\ffmpeg.exe"),
            os.path.expandvars(r"%ProgramData%\chocolatey\bin\ffmpeg.exe"),
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]:
            hits = glob.glob(pat, recursive=True)
            if hits:
                p = hits[0]
                break
    if not p:
        log("ffmpeg未検出 → imageio-ffmpeg を再インストールします...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "--force-reinstall", "imageio-ffmpeg"],
                           capture_output=True, timeout=300)
        except Exception as e:
            log("再インストール失敗:", e)
        p = _try_imageio()
    if not p:
        raise RuntimeError(
            "ffmpegが見つかりません。対処法:\n"
            "① check.bat を実行して [3] ffmpeg の結果を確認\n"
            "② ウイルス対策ソフトが ffmpeg.exe を隔離していないか確認\n"
            "③ コマンドプロンプトで winget install Gyan.FFmpeg を実行して再起動")
    _FFMPEG = p
    log("ffmpeg:", p)
    return p


def run_ffmpeg(args):
    cmd = [ffmpeg_exe(), "-y", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except FileNotFoundError:
        global _FFMPEG
        _FFMPEG = None
        raise RuntimeError("ffmpegの実行に失敗しました。check.bat で診断してください")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg処理に失敗: " + (r.stderr or "")[-500:])


def to_wav(src, dst, sr=44100, mono=False):
    args = ["-i", str(src), "-vn", "-ar", str(sr)]
    if mono:
        args += ["-ac", "1"]
    run_ffmpeg(args + [str(dst)])


def to_mp3(src, dst, bitrate="128k"):
    run_ffmpeg(["-i", str(src), "-vn", "-b:a", bitrate, str(dst)])


def compress_audio(src, out_dir, name, fmt="opus", bitrate="64k"):
    """再生用に音源を圧縮する。同じ音質なら opus の方が mp3 より半分近く小さい。

    ffmpeg のビルドによっては libopus が入っていないので、失敗したら mp3 に落とす。
    戻り値: 保存したファイル名
    """
    if fmt == "opus":
        dst = Path(out_dir) / f"{name}.opus"
        try:
            run_ffmpeg(["-i", str(src), "-vn", "-c:a", "libopus",
                        "-b:a", bitrate, "-application", "audio", str(dst)])
            return dst.name
        except Exception as e:
            log("opus変換に失敗 → mp3で保存します:", str(e)[:120])
    dst = Path(out_dir) / f"{name}.mp3"
    to_mp3(src, dst, bitrate="128k")
    return dst.name


# ---------------------------------------------------------------- 譜面生成
def build_score(song_dir, stems, grid, key_name, fifths, duration, cfg, job=None):
    """各ステムを採譜して score.json の中身を作る"""
    parts, raw = {}, {}
    cache = {}
    pitch_engine = engines.resolve("pitch", cfg)
    drum_engine = engines.resolve("drums", cfg)
    dev = engines.device(cfg)
    ext_cmd = cfg.get("pitch_external_cmd", "")
    want = engines.wanted_parts(cfg)
    todo = [k for k, _ in PART_SPEC if k in want]
    log("採譜するパート:", ", ".join(todo))
    for i, name in enumerate(todo):
        spec = PART_SPEC_MAP[name]
        if job:
            job["detail"] = f"{spec['label']} を採譜中... ({i+1}/{len(todo)})"
            job["progress"] = round(i / len(todo) * 100)
        stem_key, stem_path = pick_stem(spec, stems)
        if not stem_path:
            continue
        try:
            if spec["kind"] == "drums":
                ev = transcribe.transcribe_drums(stem_path, engine=drum_engine,
                                                 device=dev)
                notes = []
                starts = grid.time_to_beat([e["start"] for e in ev]) if ev else []
                for e, b in zip(ev, starts):
                    b = round(float(analysis.quantize(b, 4)), 4)
                    if b < -0.25:
                        continue
                    notes.append({"b": max(b, 0.0), "d": 0.25,
                                  "inst": e["inst"], "vel": e["vel"]})
                notes.sort(key=lambda n: n["b"])
                parts[name] = {"label": spec["label"], "kind": "drums",
                               "stem": stem_key, "notes": notes,
                               "engine": drum_engine,
                               "stats": {"notes": len(notes)}}
            else:
                sec_notes = transcribe.transcribe_pitched(
                    stem_path, name, engine=pitch_engine, cache=cache,
                    external_cmd=ext_cmd)
                qn = analysis.quantize_notes(sec_notes, grid, div=4)
                if spec["kind"] == "tab":
                    # 運指のやり直しに使うので、TAB譜のパートだけ元データを残す
                    raw[name] = qn
                    tabbed, strings = tabgen.make_tab(qn, spec["instrument"], "standard", 0)
                    parts[name] = {
                        "label": spec["label"], "kind": "tab",
                        "instrument": spec["instrument"], "tuning": "standard",
                        "capo": 0, "strings": strings, "stem": stem_key,
                        "engine": pitch_engine, "notes": tabbed,
                        "stats": tabgen.tab_stats(tabbed)}
                else:
                    parts[name] = {"label": spec["label"], "kind": "staff",
                                   "stem": stem_key, "engine": pitch_engine,
                                   "notes": qn, "stats": {"notes": len(qn)}}
        except Exception as e:
            traceback.print_exc()
            log(f"{name} の採譜に失敗:", e)
    return parts, raw


def run_prepare_job(job_id, song_id, get_source_wav, title_hint):
    job = jobs[job_id]

    def set_phase(i, progress=None, detail=""):
        job.update(phase=i, progress=progress, detail=detail)

    cfg = engines.load_config()
    song_dir = SONGS_DIR / song_id
    timings, _t0 = {}, [time.time()]

    def lap(name):
        """処理時間を記録する。どこが遅いか後から分かるように"""
        now = time.time()
        timings[name] = round(now - _t0[0], 1)
        _t0[0] = now
        log(f"  [{name}] {timings[name]}秒")

    try:
        song_dir.mkdir(exist_ok=True)

        # 1) 音源
        log("job", job_id, "phase 1: 音源取得")
        set_phase(0)
        title = get_source_wav(song_dir / "source.wav", job) or title_hint
        job["title"] = title
        lap("音源取得")

        # 2) パート分離
        sep_engine = engines.resolve("separator", cfg)
        dev = engines.device(cfg)
        log("job", job_id, f"phase 2: パート分離 ({sep_engine}, device={dev})")
        set_phase(1, 0, f"分離モデル: {sep_engine}")
        sep_dir = TMP_DIR / job_id
        found = separate.separate(
            song_dir / "source.wav", sep_dir, sep_engine, device=dev,
            on_progress=lambda pct, detail="": job.update(
                progress=pct, **({"detail": detail} if detail else {})))
        stems = {}
        for name, src in found.items():
            dst = song_dir / f"{name}.wav"
            shutil.move(str(src), dst)
            stems[name] = dst
        separate.cleanup(sep_dir)
        log("job", job_id, "ステム:", ", ".join(sorted(stems)))
        lap("パート分離")

        # 3) 拍・小節
        beat_engine = engines.resolve("beat", cfg)
        log("job", job_id, f"phase 3: 拍解析 ({beat_engine})")
        set_phase(2, None, "拍と小節の頭を推定中...")
        import librosa
        y, sr = librosa.load(str(song_dir / "source.wav"), sr=ANALYSIS_SR, mono=True)
        duration = len(y) / sr
        grid, beat_used = analysis.estimate_grid_auto(
            song_dir / "source.wav", y, sr, engine=beat_engine, device=dev)
        key_name, fifths, _ = analysis.estimate_key(y, sr)
        del y
        log(f"tempo={grid.tempo:.1f} {grid.beats_per_bar}/4 key={key_name}")
        lap("拍・調の解析")

        # 4) 採譜
        log("job", job_id, "phase 4: 採譜")
        set_phase(3, 0, "採譜を準備中...")
        parts, raw = build_score(song_dir, stems, grid, key_name, fifths,
                                 duration, cfg, job)
        if not parts:
            raise RuntimeError("どのパートも採譜できませんでした")
        lap("採譜")

        # 4b) コード進行
        chord_list = []
        if engines.resolve("chords", cfg) == "builtin":
            job.update(detail="コード進行を推定中...", progress=None)
            try:
                harmonic = [stems[k] for k in ("other", "guitar", "piano", "vocals")
                            if k in stems]
                chord_list = chordmod.recognize(
                    harmonic, stems.get("bass"), grid, flats=(fifths < 0))
                log("job", job_id, f"コード {len(chord_list)} 区間")
            except Exception as e:
                traceback.print_exc()
                log("コード認識に失敗:", e)
            lap("コード認識")

        # 5) 保存 + 音源をmp3化して容量を節約
        log("job", job_id, "phase 5: 譜面を生成")
        set_phase(4, None, "音源を圧縮中...")
        fmt = cfg.get("audio_format", "opus")
        keep = cfg.get("keep_stems", "used")
        if keep == "none":
            keep_set = set()
        elif keep == "all":
            keep_set = set(stems)
        else:  # 譜面に使ったステムだけ残す
            keep_set = {p.get("stem") for p in parts.values() if p.get("stem")}
        audio = {}
        targets = [("mix", song_dir / "source.wav")] + \
                  [(k, v) for k, v in stems.items() if k in keep_set]
        for key, src in targets:
            try:
                audio[key] = compress_audio(src, song_dir, key, fmt)
            except Exception as e:
                log("音源の圧縮をスキップ:", e)
                audio[key] = Path(src).name
                continue
            Path(src).unlink(missing_ok=True)
        for key, src in stems.items():   # 残さないステムのwavは捨てる
            if key not in keep_set:
                Path(src).unlink(missing_ok=True)
        Path(song_dir / "source.wav").unlink(missing_ok=True)

        measures = 1
        for p in parts.values():
            for n in p["notes"]:
                measures = max(measures, int(n["b"] // grid.beats_per_bar) + 1)

        used = {"separator": sep_engine, "beat": beat_used,
                "pitch": engines.resolve("pitch", cfg),
                "drums": engines.resolve("drums", cfg),
                "chords": engines.resolve("chords", cfg), "device": dev}
        score = {
            "id": song_id, "title": title, "duration": round(duration, 1),
            "tempo": round(grid.tempo, 1), "beats_per_bar": grid.beats_per_bar,
            "key": key_name, "fifths": fifths, "measures": measures,
            "engines": used, "grid": grid.to_dict(), "audio": audio,
            "parts": parts, "chords": chord_list, "raw": raw,
            "timings": timings, "created": time.time(),
        }
        (song_dir / "score.json").write_text(
            json.dumps(score, ensure_ascii=False), encoding="utf-8")
        (song_dir / "meta.json").write_text(json.dumps({
            "id": song_id, "title": title, "duration": round(duration, 1),
            "tempo": round(grid.tempo, 1), "key": key_name, "engines": used,
            "beats_per_bar": grid.beats_per_bar,
            "video_id": job.get("video_id"),
            "measures": measures, "created": score["created"],
            "parts": {k: v["stats"].get("notes", 0) for k, v in parts.items()},
        }, ensure_ascii=False), encoding="utf-8")

        lap("保存・圧縮")
        job.update(status="done", phase=len(PHASES), progress=None, detail="")
        total = round(sum(timings.values()), 1)
        size = sum(f.stat().st_size for f in song_dir.iterdir() if f.is_file())
        log("job", job_id, f"完了: {title} — 計{total}秒 / {size/1e6:.1f}MB")
        log("  内訳:", "  ".join(f"{k} {v}s" for k, v in timings.items()))
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=str(e))
        shutil.rmtree(song_dir, ignore_errors=True)


def start_job(get_source_wav, title_hint=""):
    job_id = uuid.uuid4().hex[:12]
    song_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running", "phases": PHASES, "phase": 0,
                    "progress": None, "detail": "", "error": None,
                    "song_id": song_id, "title": title_hint,
                    "started": time.time()}
    threading.Thread(target=run_prepare_job,
                     args=(job_id, song_id, get_source_wav, title_hint),
                     daemon=True).start()
    return job_id


# ---------------------------------------------------------------- API
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/config")
def get_config():
    return jsonify({
        "tunings": {inst: {k: v["label"] for k, v in d.items()}
                    for inst, d in tabgen.TUNINGS.items()},
        "drum_order": transcribe.DRUM_ORDER,
        "drum_labels": transcribe.DRUM_LABELS,
        "engines": engines.describe(),
    })


@app.post("/api/config")
def set_config():
    body = request.get_json(silent=True) or {}
    engines.save_config(body)
    log("設定を更新:", engines.summary_line())
    return jsonify(engines.describe())


@app.get("/api/search")
def yt_search():
    """曲名やアーティスト名でYouTubeを検索する

    extract_flat で一覧情報だけ取るので、動画1本ずつ解析するより桁違いに速い。
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})
    try:
        n = max(1, min(int(request.args.get("n", 20) or 20), 40))
    except ValueError:
        n = 20
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True,
            "skip_download": True, "socket_timeout": 20, "cachedir": False}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{n}:{q}", download=False)
    except Exception as e:
        log("検索に失敗:", e)
        return jsonify({"error": f"検索に失敗しました: {e}"}), 502

    # 取り込み済みの曲には印を付ける (動画IDで照合し、無ければタイトルで)
    have_ids, have_titles = set(), set()
    for d in SONGS_DIR.iterdir():
        m = d / "meta.json"
        if m.exists():
            try:
                j = json.loads(m.read_text(encoding="utf-8"))
                if j.get("video_id"):
                    have_ids.add(j["video_id"])
                have_titles.add(j.get("title", ""))
            except Exception:
                pass

    out = []
    for e in (info.get("entries") or []):
        vid = e.get("id")
        if not vid:
            continue
        title = e.get("title") or "無題"
        out.append({
            "id": vid, "title": title,
            "channel": e.get("uploader") or e.get("channel") or "",
            "duration": e.get("duration"),
            "views": e.get("view_count"),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "thumb": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg",
            "have": vid in have_ids or title in have_titles,
        })
    return jsonify({"results": out})


@app.post("/api/songs/youtube")
def add_youtube():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not re.match(r"^https?://", url):
        return jsonify({"error": "URLが正しくありません"}), 400

    def fetch(dst_wav, job):
        import yt_dlp
        tmp = TMP_DIR / uuid.uuid4().hex
        tmp.mkdir()

        def hook(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                done = d.get("downloaded_bytes")
                if total and done:
                    job["progress"] = round(done / total * 100)
                if d.get("speed"):
                    job["detail"] = f"{d['speed']/1e6:.1f} MB/s"
            elif d.get("status") == "finished":
                job.update(progress=None, detail="音声を変換中...")

        opts = {"format": "bestaudio/best", "outtmpl": str(tmp / "audio.%(ext)s"),
                "noplaylist": True, "quiet": True, "no_warnings": True,
                "socket_timeout": 30, "retries": 3, "fragment_retries": 5,
                "cachedir": False, "progress_hooks": [hook]}
        attempts = [
            ("標準", {}),
            ("android", {"extractor_args": {"youtube": {"player_client": ["android"]}}}),
            ("ios", {"extractor_args": {"youtube": {"player_client": ["ios"]}}}),
            ("tv", {"extractor_args": {"youtube": {"player_client": ["tv"]}}}),
            ("低画質動画から抽出", {"format": "18/worst"}),
        ]
        info, last_err = None, None
        for i, (name, extra) in enumerate(attempts):
            if i > 0:
                job.update(progress=None,
                           detail=f"取得方法を変えて再試行中 ({i+1}/{len(attempts)}: {name})...")
                for p in tmp.iterdir():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            try:
                with yt_dlp.YoutubeDL({**opts, **extra}) as ydl:
                    info = ydl.extract_info(url, download=True)
                break
            except Exception as e:
                last_err = e
        if info is None:
            raise RuntimeError(
                f"YouTubeの取得に失敗しました (全{len(attempts)}方式で試行): {last_err}\n"
                "対処: ① update-ytdlp.bat で更新 ② 別のURLで試す ③ 時間を置いて再試行")
        src = next((p for p in tmp.iterdir() if p.is_file()), None)
        if not src:
            raise RuntimeError("音声のダウンロードに失敗しました")
        job["video_id"] = info.get("id")
        job.update(progress=None, detail="音声をWAVに変換中...")
        to_wav(src, dst_wav)
        shutil.rmtree(tmp, ignore_errors=True)
        return info.get("title", "無題")

    return jsonify({"job": start_job(fetch)})


@app.post("/api/songs/upload")
def add_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "ファイルがありません"}), 400
    tmp = TMP_DIR / (uuid.uuid4().hex + Path(f.filename).suffix)
    f.save(tmp)
    title = Path(f.filename).stem

    def fetch(dst_wav, job):
        job["detail"] = "音声を変換中..."
        to_wav(tmp, dst_wav)
        tmp.unlink(missing_ok=True)
        return title

    return jsonify({"job": start_job(fetch, title)})


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "ジョブが見つかりません", "status": "error"}), 404
    return jsonify(job)


@app.get("/api/songs")
def list_songs():
    out, total = [], 0
    for d in SONGS_DIR.iterdir():
        m = d / "meta.json"
        if m.exists():
            try:
                j = json.loads(m.read_text(encoding="utf-8"))
                j["size"] = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
                total += j["size"]
                out.append(j)
            except Exception:
                pass
    out.sort(key=lambda s: -s.get("created", 0))
    return jsonify({"songs": out, "total_size": total})


@app.delete("/api/songs/<song_id>")
def delete_song(song_id):
    shutil.rmtree(SONGS_DIR / song_id, ignore_errors=True)
    return jsonify({"ok": True})


def _load_score(song_id):
    p = SONGS_DIR / song_id / "score.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save_score(song_id, score):
    (SONGS_DIR / song_id / "score.json").write_text(
        json.dumps(score, ensure_ascii=False), encoding="utf-8")


@app.get("/api/songs/<song_id>/score")
def get_score(song_id):
    score = _load_score(song_id)
    if not score:
        return jsonify({"error": "曲が見つかりません"}), 404
    slim = {k: v for k, v in score.items() if k not in ("raw", "grid")}
    slim["bar0_sec"] = score.get("grid", {}).get("bar0_sec", 0.0)
    slim["sec_per_beat"] = score.get("grid", {}).get("sec_per_beat", 0.5)
    slim["beat_times"] = score.get("grid", {}).get("beat_times", [])
    slim["phase"] = score.get("grid", {}).get("phase", 0)
    return jsonify(slim)


@app.get("/api/songs/<song_id>/audio/<name>")
def get_audio(song_id, name):
    if not re.fullmatch(r"[a-z]+", name):
        return "not found", 404
    d = SONGS_DIR / song_id
    for ext, mime in (("opus", "audio/ogg"), ("mp3", "audio/mpeg"),
                      ("wav", "audio/wav")):
        p = d / f"{name}.{ext}"
        if p.exists():
            return send_file(p, mimetype=mime, conditional=True)
    return "not found", 404


@app.post("/api/songs/<song_id>/retab")
def retab(song_id):
    """チューニング・カポ・使用フレット範囲を変えて運指を作り直す"""
    score = _load_score(song_id)
    if not score:
        return jsonify({"error": "曲が見つかりません"}), 404
    body = request.get_json(silent=True) or {}
    part = body.get("part")
    p = score["parts"].get(part)
    if not p or p["kind"] != "tab":
        return jsonify({"error": "TAB譜のパートではありません"}), 400
    raw = score.get("raw", {}).get(part) or []
    tuning = body.get("tuning", p.get("tuning", "standard"))
    capo = int(body.get("capo", 0) or 0)
    max_fret = int(body.get("max_fret", 0) or 0) or None
    notes, strings = tabgen.make_tab(raw, p["instrument"], tuning, capo, max_fret)
    p.update(tuning=tuning, capo=capo, strings=strings, notes=notes,
             stats=tabgen.tab_stats(notes))
    _save_score(song_id, score)
    return jsonify(p)


@app.post("/api/songs/<song_id>/shift")
def shift_bars(song_id):
    """小節の頭の位置を手で直す (自動推定が1拍ずれたときの逃げ道)"""
    score = _load_score(song_id)
    if not score:
        return jsonify({"error": "曲が見つかりません"}), 404
    delta = float((request.get_json(silent=True) or {}).get("beats", 0))
    if delta == 0:
        return jsonify({"ok": True})

    def move(items):
        out = []
        for n in items:
            m = dict(n)
            m["b"] = round(m["b"] - delta, 4)
            if m["b"] < -1e-6:  # 先頭より前にはみ出した分は切り詰める
                over = -m["b"]
                if m.get("d", 0) - over <= 1e-6:
                    continue
                m["d"] = round(m["d"] - over, 4)
            m["b"] = max(m["b"], 0.0)
            out.append(m)
        return out

    for p in score["parts"].values():
        p["notes"] = move(p["notes"])
    for k in list(score.get("raw", {})):
        score["raw"][k] = move(score["raw"][k])
    if score.get("chords"):
        score["chords"] = move(score["chords"])
    g = score.setdefault("grid", {})
    g["phase"] = g.get("phase", 0) + delta
    bpb = score["beats_per_bar"]
    bt = g.get("beat_times") or []
    idx = g["phase"]
    if 0 <= idx < len(bt):
        g["bar0_sec"] = bt[int(idx)]
    else:
        g["bar0_sec"] = g.get("bar0_sec", 0.0) + delta * g.get("sec_per_beat", 0.5)
    score["measures"] = max(1, max(
        [int(n["b"] // bpb) + 1 for p in score["parts"].values() for n in p["notes"]]
        or [1]))
    for p in score["parts"].values():
        if p["kind"] == "tab":
            p["stats"] = tabgen.tab_stats(p["notes"])
        else:
            p["stats"] = {"notes": len(p["notes"])}
    _save_score(song_id, score)
    return jsonify({"ok": True})


def _export_parts(score, wanted):
    out = []
    for i, name in enumerate(wanted):
        p = score["parts"].get(name)
        if not p:
            continue
        spec = PART_SPEC_MAP[name]
        out.append({"id": f"P{i+1}", "name": p["label"], "kind": p["kind"],
                    "tuning": p.get("strings"), "capo": p.get("capo", 0),
                    "clef": spec.get("clef", ("G", 2)),
                    "program": spec.get("program", 0) + 1,
                    "channel": spec.get("channel", 0) + 1,
                    "notes": p["notes"]})
    return out


@app.get("/api/songs/<song_id>/export/musicxml")
def export_musicxml(song_id):
    score = _load_score(song_id)
    if not score:
        return jsonify({"error": "曲が見つかりません"}), 404
    wanted = [x for x in (request.args.get("parts") or "").split(",") if x] \
        or list(score["parts"].keys())
    xml = export.build_musicxml(
        _export_parts(score, wanted), tempo=score["tempo"],
        beats_per_bar=score["beats_per_bar"], fifths=score.get("fifths", 0),
        title=score["title"], chords=score.get("chords") or None)
    fname = _safe_name(score["title"]) + ".musicxml"
    return Response(xml, mimetype="application/vnd.recordare.musicxml+xml",
                    headers={"Content-Disposition":
                             f"attachment; filename*=UTF-8''{_url_quote(fname)}"})


@app.get("/api/songs/<song_id>/export/midi")
def export_midi(song_id):
    score = _load_score(song_id)
    if not score:
        return jsonify({"error": "曲が見つかりません"}), 404
    wanted = [x for x in (request.args.get("parts") or "").split(",") if x] \
        or list(score["parts"].keys())
    tracks = []
    for name in wanted:
        p = score["parts"].get(name)
        if not p:
            continue
        spec = PART_SPEC_MAP[name]
        tracks.append({"name": p["label"], "program": spec.get("program", 0),
                       "channel": spec.get("channel", 0), "notes": p["notes"]})
    data = export.build_midi(tracks, tempo=score["tempo"],
                             beats_per_bar=score["beats_per_bar"],
                             title=score["title"])
    fname = _safe_name(score["title"]) + ".mid"
    return Response(data, mimetype="audio/midi",
                    headers={"Content-Disposition":
                             f"attachment; filename*=UTF-8''{_url_quote(fname)}"})


@app.get("/api/songs/<song_id>/export/txt")
def export_txt(song_id):
    score = _load_score(song_id)
    if not score:
        return jsonify({"error": "曲が見つかりません"}), 404
    name = request.args.get("part", "guitar")
    p = score["parts"].get(name)
    if not p or p["kind"] != "tab":
        return jsonify({"error": "TAB譜のパートではありません"}), 400
    text = (f"{score['title']} — {p['label']}\n"
            f"Tempo {score['tempo']} / Key {score['key']}"
            f"{' / Capo ' + str(p['capo']) if p.get('capo') else ''}\n\n")
    text += export.build_text_tab(p["notes"], p["strings"], score["beats_per_bar"])
    fname = f"{_safe_name(score['title'])}_{name}.txt"
    return Response(text.encode("utf-8"), mimetype="text/plain; charset=utf-8",
                    headers={"Content-Disposition":
                             f"attachment; filename*=UTF-8''{_url_quote(fname)}"})


def _safe_name(s):
    return re.sub(r'[\\/:*?"<>|]+', "_", (s or "score")).strip()[:60] or "score"


def _url_quote(s):
    from urllib.parse import quote
    return quote(s)


# ---------------------------------------------------------------- 起動
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        pass


if __name__ == "__main__":
    logf = open(BASE_DIR / "server.log", "a", encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, logf)
    sys.stderr = _Tee(sys.stderr, logf)
    print("=" * 56)
    print(f"  TAB Maker  http://127.0.0.1:{PORT}")
    print("  起動:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  (このウィンドウに処理ログが流れます。閉じると終了します)")
    print("=" * 56, flush=True)
    if len(str(BASE_DIR)) > 80:
        print("!" * 56)
        print("  警告: このフォルダのパスが長すぎます (%d文字)" % len(str(BASE_DIR)))
        print("  " + str(BASE_DIR))
        print("  C:\\tabmaker など浅い場所に移動して setup.bat からやり直してください")
        print("!" * 56, flush=True)
    print("使用エンジン:", engines.summary_line(), flush=True)
    for _stage, _s in engines.describe()["stages"].items():
        _miss = [o["label"] for o in _s["options"] if not o["available"]]
        if _miss:
            print(f"  ({_s['label']}: 未導入 = {', '.join(_miss)})", flush=True)
    print("  ※ 未導入のものは check.bat で理由が確認できます", flush=True)

    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", PORT))
        probe.close()
    except OSError:
        print(f"ポート{PORT}は使用中です。既に起動済みのようなのでブラウザを開きます。")
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        input("Enterキーで閉じます...")
        raise SystemExit(0)

    def _open_browser():
        for _ in range(30):
            try:
                with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            return
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True)
    except Exception:
        traceback.print_exc()
        print("\nサーバーがエラーで停止しました。server.log を確認してください。")
        input("Enterキーで閉じます...")
