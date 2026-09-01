# 基本設計書 — TAB Maker

| 項目 | 内容 |
|---|---|
| 文書ID | TM-BD-001 |
| 版 | 1.0 |
| 上位文書 | TM-REQ-001 要件定義書 |

---

## 1. システム構成

### 1.1 全体像

ローカルPC上で完結するWebアプリケーション。バックエンドはPython(Flask)、
フロントエンドは単一のHTMLファイル。外部通信は音源取得とモデルのダウンロードのみ。

```
┌─────────────────────────── 利用者のPC ───────────────────────────┐
│                                                                   │
│  ブラウザ                          Pythonプロセス (127.0.0.1:8766)  │
│  ┌──────────────┐   HTTP/JSON     ┌────────────────────────────┐  │
│  │ static/      │ ◄─────────────► │ app.py (Flask)             │  │
│  │  index.html  │                 │  ├ ジョブ管理(スレッド)     │  │
│  │  - 検索      │                 │  ├ REST API                │  │
│  │  - 譜面描画  │   音源(HTTP)     │  └ 進捗の保持              │  │
│  │  - 再生      │ ◄────────────── │                            │  │
│  └──────────────┘                 │  engines.py  エンジン選択   │  │
│                                   │  separate.py 分離           │  │
│                                   │  analysis.py 拍・調・量子化 │  │
│                                   │  transcribe.py 採譜         │  │
│                                   │  chords.py   コード認識     │  │
│                                   │  tabgen.py   運指最適化     │  │
│                                   │  export.py   楽譜出力       │  │
│                                   └────────────┬───────────────┘  │
│                                                │                  │
│   songs/<曲ID>/  譜面と音源       config.json  │  外部プロセス     │
│   ┌────────────────────────┐     設定         │  ┌─────────────┐ │
│   │ meta.json  score.json  │                  └─►│ demucs      │ │
│   │ mix.opus   bass.opus … │                     │ ffmpeg      │ │
│   └────────────────────────┘                     │ yt-dlp      │ │
│                                                  └─────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼ (音源取得・モデルDL時のみ)
                                   YouTube / モデル配布元
```

### 1.2 モジュール構成

| モジュール | 責務 | 他モジュールへの依存 |
|---|---|---|
| `app.py` | HTTPサーバー、ジョブ管理、処理の流れの制御 | 全モジュール |
| `engines.py` | エンジンの登録・利用可否判定・設定の永続化 | なし |
| `separate.py` | 音源分離。ステム名を揃えて返す | なし |
| `analysis.py` | 拍・調の推定、秒↔拍の変換、量子化 | なし |
| `transcribe.py` | 音符・打点の推定 | `export`(MIDI読み込み) |
| `chords.py` | コード進行の推定 | なし |
| `tabgen.py` | 運指の最適化 | なし |
| `export.py` | MusicXML / MIDI / テキストTAB の入出力 | `chords`(コード表記) |
| `static/index.html` | UI一式(HTML/CSS/JS を1ファイルに同梱) | — |
| `check.py` | 環境診断 | `app`, `engines` |
| `selftest.py` | 自動テスト | 全モジュール |

**依存の方向**: `app.py` が全体を束ね、処理モジュール同士は原則として互いを参照しない。
これにより各処理段を単独で差し替え・テストできる(要件 M-01)。

---

## 2. 処理フロー

### 2.1 採譜のメインフロー

```
[利用者] 検索 or URL入力 or ファイル選択
    │
    ▼
POST /api/songs/youtube  ──► ジョブIDを即座に返す(非同期)
    │                            │
    │                            ▼ 別スレッドで実行
    │                    ┌───────────────────────────────────┐
    │                    │ ① 音源取得                         │
    │                    │    yt-dlp → ffmpeg → source.wav    │
    │                    ├───────────────────────────────────┤
    │                    │ ② パート分離                       │
    │                    │    Demucs 6stem                    │
    │                    │    → vocals/drums/bass/            │
    │                    │      guitar/piano/other .wav       │
    │                    ├───────────────────────────────────┤
    │                    │ ③ 拍・小節の推定                   │
    │                    │    Beat This! → 拍時刻・小節頭     │
    │                    │    クロマ相関   → 調               │
    │                    ├───────────────────────────────────┤
    │                    │ ④ 採譜(パートごと)                │
    │                    │    音程: basic-pitch → 量子化      │
    │                    │    ドラム: ADTOF → ベロシティ推定  │
    │                    │    TAB: 運指の動的計画法           │
    │                    ├───────────────────────────────────┤
    │                    │ ④b コード進行                      │
    │                    │    クロマ + ビタビ                 │
    │                    ├───────────────────────────────────┤
    │                    │ ⑤ 保存                             │
    │                    │    score.json / meta.json          │
    │                    │    音源をOpusに圧縮、wavを破棄     │
    │                    └───────────────────────────────────┘
    │
    ▼ 0.7秒ごとにポーリング
GET /api/jobs/<id> ──► {phase, progress, detail, status}
    │
    ▼ status=done
GET /api/songs/<id>/score ──► 譜面描画
```

### 2.2 進捗の表現

ジョブは以下の状態を持ち、UIはこれをそのまま段階表示に使う。

| フィールド | 意味 |
|---|---|
| `status` | `running` / `done` / `error` |
| `phases` | 段階名の配列(5段階) |
| `phase` | 現在の段階(0始まり) |
| `progress` | 現在段階の進捗率(0〜100)。不明なら `null` |
| `detail` | 補足メッセージ(「モデルをダウンロード中」など) |
| `error` | 失敗理由 |

### 2.3 フォールバック方針(要件 R-01)

各処理段は「重いが高精度」→「軽いが低精度」の順に試し、例外が出たら次に落ちる。
処理全体は止めない。

| 処理段 | 優先 | フォールバック |
|---|---|---|
| 分離 | RoFormer+Demucs 6stem | Demucs 6stem |
| 拍 | Beat This! | librosa(自作ヒューリスティック) |
| 音符 | 外部スクリプト → basic-pitch | librosa pyin |
| ドラム | ADTOF | 内蔵の帯域判定 |
| 音源圧縮 | Opus | MP3 → 無変換 |

---

## 3. データ設計

### 3.1 ディレクトリ構成

```
C:\tabmaker\
├── *.py, static/, *.bat        プログラム本体
├── config.json                 利用者の設定(git管理外)
├── server.log                  実行ログ(git管理外)
├── songs/                      曲データ(git管理外)
│   └── <曲ID(12桁hex)>/
│       ├── meta.json           一覧表示用の軽い情報
│       ├── score.json          譜面データ本体
│       ├── mix.opus            全体ミックス
│       └── <ステム名>.opus     パート別音源(設定により省略)
├── tmp/                        処理中の一時ファイル(git管理外)
└── docs/                       設計資料
```

### 3.2 score.json

譜面データの中心となるファイル。

```jsonc
{
  "id": "a1b2c3d4e5f6",
  "title": "曲名",
  "duration": 245.3,              // 秒
  "tempo": 128.4,                 // BPM
  "beats_per_bar": 4,
  "key": "C major",
  "fifths": 0,                    // MusicXMLの調号(シャープ数。負はフラット)
  "measures": 96,
  "engines": { "separator": "htdemucs_6s", "beat": "beat_this", … },
  "timings": { "音源取得": 12.4, "パート分離": 310.2, … },
  "grid": {
    "tempo": 128.4, "beats_per_bar": 4, "sec_per_beat": 0.467,
    "beat_times": [0.31, 0.78, 1.25, …],   // 拍の時刻(テンポ変動を保持)
    "phase": 2,                            // 最初の小節頭の拍番号
    "bar0_sec": 1.25
  },
  "audio": { "mix": "mix.opus", "bass": "bass.opus" },
  "parts": {
    "bass": {
      "label": "ベース", "kind": "tab", "instrument": "bass",
      "tuning": "standard", "capo": 0,
      "strings": [43, 38, 33, 28],         // 1弦→4弦の開放弦(MIDIノート番号)
      "stem": "bass", "engine": "basic_pitch",
      "notes": [ {"b":0.0,"d":1.0,"midi":40,"vel":92,"string":4,"fret":0}, … ],
      "stats": {"notes": 412, "max_fret": 9, "min_fret": 0, "open_ratio": 0.21}
    }
  },
  "chords": [ {"b":0.0,"d":4.0,"name":"C","root":0,"quality":"maj","kind":"major"}, … ],
  "raw": { "bass": [ {"b":0.0,"d":1.0,"midi":40,"vel":92}, … ] },
  "created": 1788000000.0
}
```

**設計上のポイント**

- 時間軸はすべて**拍**(`b`=開始拍, `d`=長さ)。拍0が最初の小節頭。
  秒への変換は `grid.beat_times` を通して行う。これによりテンポ変動があっても
  譜面と音源が一致する。
- `raw` は運指のやり直し用の弦・フレット割り当て前の音符。**TAB譜のパートのみ**保持する
  (五線譜のパートは再計算しないため、持つだけ無駄)。
- `/api/songs/<id>/score` は `raw` と `grid` を除いた軽量版を返し、
  代わりに描画に必要な `beat_times` / `phase` / `sec_per_beat` を展開する。

### 3.3 パートの種別

| kind | 対象 | 描画 | MusicXML |
|---|---|---|---|
| `tab` | ギター、ベース | 弦の本数だけ線を引き、フレット番号を置く | TABクレフ + `<technical>` |
| `staff` | メロディ、ピアノ | 五線譜。符頭・符幹・加線・臨時記号 | ト音記号 |
| `drums` | ドラム | 打楽器ごとの行に記号(●/✕)を置く | 打楽器クレフ + `<unpitched>` |

### 3.4 config.json

```jsonc
{
  "separator": "auto",     // auto | htdemucs_6s | htdemucs | htdemucs_ft | roformer_hybrid
  "beat":      "auto",     // auto | beat_this | librosa
  "pitch":     "auto",     // auto | basic_pitch | librosa | external
  "drums":     "auto",     // auto | adtof | builtin
  "chords":    "builtin",  // builtin | off
  "device":    "auto",     // auto | cpu | cuda
  "parts": ["melody","guitar","bass","piano","drums"],
  "audio_format": "opus",  // opus | mp3
  "keep_stems":   "used",  // used | all | none
  "pitch_external_cmd": ""
}
```

`auto` は「インストール済みのものの中で最も推奨度が高いもの」を意味する。
GPU前提のエンジンは、`auto` では選ばれない(明示的に指定した場合のみ使う)。

---

## 4. 外部インタフェース

### 4.1 REST API

| メソッド | パス | 概要 |
|---|---|---|
| GET | `/` | UI(index.html) |
| GET | `/api/config` | チューニング一覧・ドラム表示順・エンジン一覧 |
| POST | `/api/config` | 設定の更新。更新後の状態を返す |
| GET | `/api/search?q=&n=` | YouTube検索 |
| POST | `/api/songs/youtube` | URLから採譜を開始。`{job: ジョブID}` |
| POST | `/api/songs/upload` | ファイルから採譜を開始 |
| GET | `/api/jobs/<job_id>` | ジョブの進捗 |
| GET | `/api/songs` | 曲一覧 `{songs: [...], total_size: バイト}` |
| DELETE | `/api/songs/<id>` | 曲の削除 |
| GET | `/api/songs/<id>/score` | 譜面データ(raw除く) |
| GET | `/api/songs/<id>/audio/<name>` | 音源(opus/mp3/wav の順に探す) |
| POST | `/api/songs/<id>/retab` | 運指の再計算 `{part, tuning, capo, max_fret}` |
| POST | `/api/songs/<id>/shift` | 小節頭の補正 `{beats: ±1}` |
| GET | `/api/songs/<id>/export/musicxml?parts=` | MusicXML |
| GET | `/api/songs/<id>/export/midi?parts=` | MIDI |
| GET | `/api/songs/<id>/export/txt?part=` | テキストTAB |

**エラー応答**: `{"error": "メッセージ"}` と適切なHTTPステータス。
想定外の例外は500で捕捉しJSONで返す(UIが固まらないようにするため)。
HTTP例外(404等)はそのまま通す。

### 4.2 外部プロセス

| 対象 | 呼び方 | 用途 |
|---|---|---|
| yt-dlp | Pythonライブラリ | 音源取得、検索 |
| ffmpeg | サブプロセス | 形式変換、Opus圧縮 |
| Demucs | `python -m demucs` サブプロセス | パート分離(標準出力から進捗を取得) |
| Beat This! | Pythonライブラリ | 拍・小節頭 |
| basic-pitch | Pythonライブラリ | 音符推定 |
| ADTOF | Pythonライブラリ(MIDI経由) | ドラム採譜 |
| 外部採譜スクリプト | 任意のコマンド(MIDI経由) | 拡張点 |

**ffmpegの探索**: 同梱(imageio-ffmpeg)→ PATH → 既知のインストール先 →
再インストール、の順に多段で探す。ウイルス対策ソフトによる隔離を想定した設計。

---

## 5. 画面設計

### 5.1 画面遷移

```
┌─ ホーム ───────────────────────────┐
│ ① 曲を探す(検索・結果一覧)         │
│ ② URL・ファイルから追加             │
│    └ 処理中は進捗パネルを表示        │
│ ③ 採譜した曲(一覧・容量)           │──[譜面を見る]──┐
│ ④ 処理エンジンの設定(折りたたみ)   │                 │
└─────────────────────────────────────┘                 ▼
                    ▲                        ┌─ 譜面 ──────────────────┐
                    └────[曲リストへ]─────── │ パートタブ                │
                                             │ 再生バー(音源選択)       │
                                             │ 工具列(チューニング等)   │
                                             │ 譜面(SVG・折り返し)      │
                                             │ 統計・注意書き            │
                                             └───────────────────────────┘
```

### 5.2 譜面の描画方式

SVGを文字列として組み立て、`innerHTML` で差し込む。1行あたりの小節数は
描画領域の幅から決める(1小節あたり最低210px)。

```
     ┌ コードネーム
     │      ┌ 小節番号
     C      Am
     ①      ②                    ← 各行の先頭に弦名
  E |───3───────5───────|───────  ← 拍の目盛り(薄い縦線)
  B |───────────────────|───────
  G |───2───────────────|───────
     ▲                              ← 再生位置(赤い縦線)
```

再生位置は `timeupdate` イベントごとに、`beat_times` を二分探索して拍に変換し、
小節・行・X座標を逆算して線を移動する。譜面全体の再描画は行わない。

---

## 6. 性能・容量の設計

### 6.1 処理時間の内訳(実測に基づく想定・4分の曲・CPU)

| 段階 | 時間 | 主な要因 |
|---|---|---|
| 音源取得 | 10〜30秒 | 回線速度 |
| パート分離 | **5〜12分** | Demucsの推論。**全体の8割を占める** |
| 拍・調 | 10〜30秒 | Beat This!の推論 |
| 採譜 | 1〜3分 | basic-pitch(ステム数に比例) |
| コード認識 | 5〜15秒 | CQT |
| 保存・圧縮 | 10〜30秒 | ffmpeg |

### 6.2 高速化の設計

| 手法 | 効果 | 実装 |
|---|---|---|
| ステム単位の推論キャッシュ | ギターとピアノが同じステムを使う場合、推論が1回で済む | `transcribe_pitched(cache=…)` |
| パートの絞り込み | 不要なパートの推論を丸ごと省略 | 設定 `parts` |
| 検索の簡易取得 | 動画ごとの詳細解析を行わない | `extract_flat` |
| 進捗の逐次取得 | 分離の待ち時間を体感的に短縮 | Demucsの標準出力を解析 |

### 6.3 容量削減の設計

| 手法 | 効果 | 実装 |
|---|---|---|
| Opus圧縮 | MP3 128kbpsの約半分 | `compress_audio()` |
| 中間wavの削除 | 1曲あたり数百MBを回収 | 圧縮直後に `unlink` |
| 使用ステムのみ保持 | 譜面に使わないステムを保存しない | 設定 `keep_stems` |
| `raw` をTAB譜のパートに限定 | score.jsonが約半分 | `build_score` |

**1曲あたりの想定容量(4分・既定設定)**

| 内訳 | 容量 |
|---|---|
| mix.opus | 約1.9MB |
| ステム(4つ) | 約7.7MB |
| score.json | 0.3〜1MB |
| **合計** | **約10〜11MB** |

`keep_stems="none"` にすると約2〜3MBまで落ちる。

---

## 7. エラー処理・ログ

| 種別 | 扱い |
|---|---|
| エンジンの実行失敗 | ログに記録し、軽いエンジンにフォールバックして継続 |
| 音源取得の失敗 | 取得方法を5通り試し、すべて失敗したらジョブを失敗させる |
| 処理中の失敗 | 曲ディレクトリごと削除し、原因をUIに表示 |
| 想定外の例外 | 500 + JSON、`server.log` にトレースバックを記録 |
| 依存関係の不整合 | `check.bat` が実際の例外メッセージと修復コマンドを表示 |

標準出力と `server.log` の両方に書き出す(`_Tee`)。処理中の黒い画面を閉じても
原因が追えるようにするため。

---

## 8. セキュリティ

- 待ち受けは `127.0.0.1` のみ。LANからは接続できない
- 音源名・ステム名は正規表現で検証し、パストラバーサルを防ぐ
- 認証は設けない(ローカル専用のため)
- 外部への送信は音源取得とモデルDLのみ。利用者のデータは送信しない
