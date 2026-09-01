# 精度向上のための改善案(2026年9月 調査)

現行実装の弱いところを、最近の論文・OSSと突き合わせて洗い出したもの。
「効果 ÷ 導入コスト」で並べてある。

## 実装状況

CPUで現実的なものは**すべて実装済み**。GPU前提のものも、切り替えられる形で入れてある。
エンジンはアプリの「処理エンジンの設定」から選ぶ(`config.json` に保存)。

| 改善 | 状態 | 選び方 |
|---|---|---|
| S-1 Beat This! で小節頭を取る | **実装済み・既定** | 拍・小節 → Beat This! |
| S-2 Demucs 6stem でギター/ピアノを分離 | **実装済み・既定** | パート分離 → Demucs 6stem |
| S-3 ドラムの強弱をRMSから推定 | **実装済み・常時有効** | — |
| A-1 ドラム採譜を ADTOF に | **実装済み・既定** | ドラム → ADTOF |
| A-3 コードネームを出す | **実装済み・既定**(内蔵実装) | コード進行 → 内蔵 |
| A-2 LarsNet でドラムをステム分離 | 未実装 | — |
| B-1 YourMT3+ 等の採譜モデル | **繋ぎ口を実装済み** | 音符推定 → 外部スクリプト |
| B-2 FretNet で弦・フレットを直接推定 | 未実装 | — |
| B-3 RoFormer系の分離 | **実装済み** | パート分離 → RoFormer+Demucs 6stem |

未実装の2つ(A-2 / B-2)は、どちらもモデルの重みや学習データの扱いが絡んで
導入が重いため見送っている。理由は各項目に書いた。

---

## 現行パイプラインと弱点の対応表

| 段階 | 現行の実装 | 弱点 | 差し替え候補 |
|---|---|---|---|
| ① 音源取得 | yt-dlp + ffmpeg | 特になし | — |
| ② パート分離 | Demucs `htdemucs` 4stem | **ギターが "other" に埋もれる**。シンセ・ストリングスが混ざる | `htdemucs_6s` / Mel-Band RoFormer / LarsNet |
| ③ テンポ・拍 | librosa `beat_track` + 自作の位相推定 | **小節頭を外す**。4/4固定。テンポ変化に弱い | **Beat This!**(ISMIR 2024) |
| ④ 採譜 | basic-pitch(音程)/ 帯域エネルギー(ドラム) | ドラムの分類が手作りの閾値。強弱情報なし | ADTOF / YourMT3+ / FretNet |
| ⑤ 運指 | 自作の動的計画法 | コスト関数が経験則。実際の弾きやすさのデータに基づかない | inhibition学習 / FretNet |
| — | (なし) | **コードネームが出ない** | autochord / BTC / ChordMini |

---

## 優先度 S — 効果が大きく、導入が軽い

### S-1. 小節頭の推定を Beat This! に置き換える

**いま何が問題か**: `analysis.estimate_grid()` の位相推定は「低域(バスドラム)のオンセットが強い拍が小節頭」という自作ヒューリスティック。
イントロがドラムなしで始まる曲、裏拍にキックが来る曲、シャッフルでは外す。小節頭がずれると譜面全体が読めなくなる。

**代替**: [Beat This!](https://github.com/CPJKU/beat_this)(CPJKU, ISMIR 2024)は
拍とダウンビートを同時に出力するTransformer。madmomが必要としていたDBN後処理なしで高精度を達成した論文の公式実装。

- **導入コスト: 極小**。`pip install beat-this` + `tqdm einops soxr rotary-embedding-torch`。
  PyTorchは Demucs 経由で既に入っている。モデルは `final0` で **78MB**(`small0` なら8.1MB)。MITライセンス。
- **使い方**:
  ```python
  from beat_this.inference import File2Beats
  beats, downbeats = File2Beats(checkpoint_path="final0", device="cpu")(audio_path)
  ```
  `downbeats` がそのまま小節線になるので、**位相推定のコードを丸ごと捨てられる**。
  拍子(何拍子か)もダウンビート間隔から数えられるので、3/4・6/8の曲にも対応できるようになる。
- **副次効果**: 拍がずれなくなるので量子化の精度も上がり、③以降すべてが良くなる。

> これが**一番やるべき改修**。1曲あたり数秒の追加コストで、譜面の読みやすさが根本的に変わる。

### S-2. Demucs を `htdemucs_6s` にする(ギター/ピアノの分離)

**いま何が問題か**: ギターとピアノを同じ `other` ステムから採譜しているので、シンセ・ストリングス・パッドが全部混ざる。
READMEに書いた「ギターだけの綺麗な譜面にはならない」の原因はここ。

**代替**: Demucs には **6ステム版 `htdemucs_6s`** があり、bass / drums / vocals / other に加えて
**guitar / piano** を直接出す。`app.py` の `run_demucs()` のモデル名を変えるだけ。

- **導入コスト: ほぼゼロ**(既にインストール済みのパッケージの別モデル。初回DLのみ)
- **注意**: 4ステム版より guitar / piano の精度は落ちる。特に **piano ステムの品質は現状あまり良くない**と
  Demucs 側でも言われている。ギターは実用範囲。
- **やり方**: 6stem を既定にし、ギター/ピアノの採譜だけ `guitar.wav` / `piano.wav` を使う。
  他パートは4stem版と同じ扱いで良い。両方走らせると倍の時間がかかるので、6stem一本に寄せるのが現実的。

### S-3. ドラムの強弱(ベロシティ)を付ける

**いま何が問題か**: ドラムのベロシティを onset_strength から雑に決めている。全部同じ強さに見える譜面になりがち。

**代替**: [Enhanced Automatic Drum Transcription via Drum Stem Source Separation](https://arxiv.org/abs/2509.24853)(2025)の後処理が単純で効く。
打楽器ごとに分離したステムに対し、**等ラウドネスフィルタをかけてRMSを計算(1024サンプル窓 / 10msホップ @44.1kHz)→ dB化 → グループ内のピークで正規化**してベロシティにする。

- **導入コスト: 小**(分離ステムがあれば数十行)。同論文はこれで5クラス→7クラス(クラッシュとライドを区別)にも拡張している。

---

## 優先度 A — 効果は大きいが、それなりに手間がかかる

### A-1. ドラム採譜を ADTOF に置き換える

**いま何が問題か**: `transcribe_drums()` は帯域パワーを自分で正規化して閾値で分類している。
合成音源では通ったが、実際の曲では**タムとスネアの取り違え**、シンバルの誤検出が起きる。閾値のチューニングに終わりがない。

**代替**: [ADTOF](https://github.com/MZehren/ADTOF)(ISMIR 2021 / Signals 2023)。
リズムゲームの譜面をクラウドソースして作った359時間のデータセットで学習したモデルで、
**キック・スネア・ハイハット・タム・シンバルの5クラス**を高精度で転写する。

- **導入コスト: 中**。本家は TensorFlow + madmom 依存で Windows だと詰まりやすい。
  → **[ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch)** を使うべき。
  **PyTorchだけで動き**(tensorflow / keras / madmom 不要)、性能低下は約 -0.2% F値のみ。
- **ライセンス注意**: ADTOF本体は **CC BY-NC-SA 4.0(非商用)**。個人の趣味利用なら問題ないが、配布・商用は不可。

### A-2. ドラムのステム分離を挟む(LarsNet)

[LarsNet](https://github.com/polimi-ispl/larsnet)(Toward Deep Drum Source Separation)は、
ステレオのドラムミックスを **U-Netバンクで5ステム(キック/スネア/タム/ハイハット/シンバル)に分離**する。リアルタイムより速い。

- 分離してから各ステムでオンセットを取れば、**そもそも分類問題がなくなる**(どのステムに出たかが答え)。
  S-3のベロシティ推定とも直結する。
- 現行の「1つの波形を帯域で見分ける」方式より原理的に強い。A-1と組み合わせるのが2025年の定石。

### A-3. コードネームを出す

**いま何が問題か**: ギターTAB譜が音符単位でしか出ない。実際にギターで曲をコピーするとき、**まず欲しいのはコード進行**。
音符レベルの採譜が多少荒くても、コードが合っていれば譜面として使える。

**候補**:
- [autochord](https://github.com/cjbayron/autochord) — NNLS-Chroma + Bi-LSTM-CRF。メジャー/マイナー24種+N.C.の25クラス。pip で入る。導入は一番軽い
- [BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) — 双方向Transformer。PyTorch実装。精度は上
- [ChordMini](https://github.com/ptnghia-j/ChordMini) — 疑似ラベル+知識蒸留で BTC を強化した2025年の実装

TAB譜の上にコードネームを並べるだけでも実用性がかなり上がる。MusicXML の `<harmony>` にも書き出せる。

---

## 優先度 B — 大きく作り替えるなら

### B-1. 採譜エンジンを YourMT3+ にする

basic-pitch は「楽器を問わない汎用の音符推定」で、**楽器の区別をしない**。
そのため現行実装は「先にDemucsで分けてから basic-pitch」という構成にしている。

[YourMT3+](https://arxiv.org/abs/2407.04822)(MLSP 2024)は MT3 系列のマルチ楽器転写モデルで、
時間-周波数方向の階層アテンション + Mixture of Experts により、
**パラメータ増加2.5%未満で MT3 と PerceiverTF を大きく上回る**(Instrument Note Onset F1 / Multi F1)。

- [2025 AMT Challenge](https://arxiv.org/abs/2603.27528) では8チーム中2チームが MT3 baseline を上回った。
  同時に、**密なポリフォニー・音色の似た楽器・データの多様性不足**が依然として弱点だと報告されている。
- つまり「分離してから採譜」という現行の方針自体は今も有効。**先に S-2(6stem化)をやる方が費用対効果は高い**。
- CPUでの推論はかなり重い。GPUがある前提の改修。

### B-2. ギターは弦・フレットを直接推定するモデルに変える

現行は「音高を推定 → 動的計画法で弦とフレットを割り当て」の2段構え。
運指のコスト関数(スパン・移動距離・開放弦ボーナス)は自分で決めた経験則で、**実際の弾きやすさのデータに基づいていない**。

**代替**:
- [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf)(ISMIR 2019) — CNNで**フレーム単位の運指を直接推定**。多重ピッチ推定より高精度
- [FretNet](https://arxiv.org/abs/2212.03023) / [実装](https://github.com/cwitkowitz/guitar-transcription-continuous) — 連続値のピッチコンターを音符単位でまとめて出す。チョーキング・ビブラートが表現できる
- [運指の実現可能性と対弦尤度を学習する手法](https://arxiv.org/abs/2204.08094) — 「その押さえ方が物理的に可能か」をデータから学習し、DPのコスト関数を置き換える

**注意**: これらは GuitarSet(**アコギのソロ演奏**)で学習されている。バンドのミックスから分離したギターステムに
そのまま適用するとドメインが合わない。S-2 で `guitar.wav` を作った後に通す前提で考える必要がある。

### B-3. 分離モデルを RoFormer 系にする

[Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) / [BS-RoFormer](https://arxiv.org/abs/2309.02612) は
HTDemucs より明確に上。BS-RoFormer は **vocals 8.3 / bass 7.8 / drums 9.3 / other 4.5 dB SDR**。

- さらに [アンサンブル](https://arxiv.org/abs/2410.20773)(SCNet + Mel-Band RoFormer + HT-Demucs + drumsep)が最高性能で、
  vocals は SDR 13.61 dB に達する。ただし**分離だけで何倍も時間がかかる**
- 実装は [MSST(Music-Source-Separation-Training)](https://arxiv.org/abs/2607.23395) や
  `python-audio-separator` 経由が現実的
- **GPUなしなら手を出さない方がいい**。CPUで1曲十数分が数十分になる

---

## 実装で変えたところ

| ファイル | 内容 |
|---|---|
| `engines.py` | 各処理段のバックエンド登録・利用可否判定・設定の保存 |
| `separate.py` | Demucs各種 + RoFormerハイブリッド。ステム名を揃えて返す |
| `analysis.py` | `estimate_grid_beat_this()` を追加。ダウンビートから拍子も決める |
| `chords.py` | クロマ + ビタビのコード認識(新規・外部依存なし) |
| `transcribe.py` | ADTOFバックエンド、A特性重み付けRMSによるベロシティ推定、外部スクリプト連携 |
| `export.py` | MusicXMLの `<harmony>` 出力、MIDIリーダー |
| `app.py` | ステムのフォールバック(6stem→4stem)、使用エンジンの記録 |
| `setup-gpu.bat` | CUDA版PyTorch + audio-separator を後から入れるスクリプト |

### コード認識を自作した理由

調査では autochord / BTC / ChordMini を候補に挙げたが、実装では**自前のクロマ+ビタビ**にした。

- autochord は TensorFlow と NNLS-Chroma の VAMP プラグインが要る。Windowsで詰まりやすい
- BTC / ChordMini は学習済み重みの入手と配置が手作業になる
- 一方でコード認識は、テンプレート照合 + ビタビ平滑化 + ベース音によるルート補正で
  メジャー/マイナー/7th 程度なら十分に実用的な精度が出る。**追加DLゼロ**の価値が大きい

合成音源のテストでは I-vi-IV-V の進行を 8/8 で当てている。実際の曲では当然落ちるが、
不満が出たら BTC 系に差し替えられるよう、コード認識もエンジンとして分離してある。

### 見送った2つ

- **A-2 LarsNet** — ドラムを5ステムに分離してから採譜する構成。ADTOF を入れた時点で
  分類精度の問題はかなり解消するので、モデル重みの追加DLに見合わないと判断した。
  ADTOFで不満が出たら次はこれ
- **B-2 FretNet / TabCNN** — 弦・フレットを直接推定するモデル。GuitarSet(アコギのソロ)で
  学習されているため、バンドのミックスから分離したギターステムとはドメインが合わない。
  実際に試すなら、まず6stem分離したギター音で当たりを見てからにすべき

## GPUを買ったときにやること

1. `setup-gpu.bat` を実行(CUDA版PyTorch + audio-separator)
2. アプリの「処理エンジンの設定」で切り替える
   - パート分離 → **RoFormer + Demucs 6stem**
   - 拍・ドラム → そのままでOK(自動でGPUを使う)
3. さらに攻めるなら、音符推定を「外部スクリプト」にして YourMT3+ を繋ぐ
   (`config.json` の `pitch_external_cmd` にコマンドを書く)

---

## 参考文献・実装

**拍・ダウンビート**
- Beat This! (ISMIR 2024) — https://github.com/CPJKU/beat_this / https://arxiv.org/abs/2407.21658
- BeatNet (ISMIR 2021) — https://github.com/mjhydri/BeatNet
- madmom — https://github.com/CPJKU/madmom

**音源分離**
- Demucs (htdemucs_6s) — https://github.com/facebookresearch/demucs
- Mel-Band RoFormer — https://arxiv.org/abs/2310.01809
- BS-RoFormer — https://arxiv.org/pdf/2309.02612
- 分離モデルのアンサンブル比較 (2024) — https://arxiv.org/pdf/2410.20773
- MSST 統合フレームワーク — https://arxiv.org/html/2607.23395
- LarsNet (ドラム分離) — https://github.com/polimi-ispl/larsnet / https://arxiv.org/html/2312.09663v3

**採譜(マルチ楽器)**
- YourMT3+ (MLSP 2024) — https://arxiv.org/abs/2407.04822
- MT3 — https://github.com/magenta/mt3 / https://arxiv.org/pdf/2111.03017
- 2025 AMT Challenge — https://arxiv.org/pdf/2603.27528

**ギターTAB**
- TabCNN (ISMIR 2019) — https://archives.ismir.net/ismir2019/paper/000033.pdf
- FretNet — https://arxiv.org/pdf/2212.03023 / https://github.com/cwitkowitz/guitar-transcription-continuous
- 運指の実現可能性・対弦尤度 — https://arxiv.org/pdf/2204.08094
- CRNNによるTAB転写(GuitarSet F1 0.87) — https://github.com/trimplexx/music-transcription

**ドラム採譜**
- ADTOF — https://github.com/MZehren/ADTOF / https://arxiv.org/pdf/2111.11737
- ADTOF-pytorch — https://github.com/xavriley/ADTOF-pytorch
- ドラムステム分離を併用したADT (2025) — https://arxiv.org/abs/2509.24853
- 合成→実音源の転移ギャップ — https://arxiv.org/pdf/2407.19823

**コード認識**
- autochord (ISMIR 2021 LBD) — https://github.com/cjbayron/autochord
- BTC (ISMIR 2019) — https://github.com/jayg996/BTC-ISMIR19
- ChordMini — https://github.com/ptnghia-j/ChordMini
