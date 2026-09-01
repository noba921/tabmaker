# サードパーティのライセンスについて

TAB Maker 本体は MIT ライセンスですが、実行時に利用する外部ソフトウェア・
学習済みモデルにはそれぞれ別の条件があります。**特に非商用条件のものが含まれます。**

## 必須の依存

| ソフトウェア | ライセンス | 備考 |
|---|---|---|
| [Demucs](https://github.com/facebookresearch/demucs) | MIT | 学習済みモデルの重みを含む |
| [librosa](https://github.com/librosa/librosa) | ISC | |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Unlicense | |
| [Flask](https://flask.palletsprojects.com/) | BSD-3-Clause | |
| [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) | BSD-3-Clause | |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | |
| [FFmpeg](https://ffmpeg.org/) (imageio-ffmpeg 同梱) | LGPL 2.1+ / GPL | ビルドにより異なる |

## 任意の依存(セットアップ時に導入を試みるもの)

| ソフトウェア | ライセンス | 備考 |
|---|---|---|
| [basic-pitch](https://github.com/spotify/basic-pitch) | Apache-2.0 | Spotify |
| [Beat This!](https://github.com/CPJKU/beat_this) | MIT | 学習済みモデルの重みも MIT |
| **[ADTOF](https://github.com/MZehren/ADTOF)** | **CC BY-NC-SA 4.0** | **非商用のみ。商用利用時は無効化すること** |
| [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) | 本家に準じる | ADTOFのPyTorch移植 |
| [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | MIT | モデルは [UVR](https://github.com/Anjok07/ultimatevocalremovergui) 由来。UVRへのクレジットが必要 |

### ADTOF を使う場合の注意

ドラム採譜エンジンに ADTOF を選ぶと、CC BY-NC-SA 4.0 の学習済みモデルを利用します。

- **非商用利用に限られます**
- 商用利用の必要がある場合は、アプリの「処理エンジンの設定」で
  ドラムを「内蔵ヒューリスティック」に変更してください(精度は落ちますが制約はありません)

## 生成物の取り扱い

本ソフトウェアは、利用者が用意した音源から譜面を生成する道具です。
**入力した音源と、そこから生成された譜面の権利は元の楽曲の権利者に属します。**

- 生成した譜面の再配布・販売は行わないでください
- 音源のダウンロードは、各サービスの利用規約と居住国の法令に従ってください
- 私的利用の範囲を超える使い方について、作者は一切の責任を負いません

## 引用文献

実装が参考にしている主な研究:

- F. Foscarin, J. Schlüter, G. Widmer. "Beat this! Accurate beat tracking without DBN
  postprocessing." ISMIR 2024.
- M. Zehren, M. Alunno, P. Bientinesi. "ADTOF: A large dataset of non-synthetic music
  for automatic drum transcription." ISMIR 2021.
- R. M. Bittner et al. "A lightweight instrument-agnostic model for polyphonic note
  transcription and multipitch estimation." ICASSP 2022. (basic-pitch)
- S. Rouard, F. Massa, A. Défossez. "Hybrid Transformers for Music Source Separation."
  ICASSP 2023. (Demucs v4)
- ドラムのベロシティ推定の後処理は arXiv:2509.24853 の考え方に基づく
