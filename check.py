"""環境診断ツール — run.bat がうまく動かないときに実行する"""
import socket
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OK, NG, WARN = "[OK]  ", "[NG]  ", "[WARN]"


def main():
    print("=" * 56)
    print("  TAB Maker  環境診断")
    print("=" * 56)

    # [1] Python
    v = sys.version_info
    print(f"\n[1] Python {v.major}.{v.minor}.{v.micro}")
    if (3, 10) <= (v.major, v.minor) <= (3, 12):
        print(OK, "対応バージョンです")
    else:
        print(WARN, "3.10〜3.12 を推奨します(librosa/demucs が入らないことがあります)")

    # [2] ライブラリ
    print("\n[2] ライブラリ")
    core = ["flask", "numpy", "scipy", "librosa", "soundfile", "yt_dlp", "demucs", "torch"]
    missing = []
    for name in core:
        try:
            __import__(name)
            print(OK, name)
        except Exception as e:
            print(NG, name, "->", type(e).__name__, e)
            missing.append(name)
    optional = [
        # 実際に使うモジュールまで import して確かめる
        ("basic_pitch.inference", "音符推定AI (basic-pitch)", "setup.bat"),
        ("beat_this.inference", "拍・小節頭の推定 (Beat This!)", "setup.bat"),
        ("adtof_pytorch", "ドラム採譜AI (ADTOF)", "setup.bat + Git"),
        ("audio_separator", "RoFormer分離 (GPU向け)", "setup-gpu.bat"),
    ]
    for name, what, how in optional:
        try:
            __import__(name)
            print(OK, f"{name} — {what}")
        except Exception as e:
            print(WARN, f"{what} は使えません ({how} で導入)")
            print("      理由:", type(e).__name__, str(e)[:220])

    # [2a] numpy / scipy の組み合わせ (よくある落とし穴)
    print("\n[2a] numpy と scipy の相性")
    try:
        import numpy
        import scipy
        print("      numpy", numpy.__version__, "/ scipy", scipy.__version__)
        import librosa  # noqa: F401
        print(OK, "librosa が正常に読み込めます")
    except Exception as e:
        print(NG, type(e).__name__, str(e)[:300])
        print("      → basic-pitch が入れる TensorFlow が numpy を 1系に固定する一方で、")
        print("        numpy 2系向けにビルドされた scipy が入っていると衝突します。")
        print("        修復コマンド (このフォルダで実行):")
        print('          venv\\Scripts\\python.exe -m pip install --force-reinstall "numpy<2" "scipy<1.14"')

    # [2b] エンジン設定
    print("\n[2b] 使用するエンジン")
    try:
        sys.path.insert(0, str(BASE))
        import engines
        d = engines.describe()
        print("      デバイス:", d["device"], "(CUDA:", d["cuda"], ")")
        for stage, s in d["stages"].items():
            print(f"      {s['label']}: {s['active']}")
    except Exception as e:
        print(NG, "エンジン情報の取得に失敗:", e)

    # [3] ffmpeg
    print("\n[3] ffmpeg")
    try:
        sys.path.insert(0, str(BASE))
        import app as appmod
        p = appmod.ffmpeg_exe()
        print(OK, p)
    except Exception as e:
        print(NG, e)

    # [4] ポート
    print("\n[4] ポート 8766")
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 8766))
        print(OK, "空いています")
    except OSError:
        print(WARN, "使用中です(すでに起動済みかもしれません)")
    finally:
        s.close()

    # [5] パス長
    print("\n[5] フォルダのパス")
    print("     ", BASE, f"({len(str(BASE))}文字)")
    if len(str(BASE)) > 80:
        print(NG, "長すぎます。C:\\tabmaker などに移動してください")
    else:
        print(OK, "問題ありません")

    # [6] 直近のエラーログ
    log = BASE / "server.log"
    print("\n[6] server.log の末尾")
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-15:]:
            print("     ", line)
    else:
        print("      (まだありません)")

    if missing:
        print("\n>>> 不足ライブラリがあります。setup.bat を再実行してください:", ", ".join(missing))
    print("\n" + "=" * 56)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    input("\nEnterキーで閉じます...")
