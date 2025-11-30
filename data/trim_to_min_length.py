import os
import argparse
from typing import List, Tuple

import soundfile as sf


def list_wav_files(directory: str) -> List[str]:
    return sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".wav")
        ]
    )


def gather_durations(files: List[str]) -> List[Tuple[str, int, int, float]]:
    """
    Returns list of (path, frames, sr, seconds)
    """
    out = []
    for p in files:
        try:
            info = sf.info(p)
            frames = int(info.frames)
            sr = int(info.samplerate)
            seconds = frames / max(sr, 1)
            out.append((p, frames, sr, seconds))
        except Exception as e:
            print(f"[WARN] Failed to read info for {p}: {e}")
    return out


def trim_to_last_segment(
    path: str, out_path: str, keep_seconds: float
) -> None:
    """
    Trim the audio to the last keep_seconds of content and save.
    Preserves original samplerate and channel layout.
    """
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception as e:
        print(f"[WARN] Read failed: {path}: {e}")
        return

    if audio is None:
        print(f"[WARN] Empty audio: {path}")
        return

    # Determine sample count to keep in this file's samplerate
    keep_samples = int(round(keep_seconds * sr))

    # audio shape can be (N,) or (N, C)
    total = audio.shape[0]
    if keep_samples >= total:
        trimmed = audio  # already shorter or equal
    else:
        start = total - keep_samples
        trimmed = audio[start:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        sf.write(out_path, trimmed, sr)
    except Exception as e:
        print(f"[WARN] Write failed: {out_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Trim all WAVs in a folder to the shortest duration (take last segment)."
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "wav_dir")),
        help="Input WAV directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "wav_dir_trimmed")),
        help="Output directory (non-destructive). Use --in_place to overwrite.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite input files in place (dangerous). If set, out_dir is ignored.",
    )
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    in_place = args.in_place

    files = list_wav_files(in_dir)
    if not files:
        print(f"[ERROR] No WAV files found in: {in_dir}")
        return

    meta = gather_durations(files)
    if not meta:
        print(f"[ERROR] No valid WAV metadata found in: {in_dir}")
        return

    # Use seconds to accommodate mixed sample rates
    min_seconds = min(m[3] for m in meta)
    print(f"[INFO] Found {len(meta)} files.")
    print(f"[INFO] Shortest duration (seconds): {min_seconds:.6f}")

    if in_place:
        print("[INFO] In-place trimming enabled. Original files will be overwritten.")
    else:
        print(f"[INFO] Writing trimmed files to: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

    for p, frames, sr, secs in meta:
        rel = os.path.basename(p)
        dst = p if in_place else os.path.join(out_dir, rel)
        print(f"[Trim] {rel}: {secs:.3f}s -> {min_seconds:.3f}s")
        trim_to_last_segment(p, dst, keep_seconds=min_seconds)

    print("[DONE] Trimming complete.")


if __name__ == "__main__":
    main()


