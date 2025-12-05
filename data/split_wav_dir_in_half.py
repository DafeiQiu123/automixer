#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import soundfile as sf


def split_wav_file(input_path: Path, output_dir: Path) -> bool:
    """Split a WAV file into two equal-duration parts and save as *_1.wav and *_2.wav.

    Returns True if successfully written, False if skipped (e.g., too short).
    """
    info = sf.info(str(input_path))

    # Read entire audio; works for mono or multi-channel.
    data, sample_rate = sf.read(str(input_path), always_2d=False)

    num_frames = data.shape[0]
    if num_frames < 2:
        # Too short to split into two non-empty parts
        return False

    half_frames = num_frames // 2

    # Ensure exactly equal durations by discarding any leftover frame
    first_part = data[:half_frames]
    second_part = data[half_frames : half_frames * 2]

    base = input_path.stem
    out1 = output_dir / f"{base}_1.wav"
    out2 = output_dir / f"{base}_2.wav"

    subtype = info.subtype or "PCM_16"

    sf.write(str(out1), first_part, sample_rate, subtype=subtype)
    sf.write(str(out2), second_part, sample_rate, subtype=subtype)

    return True


def process_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> Tuple[int, int]:
    """Process all .wav files in input_dir, writing results to output_dir.

    Returns (num_processed, num_skipped)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".wav":
            continue

        out1 = output_dir / f"{path.stem}_1.wav"
        out2 = output_dir / f"{path.stem}_2.wav"

        if not overwrite and out1.exists() and out2.exists():
            skipped += 1
            continue

        try:
            ok = split_wav_file(path, output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to split {path.name}: {exc}")
            skipped += 1
            continue

        if ok:
            processed += 1
        else:
            skipped += 1

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split all WAV files in a directory into two equal-duration halves. "
            "Outputs are named with _1 and _2 suffixes."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/wav_dir_trimmed"),
        help="Directory containing source WAV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/wav_dir_trimmed_split"),
        help="Directory to write split WAV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_1.wav and *_2.wav files if present.",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    processed, skipped = process_directory(args.input_dir, args.output_dir, args.overwrite)

    print(
        f"Done. Processed: {processed}, Skipped: {skipped}. Output: {args.output_dir}"  # noqa: T201
    )


if __name__ == "__main__":
    main()
