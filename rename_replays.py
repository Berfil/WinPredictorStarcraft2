"""
Rename all .SC2Replay files in the replays folder to sequential numbers.
Already-numbered files (e.g. 00001.SC2Replay) are kept in place.
New files get the next available numbers.
Run this script each time you add new replays.
"""

from pathlib import Path

REPLAY_DIR = Path(__file__).parent / "replays"
EXTENSION = ".SC2Replay"


def already_numbered(name: str) -> bool:
    return name.removesuffix(EXTENSION).isdigit()


def rename_replays():
    all_files = sorted(REPLAY_DIR.glob(f"**/*{EXTENSION}"), key=lambda p: p.name)

    # Find the highest existing number
    existing_numbers = set()
    for f in all_files:
        if already_numbered(f.name):
            existing_numbers.add(int(f.stem))

    next_num = max(existing_numbers, default=0) + 1

    renamed = 0
    for f in all_files:
        if already_numbered(f.name):
            continue
        new_name = f.parent / f"{next_num:05d}{EXTENSION}"
        f.rename(new_name)
        next_num += 1
        renamed += 1

    print(f"Renamed {renamed} files. Total replays: {len(list(REPLAY_DIR.glob(f'**/*{EXTENSION}')))}")


if __name__ == "__main__":
    rename_replays()
