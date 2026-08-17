import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "whisper": {
        "repo": "Systran/faster-whisper-large-v3-turbo",
        "dir": "whisper",
        "local": "huggingface",
    },
    "xtts": {
        "repo": "coqui/XTTS-v2",
        "dir": "xtts-v2",
        "local": "local",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VaaniAI model weights")
    parser.add_argument(
        "--root", default="models", help="destination directory for models"
    )
    parser.add_argument(
        "--component",
        choices=["all", "whisper", "xtts"],
        default="all",
    )
    args = parser.parse_args()

    root = Path(args.root)
    for name, spec in MODELS.items():
        if args.component != "all" and args.component != name:
            continue
        dest = root / spec["dir"]
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} -> {dest}")
        snapshot_download(
            repo_id=spec["repo"],
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )

    voice_path = root / "vaani.wav"
    if not voice_path.exists():
        print(
            "\nPlace a clean 10-30s Hindi voice sample of the persona "
            f"at {voice_path} for XTTS voice cloning."
        )


if __name__ == "__main__":
    main()