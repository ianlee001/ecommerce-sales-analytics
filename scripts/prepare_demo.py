"""Copy the bundled public demo artifacts into the local runtime data directory."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "demo" / "processed"
    destination = ROOT / "data" / "processed"
    copied = skipped = 0
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        target = destination / item.relative_to(source)
        if target.exists():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    print(f"Demo ready: copied {copied} files; preserved {skipped} existing files.")
    print(f"Runtime data: {destination}")


if __name__ == "__main__":
    main()
