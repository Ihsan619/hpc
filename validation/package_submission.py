"""Package each assessed task with its sources, public inputs and notebook."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
destination = root / "submission"
destination.mkdir(exist_ok=True)
for number in range(1, 5):
    target = destination / f"Task_{number:02d}.zip"
    paths = list((root / f"task{number}").rglob("*"))
    paths += [root / "Makefile", root / "README.md",
              root / "notebooks" / f"Task_{number:02d}.ipynb",
              root / "docs/input_manifest.json", root / "docs/viva_guide.md"]
    paths += list((root / "validation").glob("*.py"))
    paths += list((root / "validation").glob("*.txt"))
    paths += list((root / "evidence").glob(f"task{number}_*"))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if path.is_file() and path.suffix not in {".o", ".exe"}:
                if path.suffix or path.name in {"Makefile"}:
                    archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    print(target, target.stat().st_size, "bytes")
