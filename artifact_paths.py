"""Portable locations for read-only evidence and generated outputs."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def portable(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return "external/" + path.name

def output_directory(path):
    path = Path(path).resolve()
    for frozen in (ROOT / "results", ROOT / "source_snapshots"):
        if path == frozen or frozen in path.parents:
            raise ValueError("Derived outputs must be outside frozen evidence: " + portable(path))
    path.mkdir(parents=True, exist_ok=True)
    return path

def validation_path(name):
    return output_directory(ROOT / "output/validation") / (name + ".json")
