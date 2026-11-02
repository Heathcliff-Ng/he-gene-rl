#!/usr/bin/env python3
"""Build a source-only ZIP; never include data, local weights, or predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT_FILES = {"README.md", "README.zh-CN.md", "pyproject.toml", "MANIFEST.in",
              ".gitignore", ".gitattributes", "RELEASE_STATUS.txt", "LICENSE", "LICENSE.txt"}
ROOT_DIRS = {"src", "tests", "configs", "docs", "examples", "scripts", "reports", ".github"}
SOURCE_SUFFIXES = {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".toml", ".in"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing archive: {output}")
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink() or "__pycache__" in relative.parts:
            continue
        if ((len(relative.parts) == 1 and relative.name in ROOT_FILES)
                or (relative.parts[0] in ROOT_DIRS and path.suffix in SOURCE_SUFFIXES)):
            files.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, (Path("he-gene-rl") / path.relative_to(root)).as_posix())
    print(f"Wrote {len(files)} source/documentation files: {output}")


if __name__ == "__main__":
    main()
