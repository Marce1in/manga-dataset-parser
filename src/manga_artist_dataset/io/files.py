"""Filesystem helpers shared by pipeline stages."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory.

    Example:
        `digest = sha256_file(Path("page.png"))`.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recreate_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally replacing existing contents.

    Example:
        `recreate_dir(Path("artifacts/out"), overwrite=True)`.
    """
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            msg = f"{path} is not empty; pass --overwrite to rebuild it."
            raise FileExistsError(msg)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def workspace_path(path: Path) -> str:
    """Render paths relative to the workspace when possible.

    Example:
        `workspace_path(Path.cwd() / "README.md") == "README.md"`.
    """
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
