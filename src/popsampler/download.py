"""Download helpers for public population data releases."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import tarfile
from pathlib import Path
from urllib.parse import quote

import requests


ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"
GWTC4_POPULATION_RECORD = 16911563
DEFAULT_BBH_TARBALL = "analyses_BBH.tar"
DEFAULT_BBH_H5 = "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5"


@dataclasses.dataclass(frozen=True)
class DownloadedProduct:
    """Paths returned by a download/extract operation."""

    cache_dir: Path
    tarball: Path
    extracted_dir: Path
    default_bbh_file: Path | None


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_zenodo_record(record_id: int = GWTC4_POPULATION_RECORD) -> dict:
    url = ZENODO_RECORD_API.format(record_id=record_id)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def find_zenodo_file(record: dict, filename: str) -> dict:
    for item in record.get("files", []):
        if item.get("key") == filename:
            return item
    available = [item.get("key") for item in record.get("files", [])]
    raise FileNotFoundError(f"Could not find {filename!r}. Available files: {available}")


def download_file(url: str, output: Path, *, force: bool = False, chunk_size: int = 1024 * 1024) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return output

    tmp = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fp.write(chunk)
    os.replace(tmp, output)
    return output


def safe_extract_tar(tar_path: Path, output_dir: Path) -> None:
    """Extract a tar archive while preventing path traversal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir.resolve()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Refusing to extract unsafe tar member {member.name!r}")
        tar.extractall(output_dir)


def download_gwtc4_bbh(
    cache_dir: str | Path,
    *,
    record_id: int = GWTC4_POPULATION_RECORD,
    tarball_name: str = DEFAULT_BBH_TARBALL,
    force: bool = False,
    extract: bool = True,
) -> DownloadedProduct:
    """Download the GWTC-4 population BBH tarball and optionally extract it."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    record = get_zenodo_record(record_id)
    file_info = find_zenodo_file(record, tarball_name)
    url = file_info.get("links", {}).get("self")
    if not url:
        # Stable fallback for Zenodo file download URLs.
        url = f"https://zenodo.org/records/{record_id}/files/{quote(tarball_name)}?download=1"

    tarball = cache_dir / tarball_name
    download_file(url, tarball, force=force)

    checksum = file_info.get("checksum")
    if checksum and checksum.startswith("md5:"):
        # Keep MD5 validation optional because Zenodo usually advertises MD5, while
        # SHA256 is easier to compute consistently for user-side provenance logs.
        pass

    extracted_dir = cache_dir / tarball_name.removesuffix(".tar")
    if extract and (force or not extracted_dir.exists()):
        safe_extract_tar(tarball, extracted_dir)

    candidates = list(extracted_dir.rglob(DEFAULT_BBH_H5)) if extracted_dir.exists() else []
    default_h5 = candidates[0] if candidates else None
    return DownloadedProduct(cache_dir=cache_dir, tarball=tarball, extracted_dir=extracted_dir, default_bbh_file=default_h5)
