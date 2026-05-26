#!/usr/bin/env python
"""Step 06: stream the Zenodo tarball, extract one HDF5 file, and inspect it.

This is designed for a GitHub Actions runner. It avoids storing the full
`analyses_BBH.tar` tarball on disk: the tar stream is read sequentially and only
the requested `.h5` member is extracted.

The output directory contains compact metadata reports only. The extracted HDF5
file is not uploaded as an artifact by the workflow.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm.auto import tqdm


ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"
DEFAULT_RECORD_ID = 16911563
DEFAULT_TARBALL = "analyses_BBH.tar"
DEFAULT_H5_BASENAME = "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5"


def get_zenodo_file_url(record_id: int, filename: str) -> tuple[str, int | None]:
    response = requests.get(ZENODO_RECORD_API.format(record_id=record_id), timeout=60)
    response.raise_for_status()
    record = response.json()
    for item in record.get("files", []):
        if item.get("key") == filename:
            return (
                item.get("links", {}).get("self")
                or f"https://zenodo.org/records/{record_id}/files/{quote(filename)}?download=1",
                item.get("size"),
            )
    available = [item.get("key") for item in record.get("files", [])]
    raise FileNotFoundError(f"Could not find {filename!r}. Available: {available}")


def stream_extract_member(url: str, member_basename: str, output_dir: Path, expected_size: int | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    if expected_size is None:
        header = response.headers.get("content-length")
        expected_size = int(header) if header and header.isdigit() else None

    # Wrap raw reads to update a download progress bar while tarfile consumes bytes.
    bar = tqdm(total=expected_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Streaming tar")
    raw = response.raw
    raw.decode_content = True
    original_read = raw.read

    def read_with_progress(*args, **kwargs):
        data = original_read(*args, **kwargs)
        if data:
            bar.update(len(data))
        return data

    raw.read = read_with_progress  # type: ignore[method-assign]

    found: Path | None = None
    try:
        with tarfile.open(fileobj=raw, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if Path(member.name).name != member_basename:
                    continue
                target = output_dir / Path(member.name).name
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not extract tar member {member.name}")
                with target.open("wb") as fp:
                    shutil.copyfileobj(source, fp, length=1024 * 1024)
                found = target
                break
    finally:
        bar.close()
        response.close()

    if found is None:
        raise FileNotFoundError(f"Could not find member basename {member_basename!r} in streamed tar")
    return found


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", type=int, default=DEFAULT_RECORD_ID)
    parser.add_argument("--tarball", default=DEFAULT_TARBALL)
    parser.add_argument("--h5-basename", default=DEFAULT_H5_BASENAME)
    parser.add_argument("--workdir", default="runner_h5_work")
    parser.add_argument("--outdir", default="runner_h5_report")
    parser.add_argument("--keep-h5", action="store_true", help="Do not delete extracted HDF5 file at the end")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    url, size = get_zenodo_file_url(args.record_id, args.tarball)
    print(f"Zenodo tarball URL: {url}")
    print(f"Zenodo tarball size: {size}")

    h5_path = stream_extract_member(url, args.h5_basename, workdir, expected_size=size)
    print(f"Extracted HDF5: {h5_path}")
    print(f"Extracted HDF5 size: {h5_path.stat().st_size}")

    # Reuse the normal validation scripts so local and runner results are comparable.
    run([sys.executable, "validation/01_inspect_release.py", "--h5", str(h5_path), "--outdir", str(outdir)])
    run([sys.executable, "validation/05_inspect_h5_metadata.py", "--h5", str(h5_path), "--outdir", str(outdir)])

    with (outdir / "runner_download_summary.txt").open("w") as fp:
        fp.write(f"record_id: {args.record_id}\n")
        fp.write(f"tarball: {args.tarball}\n")
        fp.write(f"tarball_size: {size}\n")
        fp.write(f"h5_basename: {args.h5_basename}\n")
        fp.write(f"h5_size: {h5_path.stat().st_size}\n")
        fp.write(f"h5_path_on_runner: {h5_path}\n")

    if not args.keep_h5:
        os.remove(h5_path)
        print("Deleted extracted HDF5 after writing reports")


if __name__ == "__main__":
    main()
