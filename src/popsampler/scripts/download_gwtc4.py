"""Download GWTC-4.0 population products."""

from __future__ import annotations

import argparse

from popsampler.download import download_gwtc4_bbh, sha256sum


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/gwtc4", help="Local cache directory")
    parser.add_argument("--record-id", type=int, default=16911563)
    parser.add_argument("--force", action="store_true", help="Redownload/reextract existing products")
    parser.add_argument("--no-extract", action="store_true", help="Only download the tarball")
    args = parser.parse_args(argv)

    product = download_gwtc4_bbh(
        args.cache_dir,
        record_id=args.record_id,
        force=args.force,
        extract=not args.no_extract,
    )
    print(f"cache_dir       : {product.cache_dir}")
    print(f"tarball         : {product.tarball}")
    print(f"tarball_sha256  : {sha256sum(product.tarball)}")
    print(f"extracted_dir   : {product.extracted_dir}")
    print(f"default_bbh_file: {product.default_bbh_file}")


if __name__ == "__main__":
    main()
