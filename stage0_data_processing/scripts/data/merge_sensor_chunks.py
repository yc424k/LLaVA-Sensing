import argparse
import json
import pathlib


def load_chunks(folder: pathlib.Path):
    data = []
    for path in sorted(folder.glob("*_chunk_*.json")):
        with path.open("r", encoding="utf-8") as f:
            data.extend(json.load(f))
    return data


def main():
    parser = argparse.ArgumentParser("merge sensor-literature chunked datasets")
    parser.add_argument("folders", nargs="+", help="Directories that contain chunked json files")
    parser.add_argument("--output", required=True, help="Path for merged json output")
    args = parser.parse_args()

    merged = []
    for folder in args.folders:
        merged.extend(load_chunks(pathlib.Path(folder)))

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
