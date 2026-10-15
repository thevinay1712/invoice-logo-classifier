"""CLI script to download, inspect schema, print row counts, and sample 5 example rows from LogoDet-3K."""

import sys

from PIL import Image

from invoice_logo_classifier.data.download import download_and_cache, inspect_schema


def main():
    print("Initializing LogoDet-3K dataset inspection...")

    try:
        ds_dict = download_and_cache()
    except RuntimeError as err:
        print(f"\n[ERROR] Failed to load dataset: {err}", file=sys.stderr)
        sys.exit(1)

    schema_info = inspect_schema(ds_dict)

    print("-" * 60)
    print(" ROW COUNTS & SPLIT BREAKDOWN")
    print("-" * 60)
    total_rows = 0
    for split_name, ds_split in ds_dict.items():
        count = len(ds_split)
        total_rows += count
        print(f"  - Split '{split_name}': {count:,} rows")
    print(f"  TOTAL ROWS: {total_rows:,}")
    print("-" * 60)

    # Inspect 5 example rows from the first split
    first_split_name = schema_info["splits"][0]
    first_split = ds_dict[first_split_name]
    num_samples = min(5, len(first_split))

    print(f"\nPRINTING FIRST {num_samples} SAMPLE ROWS (from '{first_split_name}' split):\n")

    for idx in range(num_samples):
        row = first_split[idx]
        formatted_row = {}

        for key, val in row.items():
            if isinstance(val, Image.Image):
                formatted_row[key] = f"<PIL.Image size={val.size[0]}x{val.size[1]} mode={val.mode}>"
            elif hasattr(val, "shape"):
                formatted_row[key] = f"<ndarray shape={val.shape}>"
            else:
                formatted_row[key] = repr(val)

        print(f"Row {idx + 1}:")
        for k, v in formatted_row.items():
            print(f"   {k}: {v}")
        print()

    print("=" * 60)
    print(" Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
