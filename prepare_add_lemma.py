#!/usr/bin/env python3

"""Adds lemmas to vectors"""

import sys

import pandas as pd


def ud_ignore_tags(tag):
    return tag in ["PROPN", "PUNCT", "X", "SYM", "INTJ"]


def add_lemma(
    input_file,
    lemma_dict_path,
    output_file,
    target_size,
    *_,
    ignore_tags_lambda=ud_ignore_tags,
):
    print("Loading dictionary.")
    # Load the Lemma Dictionary as a pandas DataFrame
    lemma_df = pd.read_csv(lemma_dict_path, sep="\t")
    # lemma_df.columns = ["Word", "Lemma", "Tag", "POS"]

    # Ensure correct columns exist in Lemma Dictionary
    if not set(["Word", "Tag", "POS", "Lemma"]).issubset(lemma_df.columns):
        raise ValueError(
            "Lemma Dictionary is missing one or more of the necessary columns: 'Word', 'Tag', 'POS', 'Lemma'"
        )

    # Create a dictionary for quick lookup of Lemma values
    lemma_dict = lemma_df.set_index("Word")["Lemma"].to_dict()

    # Create a dictionary for quick lookup of Tag values
    tag_dict = lemma_df.set_index("Word")["Tag"].to_dict()

    # Create a dictionary for quick lookup of POS values
    POS_dict = lemma_df.set_index("Word")["POS"].to_dict()

    print("The dictionary loaded.")
    # Load the processed vectors file as a pandas DataFrame
    vectors_df = pd.read_csv(input_file, sep="\t")

    # Ensure correct columns exist in processed vectors
    if not set(["Word"]).issubset(vectors_df.columns):
        raise ValueError(
            "Processed vectors is missing one or more of the necessary columns: 'Word'"
        )

    # Add the 'Lemma' column by mapping from the dictionary
    vectors_df["Lemma"] = vectors_df["Word"].map(lemma_dict)

    # Add the 'Tag' column by mapping from the dictionary
    vectors_df["Tag"] = vectors_df["Word"].map(tag_dict)

    # Add the 'POS' column by mapping from the dictionary
    vectors_df["POS"] = vectors_df["Word"].map(POS_dict)

    # Remove rows with NANs
    nan_selector = vectors_df.isna().any(axis=1)
    nan_rows = vectors_df[nan_selector]
    print(f"skipping {len(nan_rows)} row(s) containing NAN values")
    print(list(nan_rows["Word"]))
    vectors_df = vectors_df[~nan_selector]

    # Remove rows with unwanted POS
    unwanted_POS_selector = vectors_df["POS"].apply(ignore_tags_lambda)
    unwanted_rows = vectors_df[unwanted_POS_selector]
    print(f"skipping {len(unwanted_rows)} row(s) with POS 'Other'")
    print(list(unwanted_rows["Word"]))
    vectors_df = vectors_df[~unwanted_POS_selector]
    print(f"{len(vectors_df)} left")

    # # Remove rows labeled as NNP or NNPS
    # unwanted_tag_selector = (vectors_df["Tag"] == "NNP") | (vectors_df["Tag"] == "NNPS")
    # unwanted_rows = vectors_df[unwanted_tag_selector]
    # print(f"skipping {len(unwanted_rows)} row(s) with Tag NNP or NNPS")
    # print(list(unwanted_rows["Word"]))
    # vectors_df = vectors_df[~unwanted_tag_selector]

    # Reorder the columns to insert 'Lemma' right after 'Word', 'Tag' right after 'Lemma', and 'POS' right after 'Tag'
    columns_order = list(vectors_df.columns)
    columns_order.insert(
        columns_order.index("Word") + 1, columns_order.pop(columns_order.index("Lemma"))
    )
    columns_order.insert(
        columns_order.index("Lemma") + 1, columns_order.pop(columns_order.index("Tag"))
    )
    columns_order.insert(
        columns_order.index("Tag") + 1, columns_order.pop(columns_order.index("POS"))
    )
    vectors_df = vectors_df[columns_order]

    # Select first N rows
    vectors_df = vectors_df.head(target_size)

    # Write the updated dataframe to a new TSV file
    vectors_df.to_csv(output_file, sep="\t", index=False)

    print(f"Updated processed vectors saved to {output_file}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if (ln := len(args)) != 4:
        print(f"incorrect no. of arguments: {ln} found", file=sys.stderr)
        print(f"4 arguments expected:", file=sys.stderr)
        print(f"\t1. lemma dictionary path", file=sys.stderr)
        print(f"\t2. processed vectors path", file=sys.stderr)
        print(f"\t3. output file path", file=sys.stderr)
        print(f"\t4. target size", file=sys.stderr)

        sys.exit(1)

    lemma_dict_path, processed_vectors_path, output_file_path, target_size = args

    target_size = int(target_size)

    add_lemma(processed_vectors_path, lemma_dict_path, output_file_path, target_size)
