#!/usr/bin/env python3

import sys

import pandas as pd


def add_POS_labeling(lemma_dict_path, POS_mapping_path, output_path):
    lemma_dict_df = pd.read_csv(lemma_dict_path, sep="\t")
    # lemma_dict_df.columns = ['Word', 'Lemma', 'Tag']
    # POS_mapping_df = pd.read_csv(POS_mapping_path, sep="\t")

    # mapping = POS_mapping_df.set_index("Tag")["POS"].to_dict()

    # lemma_dict_df["POS"] = lemma_dict_df["Tag"].map(mapping)

    lemma_dict_df['POS'] = lemma_dict_df['Tag'].str[0]

    lemma_dict_df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]

    if (ln := len(args)) != 3:
        print(f"incorrect no. of arguments: {ln} found", file=sys.stderr)
        print(f"3 arguments expected:", file=sys.stderr)
        print(f"\t1. word-lemma dictionary path", file=sys.stderr)
        print(f"\t2. POS mapping path", file=sys.stderr)
        print(f"\t3. output file path", file=sys.stderr)

        sys.exit(1)

    add_POS_labeling(*args)
