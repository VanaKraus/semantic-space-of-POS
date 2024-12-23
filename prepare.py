#!/usr/bin/env python3

import sys

import pandas as pd
import re


pos_legend = {
    "N": ("substantivum", "Noun"),
    "A": ("adjektivum", "Adjective"),
    "P": ("zájmeno", "Pronoun"),
    "C": ("číslovka", "Numeral"),
    "V": ("sloveso", "Verb"),
    "D": ("adverbium", "Adverb"),
    "R": ("předložka", "Preposition"),
    "J": ("spojka", "Conjunction"),
    "T": ("částice", "Particle"),
    "I": ("citoslovce", "Interjection"),
}


def get_pos_info(tag):
    pos_code = tag[0] if tag else None
    pos_czech, pos_english = pos_legend.get(pos_code, (None, None))
    return pos_code, pos_czech, pos_english


# MAX_ROWS  Maximum number of words to keep, the rest will be truncated


def prepare_vectors(input_file, output_file, MAX_ROWS=1000):
    data = []

    # Load the dataset and remove the first line
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            linesplt = line.strip().split()

            if len(linesplt) not in (101, 0):
                print(linesplt)
            else:
                data += [linesplt]

    # Limit the number of rows to MAX_ROWS + some overhead for interpunction that will be filtered out later
    data = data[: int(MAX_ROWS * 1.5)]

    # Determine the number of columns dynamically based on the first data row
    num_vector_columns = len(data[0]) - 1

    # Prepare the header
    header = ["Word"] + [f"D{i}" for i in range(1, num_vector_columns + 1)]
    df = pd.DataFrame(data, columns=header)

    # Save the modified DataFrame to a new TSV file
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

    print(f"Processing complete. The output has been saved as '{output_file}'.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if (ln := len(args)) != 3:
        print(f"incorrect no. of arguments: {ln} found", file=sys.stderr)
        print(f"3 arguments expected:", file=sys.stderr)
        print(f"\t1. original vectors", file=sys.stderr)
        print(f"\t2. output file path", file=sys.stderr)
        print(f"\t3. max no. of rows", file=sys.stderr)

        sys.exit(1)

    input_file, output_file, max_rows = args
    max_rows = int(max_rows)

    prepare_vectors(input_file, output_file, max_rows)
