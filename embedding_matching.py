import pandas as pd

DIMENSIONS = 100
ENCODING = "latin-1"  # UTF-8 doesn't work for Czech embeddings for some reason


def match_embeddings(
    embeddings_path, words_path, out_tsv, target_size: int, encoding=ENCODING
):
    def log_progress(line, filled):
        print(f"{line} lines w/ {filled} embeddings filled")

    df = pd.read_csv(words_path, sep="\t", encoding=encoding)
    word_list = list(df["Word"])
    word_set = set(word_list)

    dimensions_selector = [f"D{i+1}" for i in range(DIMENSIONS)]

    with open(embeddings_path, "r", encoding=encoding) as f:
        ctr_lines, ctr_df = 0, 0
        for line in f:
            ctr_lines += 1
            if ctr_lines % 10000 == 0:
                log_progress(ctr_lines, ctr_df)

            linespl = line.strip().split(" ")

            if len(linespl) != DIMENSIONS + 1:
                continue

            word = linespl[0]
            if word not in word_set:
                continue

            windex = word_list.index(word)
            df.loc[windex, dimensions_selector] = linespl[1:]

            ctr_df += 1
            if ctr_df % 50 == 0:
                log_progress(ctr_lines, ctr_df)

    notna_selector = df.notna().all(axis=1)
    print(f"{int(notna_selector.sum())} valid words matched ({target_size=})")

    df = (
        df.loc[notna_selector]
        .sort_values(by=["Freq_Sum"], ascending=False)[:target_size]
        .reset_index(drop=True)
    )

    df.to_csv(out_tsv, sep="\t", encoding=ENCODING, index=False)

    return df.astype({s: "float" for s in dimensions_selector})
