#!/usr/bin/env python3

import os
import requests
import zipfile

import pandas as pd

dname = os.path.dirname(os.path.realpath(__file__))


def get_zip(url, down_dir=None):
    if down_dir is None:
        down_dir = dname

    print(f"Download {url}")
    req = requests.get(url)

    bname = os.path.basename(url)
    fname = os.path.join(down_dir, bname)
    print(f"Write to {fname}")
    with open(fname, "wb") as f:
        f.write(req.content)

    edirname = os.path.splitext(fname)[0]
    print(f"Extract {bname} to {edirname}")
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(edirname)


def sorted_df(path: str, lines: int | None = None) -> pd.DataFrame:
    return (
        pd.read_csv(path, sep="\t", nrows=lines)
        if lines
        else pd.read_csv(path, sep="\t")
    )


def sorted_df_pivot_long(df: pd.DataFrame) -> pd.DataFrame:
    mdf = df.copy()

    mdf["Variants"] = mdf["Variants"].str.split(",")
    mdf = mdf.explode("Variants", ignore_index=True)

    mdf[["POS_Lemma", "Frequency"]] = mdf["Variants"].str.extract(r"^(.+:?):(.+)$")
    mdf[["POS", "Lemma"]] = mdf["POS_Lemma"].str.extract(r"^([A-Z]+)_(.+)$")
    mdf = mdf.astype(
        {"Frequency": "int64", "Word": "str", "POS": "str", "Lemma": "str"}
    )

    mdf.drop(columns=["Variants"], inplace=True)

    return mdf


def sorted_df_filter_POS(df: pd.DataFrame, POS_stoplist: list[str]) -> pd.DataFrame:
    return df.loc[-df["POS"].isin(POS_stoplist)].reset_index(drop=True)


def sorted_df_widen_filtered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(Freq_Sum=df.groupby("Word")["Frequency"].transform("sum"))
    df.drop(columns=["Total Frequency"], inplace=True)

    df = (
        df.assign(
            Lemmas=df.groupby("Word")["POS_Lemma"].transform(lambda x: ",".join(set(x)))
        )
        .groupby(["Word", "POS", "Freq_Sum", "Lemmas"])["Frequency"]
        .sum()
        .reset_index()
        .sort_values(by=["Freq_Sum", "Word", "Frequency"], ascending=False)
    )

    df = df.assign(POS_Top=df.groupby("Word")["POS"].transform(lambda x: list(x)[0]))

    df["Frequency_Relative"] = df["Frequency"] / df["Freq_Sum"]

    df = (
        df.pivot(
            index=["Word", "Freq_Sum", "POS_Top", "Lemmas"],
            columns="POS",
            values="Frequency_Relative",
        )
        .sort_values(by=["Freq_Sum"], ascending=False)
        .reset_index()
    )

    df.columns.names = [None]

    df = df.drop(columns=["nan"]).fillna(0)

    return df


def compile_sorted_df(
    path: str, lines: int | None = None, POS_stoplist: list[str] | None = None
) -> pd.DataFrame:
    df = sorted_df(path, lines)
    df = sorted_df_pivot_long(df)
    if POS_stoplist:
        df = sorted_df_filter_POS(df, POS_stoplist)
    df = sorted_df_widen_filtered(df)

    return df


if __name__ == "__main__":
    get_zip("http://milicka.cz/sklad/sorted.zip")
    get_zip("http://milicka.cz/sklad/flat.zip")
