#!/usr/bin/env python3

import requests
import sys
import os
import pandas as pd


def morphodita(filename, model_name="czech-morfflex2.0-pdtc1.0-220710"):
    with open(filename, "r") as fs:
        # this way tokens are separated by \n\n (double new line)
        # morphodita reads this as if each token was its own sentence
        # meaning the tagging hopefully shouldn't be contextual
        text = "\n".join(fs.readlines())

    query = {
        "data": text,
        "model": model_name,
        # "model": "english-morphium-wsj-140407",
        "guesser": "no",
        "input": "vertical",
        "output": "vertical",
    }

    response = requests.post(
        "https://lindat.mff.cuni.cz/services/morphodita/api/tag", data=query
    )

    with open(os.path.join("annotated", filename), "w+") as fs:
        fs.write(
            response.json()["result"]
            .replace('"', "")
            .replace("'", "")
            .replace("\n\n", "\n")
        )


def upipe(filename, model_name="czech-ud-1.2-160523"):
    with open(filename, "r") as fs:
        text = "\n".join(fs.readlines())

    query = {
        "data": text,
        "model": model_name,
        "tagger": "yes",
        "input": "vertical",
        "output": "conllu",
    }

    response = requests.post(
        "https://lindat.mff.cuni.cz/services/udpipe/api/process", data=query
    )

    with open(os.path.join("annotated", filename), "w+") as fs:
        lines = [x for x in response.json()["result"].split("\n") if len(x) > 0 and x[0] == '1']
        rows = [line.split("\t") for line in lines]
        df = pd.DataFrame(rows, columns=["id", "Word", "Lemma", "POS", "Tag", "Feats", "head", "deprel", "deps", "misc"])
        df = df[["Word", "Lemma", "POS", "Tag", "Feats"]]
        fs.write(df.to_csv(sep="\t", index=False))




if __name__ == "__main__":
    for arg in sys.argv[1:]:
        morphodita(arg)
