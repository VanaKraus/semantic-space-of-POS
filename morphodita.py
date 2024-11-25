#!/usr/bin/env python3

import requests
import sys
import os


def morphodita(filename, model_name= "czech-morfflex2.0-pdtc1.0-220710"):
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


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        morphodita(arg)
