#!/usr/bin/env python3

import requests
import sys
import os


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        with open(arg, "r") as fs:
            # this way tokens are separated by \n\n (double new line)
            # morphodita reads this as if each token was its own sentence
            # meaning the tagging hopefully shouldn't be contextual
            text = "\n".join(fs.readlines())

        query = {
            "data": text,
            "model": "czech-morfflex2.0-pdtc1.0-220710",
            # "model": "english-morphium-wsj-140407",
            "guesser": "no",
            "input": "vertical",
            "output": "vertical",
        }

        response = requests.post(
            "https://lindat.mff.cuni.cz/services/morphodita/api/tag", data=query
        )

        with open(os.path.join("annotated", arg), "w+") as fs:
            fs.write(
                response.json()["result"]
                .replace('"', "")
                .replace("'", "")
                .replace("\n\n", "\n")
            )
