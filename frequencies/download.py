#!/usr/bin/env python3

import os
import requests
import zipfile

dname = os.path.dirname(os.path.realpath(__file__))


def get_zip(url):
    print(f"Download {url}")
    req = requests.get(url)

    bname = os.path.basename(url)
    fname = os.path.join(dname, bname)
    print(f"Write to {fname}")
    with open(fname, "wb") as f:
        f.write(req.content)

    edirname = os.path.splitext(fname)[0]
    print(f"Extract {bname} to {edirname}")
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(edirname)


if __name__ == "__main__":
    get_zip("http://milicka.cz/sklad/sorted.zip")
    get_zip("http://milicka.cz/sklad/flat.zip")
