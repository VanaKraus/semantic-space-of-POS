import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from .chart import make_chart


def create_umap(
    feature_names,
    input_file,
    output_tsv_path,
    n_components,
):
    # Load the TSV file
    df = pd.read_csv(input_file, sep="\t")

    # Extract feature columns
    feature_names_clean = [n for n in feature_names if n in df.columns]
    features = df.loc[:, feature_names_clean]

    # Perform UMAP to reduce dimensionality to 2 dimensions
    # Using parameters that make it somewhat similar to t-SNE
    umap = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=30,  # Similar to perplexity in t-SNE
        min_dist=0.3,  # orig 0.1
    )
    umap_results = umap.fit_transform(features)

    # Create a new DataFrame for the output TSV
    umap_df = df.drop(columns=[col for col in df.columns if col.startswith("D")])

    for i in range(n_components):
        umap_df[f"UMAP {i+1}"] = umap_results[:, i]

    # Save the new DataFrame as a TSV file
    umap_df.to_csv(output_tsv_path, sep="\t", index=False)


# Your existing feature lists and function calls
orig_features = [f"D{i}" for i in range(1, 101)]
model_prob = [
    "Adjective",
    "Adverb",
    "Conjunction",
    "Noun",
    "Numeral",
    "Particle",
    "Preposition",
    "Pronoun",
    "Verb",
    "Interjection",
]

penn_treebank_tagset = [
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
]

english_POS = [
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Determiner",
    "Pronoun",
    "Preposition_Conjunction",
    "Particle",
    "Interjection",
    "Number",
    # "Other",
]

czech_POS = [
    "A",
    "P",
    "C",
    "R",
    "D",
    "T",
    "I",
    "V",
    "J",
    "N",
]

ud_POS = [
    "ADJ",  # adjective
    "ADP",  # adposition
    "ADV",  # adverb
    "AUX",  # auxiliary
    "CCONJ",  # coordinating conjunction
    "DET",  # determiner
    "NOUN",  # noun
    "NUM",  # numeral
    "PART",  # particle
    "SCONJ",  # subordinating conjunction
    "VERB",  # verb
]


# Updated function calls with new output filenames
if __name__ == "__main__":
    create_umap(
        orig_features,
        "processed_vectors_annot.tsv",
        "fig/umap.pdf",
        "fig/umap_pos.html",
        "umap.tsv",
        "plain embeddings",
        "Czech (CoNLL17)",
    )

    create_umap(
        czech_POS,
        "processed_vectors_annot_SVM.tsv",
        "fig/umap_svm.pdf",
        "fig/umap_svm_pos.html",
        "umap_svm.tsv",
        "SVM",
        "Czech (CoNLL17)",
    )
