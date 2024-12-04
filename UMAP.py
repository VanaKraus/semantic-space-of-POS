import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from .chart import make_chart


def create_umap_visualization(
    feature_names,
    input_file,
    output_pdf_path,
    output_html_path,
    output_tsv_path,
    model,
    lang,
):
    # Load the TSV file
    df = pd.read_csv(input_file, sep="\t")

    # Extract feature columns
    feature_names_clean = [n for n in feature_names if n in df.columns]
    features = df.loc[:, feature_names_clean]

    # Extract POS column for color coding
    # pos_labels = df["POS"]

    # Perform UMAP to reduce dimensionality to 2 dimensions
    # Using parameters that make it somewhat similar to t-SNE
    umap = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=30,  # Similar to perplexity in t-SNE
        min_dist=0.3,  # orig 0.1
    )
    umap_results = umap.fit_transform(features)

    # # Prepare a color map for different POS tags
    # unique_pos = pos_labels.unique()
    # colors = plt.colormaps["tab10"]

    # # Create a dictionary for POS to color mapping
    # pos_to_color = {pos: colors(i) for i, pos in enumerate(unique_pos)}

    # # Plot the UMAP results and save as PDF
    # plt.figure(figsize=(12, 8))

    # # Assign colors to each point based on POS tag and plot
    # for pos in unique_pos:
    #     indices = pos_labels == pos
    #     plt.scatter(
    #         umap_results[indices, 0],
    #         umap_results[indices, 1],
    #         color=pos_to_color[pos],
    #         label=pos,
    #         alpha=0.2,
    #         s=10,
    #     )

    # # Add legend and labels
    # plt.title("UMAP Visualization of Word Vectors (Colored by POS)")
    # plt.xlabel("UMAP Dimension 1")
    # plt.ylabel("UMAP Dimension 2")
    # legend = plt.legend(loc="best", markerscale=3, fontsize="small", frameon=True)
    # for handle in legend.legend_handles:
    #     handle.set_alpha(0.8)

    # # Save the plot as a PDF
    # plt.savefig(output_pdf_path, format="pdf")
    # plt.close()

    # Create a new DataFrame for the output TSV
    umap_df = df.drop(columns=[col for col in df.columns if col.startswith("D")])
    umap_df["UMAP 1"] = umap_results[:, 0]
    umap_df["UMAP 2"] = umap_results[:, 1]

    # Save the new DataFrame as a TSV file
    umap_df.to_csv(output_tsv_path, sep="\t", index=False)

    make_chart(
        umap_df, output_html_path, output_pdf_path, "UMAP", model, lang, "POS", "UMAP"
    )


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
    create_umap_visualization(
        orig_features,
        "processed_vectors_annot.tsv",
        "fig/umap.pdf",
        "fig/umap_pos.html",
        "umap.tsv",
        "plain embeddings",
        "Czech (CoNLL17)",
    )

    create_umap_visualization(
        czech_POS,
        "processed_vectors_annot_SVM.tsv",
        "fig/umap_svm.pdf",
        "fig/umap_svm_pos.html",
        "umap_svm.tsv",
        "SVM",
        "Czech (CoNLL17)",
    )
