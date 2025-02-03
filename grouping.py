import pandas as pd
import networkx as nx
import numpy as np


def create_groups(path: str, out_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    # Step 1: Create edges for the graph
    edges = []
    lemma_to_rows = {}

    for idx, row in df.iterrows():
        lemmas = set(row["Lemmas"].split(","))  # Split lemmas into a set
        for lemma in lemmas:
            if lemma not in lemma_to_rows:
                lemma_to_rows[lemma] = []
            lemma_to_rows[lemma].append(idx)  # Store row index for each lemma

    # Create edges between rows that share a lemma
    for rows in lemma_to_rows.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                edges.append((rows[i], rows[j]))

    # Step 2: Create a graph and find connected components
    G = nx.Graph()
    G.add_edges_from(edges)
    connected_components = list(nx.connected_components(G))

    # Step 3: Assign group IDs
    group_mapping = {}
    for group_id, component in enumerate(connected_components):
        for index in component:
            group_mapping[index] = group_id

    # Step 4: Assign groups to the DataFrame
    # Find NaNs (unassigned groups)
    nan_mask = df.index.map(group_mapping).isna()

    # Assign random group numbers to NaN rows
    random_groups = np.random.randint(
        len(df), 1000 * len(df), size=nan_mask.sum()
    )  # Adjust range as needed

    # Map groups, replacing NaNs with random values
    df["Group"] = df.index.map(group_mapping)
    df.loc[nan_mask, "Group"] = random_groups
    df["Group"] = df["Group"].astype(int)  # Ensure integer type

    cols = df.columns.tolist()
    cols.insert(cols.index("Lemmas") + 1, cols.pop(cols.index("Group")))
    df = df[cols]

    if out_path:
        df.to_csv(out_path, sep="\t")

    return df
