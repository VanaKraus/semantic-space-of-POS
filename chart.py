import plotly.express as px
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ud_color_scheme = {
    "ADJ": "#e60049",
    "ADP": "#dc0ab4",
    "ADV": "#50e991",
    "AUX": "#fd7f6f",
    "CCONJ": "#9b19f5",
    "DET": "#ffa300",
    "NOUN": "#0bb4ff",
    "NUM": "#b3d4ff",
    "PART": "#00bfa0",
    "PRON": "#77ac35",
    "SCONJ": "#7c1158",
    "VERB": "#e6d800",
}

legend_keys = list(ud_color_scheme.keys())
legend_keys.sort()


def humanify_probability(prob: float) -> str:
    rounded = round(prob, 3)

    return f"{rounded:.3f}" if rounded >= 0.001 else "—"


def make_chart(
    data: DataFrame | str,
    output_html,
    output_pdf,
    color,
    axis_label_basename,
    title,
    hover_data,
):
    data_tmp = read_csv(data, sep="\t") if isinstance(data, str) else data.copy()

    # Prepare the DataFrame for plotting
    # For demonstration, we will use the first two vector columns as X and Y axes (assuming t-SNE/umap has reduced to 2 dimensions)
    x_column = f"{axis_label_basename} 1"
    y_column = f"{axis_label_basename} 2"

    position_cols = [x_column, y_column]

    mhover_cols = [c for c in hover_data if c in data_tmp]
    for col in mhover_cols:
        data_tmp[col] = data_tmp[col].transform(humanify_probability)

    data_tmp = data_tmp[["Word", color] + mhover_cols + position_cols]

    # Create an interactive scatter plot using Plotly
    fig = px.scatter(
        data_tmp,
        x=x_column,
        y=y_column,
        color=color,
        color_discrete_map=ud_color_scheme,
        category_orders={color: legend_keys},
        hover_name="Word",
        hover_data=mhover_cols,
        title=title,
        # opacity=0.5,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        legend={"itemsizing": "constant"},
    )

    fig.write_html(output_html)

    fig.update_traces(marker={"size": 2.5})
    fig.write_image(output_pdf, format="pdf")

    print(f"{title} generated successfully.")


def make_chart_3d(
    data: DataFrame | str,
    output_html,
    output_pdf,
    color,
    axis_label_basename,
    title,
    hover_data,
):
    data_tmp = read_csv(data, sep="\t") if isinstance(data, str) else data.copy()

    # Prepare the DataFrame for plotting
    # For demonstration, we will use the first two vector columns as X and Y axes (assuming t-SNE/umap has reduced to 2 dimensions)
    x_column = f"{axis_label_basename} 1"
    y_column = f"{axis_label_basename} 2"
    z_column = f"{axis_label_basename} 3"

    position_cols = [x_column, y_column, z_column]

    mhover_cols = [c for c in hover_data if c in data_tmp]
    for col in mhover_cols:
        data_tmp[col] = data_tmp[col].transform(humanify_probability)

    data_tmp = data_tmp[["Word", color] + mhover_cols + position_cols]

    # Create an interactive scatter plot using Plotly
    fig = px.scatter_3d(
        data_tmp,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color,
        color_discrete_map=ud_color_scheme,
        category_orders={color: legend_keys},
        hover_name="Word",
        hover_data=mhover_cols,
        title=title,
        template="plotly_white",
        # opacity=0.5,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        legend={"itemsizing": "constant"},
    )
    fig.update_traces(marker={"size": 2.5})

    fig.write_html(output_html)
    fig.write_image(output_pdf, format="pdf")

    print(f"{title} generated successfully.")


def plot_confusion_matrix(matrix_path, output_path, title):
    matrix_df = read_csv(matrix_path, sep="\t")
    pos = matrix_df.columns
    matrix_df = matrix_df.astype({p: "int" for p in pos})

    matrix_df_rel = matrix_df.astype({p: "float" for p in pos})
    matrix_df_rel["sum"] = matrix_df_rel.sum(axis=1)
    for p in pos:
        matrix_df_rel[p] = matrix_df_rel[p] / matrix_df_rel["sum"]
    matrix_df_rel = matrix_df_rel.drop(columns=["sum"])

    matrix_df.index = pos
    matrix_df_rel.index = pos

    plt.figure(figsize=(7.3, 6))

    # Format the difference values
    def format_value(val):
        return f"{val:.0f}" if not np.isnan(val) else ""

    formatted_results = matrix_df.map(format_value)
    print(formatted_results)

    # Plot heatmap
    ax = sns.heatmap(
        matrix_df_rel, annot=formatted_results, cmap="rocket_r", square=True, fmt=""
    )
    plt.yticks(rotation=0)

    ax.collections[0].colorbar.set_label("Preference strength by each target POS")
    ax.set(xlabel="Predicted POS", ylabel="Target POS")

    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_path)
