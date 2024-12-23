import plotly.express as px
from pandas import DataFrame, read_csv

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


def make_chart(
    data: DataFrame | str,
    output_html,
    output_pdf,
    color,
    axis_label_basename,
    title,
    hover_data,
):
    if isinstance(data, str):
        data = read_csv(data, sep="\t")

    mhover_data = [c for c in hover_data if c in data]

    # Prepare the DataFrame for plotting
    # For demonstration, we will use the first two vector columns as X and Y axes (assuming t-SNE/umap has reduced to 2 dimensions)
    x_column = f"{axis_label_basename} 1"
    y_column = f"{axis_label_basename} 2"

    # Create an interactive scatter plot using Plotly
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color,
        color_discrete_map=ud_color_scheme,
        category_orders={color: legend_keys},
        hover_name="Word",
        hover_data=[x_column, y_column] + mhover_data,
        title=title,
        # opacity=0.5,
    )
    fig = fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)")
    fig.update_layout(legend={"itemsizing": "constant"})

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
    if isinstance(data, str):
        data = read_csv(data, sep="\t")

    # Prepare the DataFrame for plotting
    # For demonstration, we will use the first two vector columns as X and Y axes (assuming t-SNE/umap has reduced to 2 dimensions)
    x_column = f"{axis_label_basename} 1"
    y_column = f"{axis_label_basename} 2"
    z_column = f"{axis_label_basename} 3"

    mhover_data = [c for c in hover_data if c in data]

    # Create an interactive scatter plot using Plotly
    fig = px.scatter_3d(
        data,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color,
        color_discrete_map=ud_color_scheme,
        category_orders={color: legend_keys},
        hover_name="Word",
        hover_data=[x_column, y_column, z_column] + mhover_data,
        title=title,
        # opacity=0.5,
    )
    fig = fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)")
    fig.update_layout(legend={"itemsizing": "constant"})
    fig.update_traces(marker={"size": 2.5})

    fig.write_html(output_html)
    fig.write_image(output_pdf, format="pdf")

    print(f"{title} generated successfully.")
