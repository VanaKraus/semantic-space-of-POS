import plotly.express as px


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
    "SCONJ": "#7c1158",
    "VERB": "#e6d800",
}

legend_keys = list(ud_color_scheme.keys())
legend_keys.sort()


def make_chart(
    df, output_html, output_pdf, method, model, lang, color, axis_label_basename
):
    # Prepare the DataFrame for plotting
    # For demonstration, we will use the first two vector columns as X and Y axes (assuming t-SNE/umap has reduced to 2 dimensions)
    x_column = f"{axis_label_basename} 1"
    y_column = f"{axis_label_basename} 2"

    # Create an interactive scatter plot using Plotly
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color,
        color_discrete_map=ud_color_scheme,
        category_orders={color: legend_keys},
        hover_name="Word",
        title=f"Interactive map of {color} in {lang} ({method} of {model})",
        # opacity=0.5,
    )
    fig = fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)")

    fig.write_html(output_html)
    fig.write_image(output_pdf, format="pdf")

    print(f"Interactive {method} of {model} scatter plot generated successfully.")


def make_chart_3d(
    df, output_html, output_pdf, method, model, lang, color, axis_label_basename
):
    # Prepare the DataFrame for plotting
    # For demonstration, we will use the first two vector columns as X and Y axes (assuming t-SNE/umap has reduced to 2 dimensions)
    x_column = f"{axis_label_basename} 1"
    y_column = f"{axis_label_basename} 2"
    z_column = f"{axis_label_basename} 3"

    # Create an interactive scatter plot using Plotly
    fig = px.scatter_3d(
        df,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color,
        color_discrete_map=ud_color_scheme,
        category_orders={color: legend_keys},
        hover_name="Word",
        title=f"Interactive map of {color} in {lang} ({method} of {model})",
        # opacity=0.5,
    )
    fig = fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)")

    fig.write_html(output_html)
    fig.write_image(output_pdf, format="pdf")

    print(f"Interactive {method} of {model} scatter plot generated successfully.")
