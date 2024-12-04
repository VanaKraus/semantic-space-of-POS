import plotly.express as px


ud_color_scheme = {
    "ADJ": "#1f77b4",  # Blue
    "ADP": "#ff7f0e",  # Orange
    "ADV": "#2ca02c",  # Green
    "AUX": "#d62728",  # Red
    "CCONJ": "#9467bd",  # Purple
    "DET": "#8c564b",  # Brown
    "NOUN": "#e377c2",  # Pink
    "NUM": "#7f7f7f",  # Gray
    "PART": "#bcbd22",  # Yellow-green
    "SCONJ": "#17becf",  # Cyan
    "VERB": "#ff9896",  # Light red
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
        opacity=0.8,
    )
    fig = fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)")

    fig.write_html(output_html)
    fig.write_image(output_pdf, format="pdf")

    print(f"Interactive {method} of {model} scatter plot generated successfully.")
