import plotly.express as px


def make_chart(df, output_file, method, model, lang, color):
    # Prepare the DataFrame for plotting
    # For demonstration, we will use the first two vector columns as X and Y axes (assuming t-SNE/umap has reduced to 2 dimensions)
    x_column = f"{method} 1"
    y_column = f"{method} 2"

    # Create an interactive scatter plot using Plotly
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color,
        hover_name="Word",
        title=f"Interactive map of {color} in {lang} ({method} of {model})",
    )
    fig = fig.update_layout(
        # template='plotly_dark',
        plot_bgcolor="rgba(0, 0, 0, 0)"
        # paper_bgcolor='rgba(0, 0, 0, 0)',
    )

    # Export the interactive chart
    # Export as an interactive HTML file
    fig.write_html(output_file)

    # Show the interactive plot
    # fig.show()

    print(f"Interactive {method} of {model} scatter plot generated successfully.")
