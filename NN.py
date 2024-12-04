#!/usr/bin/env python3

import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

from .chart import make_chart, make_chart_3d


def train(
    input_file: str,
    output_file: str,
    output_fig_file_base: str,
    n_epochs: int,
    bottleneck_dimensions: int,
):

    # Custom callback to print precision after each epoch
    class EpochEndCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            accuracy = logs.get("accuracy")
            val_accuracy = logs.get("val_accuracy")
            print(
                f"Epoch {epoch + 1} - Accuracy: {accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )

    # Read the TSV file
    data = pd.read_csv(input_file, sep="\t")

    # Extract the feature columns
    feature_cols = [cname for cname in data if cname.startswith("D")]
    X = data[feature_cols].values  # Convert to NumPy array

    # Extract the target column (POS)
    y = data["POS"].values  # Convert to NumPy array

    # Extract the group column ('Lemma')
    groups = data["Lemma"].values

    # Encode the POS labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Map integer labels back to POS labels
    pos_labels = label_encoder.inverse_transform(y_encoded)

    # Convert labels to one-hot encoding
    y_one_hot = to_categorical(y_encoded)

    # Split the data into two halves, grouped by 'Lemma'
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_idx_A, train_idx_B = next(gss.split(X, y_encoded, groups=groups))

    # Create an empty list to collect evaluation results
    eval_results = []

    # Loop over the two splits
    for i, (train_idx, eval_idx) in enumerate(
        [(train_idx_A, train_idx_B), (train_idx_B, train_idx_A)]
    ):
        set_label = "A" if i == 0 else "B"

        # Split the data into training and evaluation sets
        X_train, X_eval = X[train_idx], X[eval_idx]
        y_train, y_eval = y_one_hot[train_idx], y_one_hot[eval_idx]
        pos_labels_eval = pos_labels[eval_idx]

        # Define the neural network
        input_dim = X.shape[1]
        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation="relu")(inputs)
        x = Dense(64, activation="relu")(x)
        x = Dense(16, activation="relu")(x)

        # N-D latent space
        bottleneck = Dense(
            bottleneck_dimensions, activation="linear", name="bottleneck"
        )(x)
        outputs = Dense(y_one_hot.shape[1], activation="softmax")(bottleneck)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Train the model
        model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=32,
            validation_data=(X_eval, y_eval),
            verbose=0,  # Suppress default Keras output
            callbacks=[EpochEndCallback()],  # Custom callback for epoch-end updates)
        )

        print(f"Training {i+1} done")
        # Extract embeddings and probabilities for the evaluation set
        bottleneck_model = Model(
            inputs=inputs, outputs=model.get_layer("bottleneck").output
        )
        embeddings_eval = bottleneck_model.predict(X_eval, verbose=0)
        print(f"Prediction of bottleneck layer done ({i+1}).")
        probabilities_eval = model.predict(X_eval, verbose=0)
        print(f"Prediction of the final layer done ({i+1}).")

        # Create a DataFrame for the evaluation results
        data_eval = data.iloc[eval_idx].copy()
        for dim in range(bottleneck_dimensions):
            data_eval[f"Bottleneck layer {dim+1}"] = embeddings_eval[:, dim]

        # Get POS category names in the same order as in the probabilities
        pos_category_names = label_encoder.classes_

        # Add probability columns for each POS category
        for idx, pos in enumerate(pos_category_names):
            data_eval[f"Probability_{pos}"] = probabilities_eval[:, idx]

        # Indicate whether the data point was in set A or B
        data_eval["Set"] = set_label

        # Append the evaluation results
        eval_results.append(data_eval)

        # Prepare a color map for different POS tags
        unique_pos = np.unique(pos_labels_eval)

        match (bottleneck_dimensions):
            case 2:
                # Create a color map
                cmap = plt.cm.get_cmap("tab10", len(unique_pos))

                # Create a dictionary for POS to color mapping
                pos_to_color = {pos: cmap(j) for j, pos in enumerate(unique_pos)}

                # Plot the embeddings and save as PDF
                plt.figure(figsize=(12, 8))

                # Assign colors to each point based on POS tag and plot
                for pos in unique_pos:
                    indices = pos_labels_eval == pos
                    plt.scatter(
                        embeddings_eval[indices, 0],
                        embeddings_eval[indices, 1],
                        color=pos_to_color[pos],
                        label=pos,
                        alpha=0.7,
                        s=10,
                    )

                # Add legend and labels
                plt.title(f"2D Visualization of Evaluation Set {i+1} (Set {set_label})")
                plt.xlabel("Bottleneck layer 1")
                plt.ylabel("Bottleneck layer 2")
                legend = plt.legend(
                    loc="best", markerscale=3, fontsize="small", frameon=True
                )
                for handle in legend.legend_handles:
                    handle.set_alpha(0.8)

                # Save the plot as a PDF
                plt.savefig(f"{output_fig_file_base}_{set_label}.pdf", format="pdf")
                plt.close()

                make_chart(
                    data_eval,
                    f"{output_fig_file_base}_{set_label}.html",
                    f"{output_fig_file_base}_{set_label}.pdf",
                    "NN Bottleneck",
                    "—",
                    f"Evaluation Set {i+1} (Set {set_label})",
                    "POS",
                )

            case 3:
                make_chart_3d(
                    data_eval,
                    f"{output_fig_file_base}_{set_label}.html",
                    f"{output_fig_file_base}_{set_label}.pdf",
                    "NN Bottleneck",
                    "—",
                    f"Evaluation Set {i+1} (Set {set_label})",
                    "POS",
                )

            case _:
                print(
                    "Warning: visualizations only supported for 2- and 3-dimensional bottlenecks"
                )

    # Combine the evaluation results
    data_combined = pd.concat(eval_results, ignore_index=True)

    # Remove the D1-D100 columns
    data_combined = data_combined.drop(columns=feature_cols)

    cols = data_combined.columns.tolist()
    pos_index = cols.index("POS")
    cols.insert(pos_index + 1, cols.pop(cols.index("Set")))
    data_combined = data_combined[cols]

    # Save the combined data to output_file
    data_combined.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]

    if (ln := len(args)) != 5:
        print(f"incorrect no. of arguments: {ln} found", file=sys.stderr)
        print(f"5 arguments expected:", file=sys.stderr)
        print(f"\t1. processed vectors w/ lemmas and POS", file=sys.stderr)
        print(f"\t2. output file path", file=sys.stderr)
        print(f"\t3. vizualization file basename", file=sys.stderr)
        print(f"\t4. no. of training epochs", file=sys.stderr)
        print(f"\t5. bottleneck dimensions", file=sys.stderr)

        sys.exit(1)

    input_file, output_file, vizualization_file, epochs, bottleneck_dim = args
    epochs, bottleneck_dim = int(epochs), int(bottleneck_dim)

    train(input_file, output_file, vizualization_file, epochs, bottleneck_dim)
