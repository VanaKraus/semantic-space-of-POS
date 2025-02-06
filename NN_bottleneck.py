#!/usr/bin/env python3

import re
import sys
from collections.abc import Iterable, Callable
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical, set_random_seed
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.layers import Input, Dense, Dropout


def _one_hot(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray[str]]:
    # Extract the target column (POS)
    y = df["POS_Top"].values  # Convert to NumPy array

    # Encode the POS labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Convert labels to one-hot encoding
    return to_categorical(y_encoded), label_encoder.classes_


def _proportional(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray[str]]:
    # POS columns start at the 5th position and are followed by a 100D vector
    return df.iloc[:, 5:-100].values, df.columns[5:-100]


_Y_PREPS: dict[str, Callable[[pd.DataFrame], tuple[np.ndarray, list[str]]]] = {
    "one_hot": _one_hot,
    "proportional": _proportional,
}


def train(
    input_file: str,
    output_file: str,
    output_file_sets: Iterable[str],
    confusion_matrices_files: Iterable[str],
    n_epochs: int,
    layers: Iterable[int],
    bottleneck_dimensions: int,
    seed: int,
    y_preparation: Literal["one_hot", "proportional"],
):
    print(
        f"Bottleneck NN: {input_file=} {output_file=} {output_file_sets=} {n_epochs=} {layers=} {bottleneck_dimensions=} {seed=} {y_preparation=}"
    )

    set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

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
    data.columns = data.columns.str.removesuffix("_rf")

    # Extract the feature columns
    feature_cols = [cname for cname in data if re.match(r"D[0-9]+", cname)]
    X = data[feature_cols].values  # Convert to NumPy array

    # Convert labels to one-hot encoding
    y_prepared, pos_category_names = _Y_PREPS[y_preparation](data)
    pos_category_names_dict = {i: n for i, n in enumerate(pos_category_names)}

    groups = data["Group"].values

    # Split the data into two halves, grouped by 'Lemma'
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    train_idx_A, train_idx_B = next(gss.split(X, y_prepared, groups=groups))

    # Create an empty list to collect evaluation results
    eval_results = []

    # Loop over the two splits
    for i, (train_idx, eval_idx) in enumerate(
        [(train_idx_A, train_idx_B), (train_idx_B, train_idx_A)]
    ):
        set_label = "A" if i == 0 else "B"

        # Split the data into training and evaluation sets
        X_train, X_eval = X[train_idx], X[eval_idx]
        y_train, y_eval = y_prepared[train_idx], y_prepared[eval_idx]

        # Define the neural network
        input_dim = X.shape[1]
        inputs = Input(shape=(input_dim,))

        x = inputs
        for dim in layers:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(0.3)(x)

        # N-D latent space
        bottleneck = Dense(
            bottleneck_dimensions, activation="linear", name="bottleneck"
        )(x)
        outputs = Dense(y_prepared.shape[1], activation="softmax")(bottleneck)

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

        probabilities_max = np.argmax(probabilities_eval, axis=1)
        y_eval_max = np.argmax(y_eval, axis=1)

        probabilities_labeled = np.vectorize(pos_category_names_dict.get)(
            probabilities_max
        )
        y_eval_labeled = np.vectorize(pos_category_names_dict.get)(y_eval_max)

        # Confusion matrix
        pd.options.display.float_format = "{:.0f}".format
        conf_matrix = pd.DataFrame(
            to_categorical(probabilities_max), columns=pos_category_names
        )
        conf_matrix["target"] = y_eval_labeled
        conf_matrix = conf_matrix.groupby("target").sum()
        print("Confusion Matrix:\n")
        print(conf_matrix)
        print()

        print("Classification Report:\n")
        print(
            classification_report(
                y_eval_labeled, probabilities_labeled, zero_division=1
            )
        )

        conf_matrix.to_csv(confusion_matrices_files[i], sep="\t", index=False)

        # Relative confusion matrix
        pd.options.display.float_format = "{:.2f}".format
        rel_cm = conf_matrix.copy()
        rel_cm["sum"] = rel_cm.sum(axis=1)
        for pos in pos_category_names:
            rel_cm[pos] = rel_cm[pos] / rel_cm["sum"]
        rel_cm = rel_cm.drop(columns=["sum"])
        print("Relative confusion matrix:\n")
        print(rel_cm)
        print()

        # Avg. output activation for each POS
        df_avg = pd.DataFrame(probabilities_eval, columns=pos_category_names)
        df_avg["target"] = y_eval_labeled
        df_avg_mean = df_avg.groupby("target").mean()
        df_avg_std = df_avg.groupby("target").agg("std")
        print("Mean output activation of each POS node for each target POS:\n")
        print(df_avg_mean)
        print()
        print("Mean standard deviation of each POS node for each target POS:\n")
        print(df_avg_std)
        print()

        # Create a DataFrame for the evaluation results
        data_eval = data.iloc[eval_idx].copy()
        for dim in range(bottleneck_dimensions):
            data_eval[f"Bottleneck layer {dim+1}"] = embeddings_eval[:, dim]

        # Add probability columns for each POS category
        for idx, pos in enumerate(pos_category_names):
            data_eval[pos] = probabilities_eval[:, idx]

        # Indicate whether the data point was in set A or B
        data_eval["Training_Round"] = set_label

        # Append the evaluation results
        eval_results.append(data_eval)

        data_eval.drop(columns=feature_cols).to_csv(
            output_file_sets[i], sep="\t", index=False
        )

    # Combine the evaluation results
    data_combined = pd.concat(eval_results, ignore_index=True)

    # Remove the D1-D100 columns
    data_combined = data_combined.drop(columns=feature_cols)

    cols = data_combined.columns.tolist()
    pos_index = cols.index("POS_Top")
    cols.insert(pos_index + 1, cols.pop(cols.index("Training_Round")))
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

    train(input_file, output_file, vizualization_file, epochs, [128], bottleneck_dim)
