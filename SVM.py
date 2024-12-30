#!/usr/bin/env python3

import sys

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit


def predict_class_probabilities(
    input_file, output_file, kernel="rbf", C=1.0, gamma="scale", seed: int | None = None
):
    # Read the TSV file
    data = pd.read_csv(input_file, sep="\t")

    # Check for NANs
    nan_selector = data.isna().any(axis=1)
    if any(nan_selector):
        nan_rows = data[nan_selector]
        print(f"skipping {len(nan_rows)} row(s) containing NAN values")
        print(list(nan_rows["Word"]))
        data = data[~nan_selector]

    # Extract the feature columns (D1-D100)
    feature_cols = ["D" + str(i) for i in range(1, 101)]
    X = data[feature_cols]

    # Extract the target column (POS)
    y = data["POS"]

    # Extract the group column (Lemma)
    groups = data["Lemma"]

    # Create a GroupShuffleSplit object
    print("Shuffle")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)

    # Split the data into two halves based on the 'Lemma' column
    print("Split")
    A_index, B_index = next(gss.split(X, y, groups))

    # Get the A and B sets
    X_A, X_B = X.iloc[A_index], X.iloc[B_index]
    y_A, y_B = y.iloc[A_index], y.iloc[B_index]

    # First round: Train on A, validate on B
    print("First round: Train on A, validate on B")
    model_A = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=seed)
    model_A.fit(X_A, y_A)

    probabilities_A = model_A.predict_proba(X_B)
    prob_df_A = pd.DataFrame(probabilities_A, columns=model_A.classes_, index=X_B.index)
    prob_df_A["Training_Round"] = "A"

    # Second round: Train on B, validate on A
    print("Second round: Train on B, validate on A")
    model_B = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=seed)
    model_B.fit(X_B, y_B)

    probabilities_B = model_B.predict_proba(X_A)
    prob_df_B = pd.DataFrame(probabilities_B, columns=model_B.classes_, index=X_A.index)
    prob_df_B["Training_Round"] = "B"

    print("Process")

    # Combine the predicted probabilities from both rounds
    prob_df = pd.concat([prob_df_A, prob_df_B])

    # Combine the original data (excluding D1-D100) with the predicted probabilities
    output_data = data.drop(columns=feature_cols)
    output_data = pd.concat([output_data, prob_df], axis=1)

    # Move the "Training_Round" column to be just after the "POS" column
    columns = output_data.columns.tolist()
    czech_index = columns.index("POS")
    columns.insert(czech_index + 1, columns.pop(columns.index("Training_Round")))
    output_data = output_data[columns]

    # Fill missing values with zeros
    output_data.fillna(0, inplace=True)

    # Save the output data to a new TSV file
    output_data.to_csv(output_file, sep="\t", index=False)

    print(f"Output saved to {output_file}")

    # Print success metrics for both rounds
    y_pred_A = model_A.predict(X_B)
    y_pred_B = model_B.predict(X_A)

    print("Round 1 (Train: A, Validate: B)")
    print("Accuracy:", accuracy_score(y_B, y_pred_A))
    print("Confusion Matrix:\n", confusion_matrix(y_B, y_pred_A))
    print(
        "Classification Report:\n",
        classification_report(y_B, y_pred_A, zero_division=1),
    )

    print("\nRound 2 (Train: B, Validate: A)")
    print("Accuracy:", accuracy_score(y_A, y_pred_B))
    print("Confusion Matrix:\n", confusion_matrix(y_A, y_pred_B))
    print(
        "Classification Report:\n",
        classification_report(y_A, y_pred_B, zero_division=1),
    )


# Example usage
if __name__ == "__main__":
    args = sys.argv[1:]

    if (ln := len(args)) != 5:
        print(f"incorrect no. of arguments: {ln} found", file=sys.stderr)
        print(f"5 arguments expected:", file=sys.stderr)
        print(f"\t1. processed vectors w/ lemmas and POS", file=sys.stderr)
        print(f"\t2. output file path", file=sys.stderr)
        print(f"\t3. kernel ('linear', 'poly', 'rbf', 'sigmoid')", file=sys.stderr)
        print(f"\t4. regularization parameter (e.g. 1.0)", file=sys.stderr)
        print(
            f"\t5. gamma / kernel coefficient ('scale', 'auto', or a float value)",
            file=sys.stderr,
        )

        sys.exit(1)

    input_file, output_file, kernel, C, gamma = args

    C = float(C)
    if gamma not in ("scale", "auto"):
        gamma = float(gamma)

    predict_class_probabilities(input_file, output_file, kernel, C, gamma)
