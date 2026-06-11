import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score

def load_data(file):
    """
    Load raw BRFSS CSV file.

     Parameters
    ----------
    file : str or path-like
        Path to the BRFSS CSV file to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw BRFSS survey data.

    """
    return pd.read_csv(file, sep=",", low_memory=False)


def load_clean_data(file):
    """
     Load cleaned CSV file into DataFrame.

     Parameters
     ----------
     file : str
         Path to the cleaned CSV.

     Returns
     -------
     pd.DataFrame
        DataFrame containing the cleaned BRFSS survey data.
    """
    return pd.read_csv(file, low_memory=False)


def find_existing_column(df, possible_names):
    """
    Return the first column name from possible_names that exists in df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search for column names.
    possible_names : list of str
        Ordered list of column names to look for.

    Returns
    -------
    str or None
        The first matching column name found in df, or None if no match exists.
    """
    for col in possible_names:
        if col in df.columns:
            return col
    return None


def build_column_map(df):
    """
    Build a column map using BRFSS variable names.

    Parameters
    ----------
    df : pd.DataFrame
        BRFSS DataFrame.

    Returns
    -------
    dict 
        Dictionary mapping standardized variable names to their corresponding 
        column name. We found these columns on the BRFSS key (linked to paper).
    """
    return {
        "income":       find_existing_column(df, ["INCOME3"]),
        "education":    find_existing_column(df, ["EDUCA"]),
        "employment":   find_existing_column(df, ["EMPLOY1"]),
        "insurance":    find_existing_column(df, ["PERSDOC3"]),
        "diabetes":     find_existing_column(df, ["DIABETE4"]),
        "hypertension": find_existing_column(df, ["_MICHD"]),
        "cholesterol":  find_existing_column(df, ["CHCSCNC1"]),
        "age":          find_existing_column(df, ["_AGE80"]),
        "sex":          find_existing_column(df, ["SEXVAR"])}


def clean_brfss_data(df, column_map):
    """
    This function keeps the columns used in the project and drops incomplete 
    rows. It also recodes diabetes as a binary variable.

    Parameters
    ----------
    df : pd.DataFrame
        BRFSS DataFrame.
    column_map : dict
        Standardized variable names to actual column names in df.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the selected columns wihtout missing data.

    """
    selected = {k: v for k, v in column_map.items() if v is not None}
    clean_df = df[list(selected.values())].copy()
    clean_df.rename(columns={v: k for k, v in selected.items()}, inplace=True)

    for col in clean_df.columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        
    # recode missing/refused to be na
    global_missing = {
        77: np.nan,   88: np.nan,  99: np.nan,
        777: np.nan,  888: np.nan, 999: np.nan,
        7777: np.nan, 8888: np.nan, 9999: np.nan}
    clean_df.replace(global_missing, inplace=True)

    # For these columns 7 and 9 mean missing/refused
    single_digit_cols = ["income", "education", "diabetes",
                         "hypertension", "cholesterol", "sex", "insurance"]
    
    for col in single_digit_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].replace({7: np.nan, 9: np.nan})

    # In employment 7 is a real category, but 9 is missing
    if "employment" in clean_df.columns:
        clean_df["employment"] = clean_df["employment"].replace({9: np.nan})

    before = len(clean_df)
    clean_df.dropna(inplace=True)
    after = len(clean_df)
    print(f"Dropped {before - after} rows with missing values.")

    # Recode diabetes to binary: 1 = yes, 3 = no
    if "diabetes" in clean_df.columns:
        clean_df = clean_df[clean_df["diabetes"].isin([1, 3])].copy()
        clean_df["diabetes"] = clean_df["diabetes"].map({1: 1, 3: 0})

    return clean_df


def save_clean_model_file(df, file_name="clean_brfss_data.csv"):
    """
    Save cleaned dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    file_name : str, optional
        Cleaned file name. 

    Returns
    -------
    None
    """
    df.to_csv(file_name, index=False)
    print(f"Saved cleaned data as {file_name}")
      
# label socioeconomic factors as nominal or ordinal for modeling
NOMINAL_COLS = ["employment", "insurance"]
ORDINAL_COLS = ["income", "education", "age", "sex"]


def encode_features(X):
    """
    Encode nominal columns, keep ordinal/continuous columns numeric.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame containing NOMINAL_COLS, ORDINAL_COLS.

    Returns
    -------
    pd.DataFrame
        DataFrame with nominal columns replaced by dummies and 
        ordinal/continuous columns kept as numeric. 
    """
    nominal_present = [i for i in NOMINAL_COLS if i in X.columns]
    ordinal_present = [i for i in ORDINAL_COLS if i in X.columns]
    if nominal_present:
        dummies = pd.get_dummies(X[nominal_present].astype(str), \
                                 prefix=nominal_present)
    else:
        dummies = pd.DataFrame(index=X.index)
    return pd.concat([X[ordinal_present].reset_index(drop=True),
                      dummies.reset_index(drop=True)], axis=1)


def prepare_features_and_target(df, target_col):
    """
    Select predictor variables and one target variable.

     Parameters
    ----------
    df : pd.DataFrame
        Cleaned BRFSS DataFrame.
    target_col : str
        Name of the column to use as the target variable.

    Returns
    -------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target variable aligned to X.
    """
    feature_cols = ORDINAL_COLS + NOMINAL_COLS
    available    = [i for i in feature_cols if i in df.columns]
    model_df     = df[available + [target_col]].dropna().copy()
    X_raw        = model_df[available]
    y            = model_df[target_col]
    return encode_features(X_raw), y


def split_train_validation_test(X, y, train_size=0.6, val_size=0.2,
                                test_size=0.2, random_state=42):
    """
    Split into train / validation / test.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target variable aligned to X.
    train_size : float
        Proportion of data for training.
    val_size : float
        Proportion of data for validation. 
    test_size : float
        Proportion of data for testing. 
    random_state : int
        Random seed for reproducibility. 

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
        Feature splits for training, validation, and testing.
    y_train, y_val, y_test : pd.Series
        Target splits for training, validation, and testing.

    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_size), random_state=random_state, \
            stratify=y)
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=random_state, \
            stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_datasets(X_train, X_val, X_test):
    """
    Standardize all features using the training set only.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix used to fit the scaler.
    X_val : pd.DataFrame 
        Validation feature matrix to transform.
    X_test : pd.DataFrame 
        Test feature matrix to transform.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
        Standardized feature nbased on training set.
    """
    scaler = StandardScaler()
    return (scaler.fit_transform(X_train), scaler.transform(X_val),
            scaler.transform(X_test))


def use_sample(X, y, max_rows=None, random_state=42):
    """
    Optionally subsample to speed up run time.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target variable aligned to X.
    max_rows : int
        Maximum number of rows to retain.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_sampled : pd.DataFrame
        Subsampled feature matrix, or original X if no sampling was needed.
    y_sampled : pd.Series
        Subsampled target series aligned to X_sampled, or original y.

    """
    if max_rows is None or len(X) <= max_rows:
        return X, y
    sampled = X.copy()
    sampled["_target"] = y.values
    sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.drop(columns=["_target"]), sampled["_target"]

def predict_one(X_train, y_train, test_row, k):
    """
    Predict the class label for a single test observation using KNN.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels aligned to X_train.
    test_row : np.ndarray
        Single observation.
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    label : int 
        Predicted class label determined by majority vote among the
        k nearest neighbors.
    """
    dists = np.sqrt(np.sum((X_train - test_row) ** 2, axis=1))
    nearest = np.argsort(dists)[:k]
    vals, counts = np.unique(y_train[nearest], return_counts=True)
    return vals[np.argmax(counts)]


def predict_all(X_train, y_train, X_test, k):
    """
    Predict class labels for all rows in a test set using KNN.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels aligned to X_train.
    X_test : np.ndarray
        Test features.
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    y_pred : np.ndarray
        Predicted class labels, one per row in X_test.

    """
    return np.array([predict_one(X_train, y_train, row, k) for row in X_test])


def evaluate_model(y_true, y_pred):
    """
    Compute classification metrics for a set of predictions.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels returned by the classifier.

    Returns
    -------
    accuracy : float
        Overall fraction of correctly classified observations.
    precision : float
        Weighted average precision across all classes.
    recall : float
        Weighted average recall across all classes.
    f1 : float
        Weighted average F1 score across all classes.
    """
    return (accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0))

def run_logistic_regression(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a logistic regression model for one target variable.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels aligned to X_train.
    X_test : np.ndarray
        Scaled test feature.
    y_test : np.ndarray
        Test labels aligned to X_test.
    feature_names : list or pd.Index
        Names of the feature columns used to fit the model.

    Returns
    -------
    acc : float
        Accuracy of the logistic regression model on the test set.
    pre : float
        Weighted precision on the test set.
    rec : float
        Weighted recall on the test set.
    f1 : float
        Weighted F1 score on the test set.
    importance_df : pd.DataFrame
        DataFrame of feature coefficients sorted by absolute value.
    """
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc, pre, rec, f1 = evaluate_model(y_test, y_pred)

    print("\n--- Logistic Regression Results ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({"feature": feature_names,
        "coefficient": coefficients,
        "abs_value": np.abs(coefficients)
        }).sort_values(by="abs_value", ascending=False)

    return acc, pre, rec, f1, importance_df

def test_k_values(X_train, y_train, X_val, y_val, k_values):
    """
    Evaluate KNN performance across multiple values of k on a validation set.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature.
    y_train : np.ndarray
        Training labels aligned to X_train.
    X_val : np.ndarray
        Validation feature.
    y_val : np.ndarray
        Validation labels aligned to X_val.
    k_values : list of int
        Values of k to evaluate.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame k value, accuracy, precision_weighted, recall_weighted, 
        and f1_weighted.
    best_k : int
        Value of k with the highest weighted F1 score on the validation set.
    """
    results = []
    for k in k_values:
        y_pred = predict_all(X_train, y_train, X_val, k)
        acc, pre, rec, f1 = evaluate_model(y_val, y_pred)
        results.append({"k": k, "accuracy": acc, "precision_weighted": pre,
                        "recall_weighted": rec, "f1_weighted": f1})
        print(f"  k={k} Acc={acc:.4f} Pre={pre:.4f} Rec={rec:.4f} F1={f1:.4f}")
    df_r = pd.DataFrame(results)
    return df_r, int(df_r.loc[df_r["f1_weighted"].idxmax(), "k"])

def save_predictions_with_features(X_test_df, y_test, y_pred, target_col):
    """
    Save feature values + actual + predicted to CSV.

    Parameters
    ----------
    X_test_df : pd.DataFrame
        Test feature with original column names.
    y_test : array-like
        Ground truth labels aligned to X_test_df.
    y_pred : array-like
        Predicted labels aligned to X_test_df.
    target_col : str
        Name of the target variable.

    Returns
    -------
    output : pd.DataFrame
        Combined DataFrame containing all feature columns plus
        'actual_{target_col}' and 'predicted_{target_col}' columns.
    """
    output = X_test_df.copy().reset_index(drop=True)
    output[f"actual_{target_col}"]    = y_test
    output[f"predicted_{target_col}"] = y_pred
    fname = f"knn_{target_col}_predictions_full.csv"
    output.to_csv(fname, index=False)
    print(f"  Saved {fname}")
    return output


# Creating the graphs and labeling using info from BRFSS key

EDUCATION_LABELS = {
    1: "Never attended", 2: "Grades 1-8",   3: "Grades 9-11",
    4: "Grade 12/GED",   5: "Some college", 6: "College grad"}
INCOME_LABELS = {
    1: "<$10k",     2: "$10-15k",   3: "$15-20k",   4: "$20-25k",
    5: "$25-35k",   6: "$35-50k",   7: "$50-75k",   8: "$75-100k",
    9: "$100-150k", 10: "$150-200k", 11: ">$200k"}
SEX_LABELS        = {1: "Male", 2: "Female"}
EMPLOYMENT_LABELS = {
    1: "Employed wages",   2: "Self-employed",    3: "Out of work >1yr",
    4: "Out of work <1yr", 5: "Homemaker",        6: "Student",
    7: "Retired",          8: "Unable to work"}
INSURANCE_LABELS = {1: "Yes, one plan", 2: "Yes, multiple", 3: "No insurance"}
TARGET_CLASS_LABELS = {
    "diabetes":     {0: "Not diabetic", 1: "Diabetic"},
    "hypertension": {1: "Has condition", 2: "No condition"},
    "cholesterol":  {1: "Diagnosed",     2: "Not diagnosed"}}


def add_readable_labels(df, target):
    """
    Add labels to the cleaned dataframe for actual outcomes vs factors.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned BRFSS DataFrame.
    target : str
        Name of the target column to create the 'actual_label' column.

    Returns
    -------
    pd.DataFrame
        A df with labels that match the BRFSS key.
    """
    df = df.copy()

    actual_col = target
    df["actual_label"] = df[actual_col].map(
        TARGET_CLASS_LABELS.get(target, {})).fillna(df[actual_col].astype(str))

    # Socioeconomic feature labels
    if "education" in df.columns:
        df["education_label"] = df["education"].map(EDUCATION_LABELS)
        df["education_order"] = df["education"]
    if "income" in df.columns:
        df["income_label"] = df["income"].map(INCOME_LABELS)
        df["income_order"] = df["income"]
    if "sex" in df.columns:
        df["sex_label"] = df["sex"].map(SEX_LABELS)
    if "age" in df.columns:
        bins   = [17, 24, 34, 44, 54, 64, 74, 80]
        labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", \
                  "75-80"]
        df["age_group"] = pd.cut(df["age"], bins=bins,\
                                 labels=labels).astype(str)
    if "employment" in df.columns:
        df["employment_label"] = df["employment"].map(EMPLOYMENT_LABELS)
    if "insurance" in df.columns:
        df["insurance_label"] = df["insurance"].map(INSURANCE_LABELS)

    return df


def make_chart(df, x_col, x_title, target, sort_col=None):
    """
    Stacked bar chart showing the proportion of each health outcome class for 
    each socioeconomic category.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing label columns.
    x_col : str
        Column name to use as the x-axis categories.
    x_title : str
        Title for the x-axis.
    target : str
        Name of the target variable.
    sort_col : str or None
        Column name containing numeric codes for sorting x-axis categories
        in ascending order. If None categories are sorted alphabetically. 
        
    Returns
    -------
    alt.Chart
        Altair Chart object representing the stacked bar chart.

    """
    labels  = "actual_label"
    counts = (df.groupby([x_col, labels]).size().reset_index(name="count"))
    totals              = counts.groupby(x_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    if sort_col and sort_col in df.columns:
        order_map = df[[x_col, sort_col]].drop_duplicates()\
            .set_index(x_col)[sort_col]
        counts["_sort"] = counts[x_col].map(order_map)
        x_enc = alt.X(f"{x_col}:N", title=x_title,
                      sort=alt.EncodingSortField(field="_sort", \
                                                 order="ascending"))
    else:
        x_enc = alt.X(f"{x_col}:N", title=x_title, sort="ascending")

    class_labels = list(TARGET_CLASS_LABELS.get(target, {}).values())
    color_scale  = alt.Scale(domain=class_labels,
                             range=["#4C78A8", "#F58518", "#54A24B","#E45756"])
    selection    = alt.selection_point(fields=[labels], bind="legend")

    return (alt.Chart(counts).mark_bar().encode(x=x_enc,
            y=alt.Y("proportion:Q", title="Proportion",
                    axis=alt.Axis(format="%")),
            color=alt.Color(f"{labels}:N", title="Actual outcome",
                            scale=color_scale),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[alt.Tooltip(f"{x_col}:N", title=x_title),
                alt.Tooltip(f"{labels}:N", title="Actual outcome"),
                alt.Tooltip("proportion:Q", title="Proportion"),
                alt.Tooltip("count:Q", title="Count")])
        .add_params(selection).properties(width=440, height=280,
                    title=f"Actual {target.capitalize()} by {x_title}"))


def build_target_charts(df: pd.DataFrame, target: str):
    """
    Build all charts for one health outcome using actual values.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned BRFSS DataFrame containing numeric coded columns.
    target : str
        Name of the target variable to visualize.

    Returns
    -------
    alt.VConcatChart
        Altair chart object containing all socioeconomic feature charts.
    """
    df = add_readable_labels(df, target)
    charts = []

    if "education_label" in df.columns:
        charts.append(make_chart(df, "education_label", "Education Level", 
                                 target, "education_order"))
    if "income_label" in df.columns:
        charts.append(make_chart(df, "income_label", "Household Income", 
                                 target, "income_order"))
    if "age_group" in df.columns:
        charts.append(make_chart(df, "age_group", "Age Group", target))
    if "sex_label" in df.columns:
        charts.append(make_chart(df, "sex_label", "Sex", target))
    if "employment_label" in df.columns:
        charts.append(make_chart(df, "employment_label", "Employment Status", 
                                 target))
    if "insurance_label" in df.columns:
        charts.append(make_chart(df, "insurance_label", "Insurance Coverage", 
                                 target))

    rows = []
    for i in range(0, len(charts), 2):
        pair = charts[i:i+2]
        rows.append(alt.hconcat(*pair).resolve_scale(color="shared"))

    return alt.vconcat(*rows).properties(title=f"── {target.upper()} ──")


def build_dashboard(df: pd.DataFrame, targets: list,
                    output_file="brfss_dashboard.html"):
    """
    Build and save HTML dashboard for all health outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned BRFSS DataFrame.
    targets : list of str
        Target variable names to include.
    output_file : str, optional
        File path for the saved HTML dashboard. Default 'brfss_dashboard.html'.

    Returns
    -------
    None
        Saves the dashboard to output_file and prints a confirmation message.
    """
    alt.data_transformers.enable("default", max_rows=None)
    pages = [build_target_charts(df, target)
             for target in targets if target in df.columns]
    dashboard = (alt.vconcat(*pages).configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_title(fontSize=16, anchor="start"))

    dashboard.save(output_file)
    print(f"\nDashboard saved to {output_file}")

def run_knn_for_target(df, target_col, k_values, max_rows=10000):
    """
    Run the full KNN training for a single target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned BRFSS DataFrame.
    target_col : str
        Name of the target column to predict.
    k_values : list of int
        Values of k to evaluate.
    max_rows : int or None, optional
        Maximum number of rows to sample before splitting. 

    Returns
    -------
    dict
        Dictionary containing model performance metrics for both KNN
        and logistic regression.
    
    """
    print(f"\n{'=' * 50}")
    print(f"RUNNING KNN FOR: {target_col.upper()}")
    print(f"{'=' * 50}")

    X, y = prepare_features_and_target(df, target_col)
    print("Features:", X.columns.tolist())
    print("Target classes:", sorted(y.unique()))

    X, y = use_sample(X, y, max_rows=max_rows)
    print("Shape after sampling:", X.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_train_validation_test(X, y)
    feature_names = X.columns
    X_tr_s, X_va_s, X_te_s = scale_datasets(X_train, X_val, X_test)

    y_tr = y_train.to_numpy()
    y_va = y_val.to_numpy()
    y_te = y_test.to_numpy()

    results_df, best_k = test_k_values(X_tr_s, y_tr, X_va_s, y_va, k_values)

    y_pred = predict_all(X_tr_s, y_tr, X_te_s, best_k)
    acc, pre, rec, f1 = evaluate_model(y_te, y_pred)

    log_acc, log_pre, log_rec, log_f1, log_importance_df = \
        run_logistic_regression(X_tr_s, y_tr, X_te_s, y_te, feature_names)

    print("\nTop 3 Important Variables:")
    print(log_importance_df.head(3)[["feature", "coefficient"]])

    print(f"\nBest k: {best_k}")

    print("\nPerformance:")

    print("KNN:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    print("Logistic Regression:")
    print(f"  Accuracy : {log_acc:.4f}")
    print(f"  Recall   : {log_rec:.4f}")
    print(f"  F1 Score : {log_f1:.4f}")

    results_df.to_csv(f"knn_{target_col}_k_results.csv", index=False)
    X_test_df = X_test.reset_index(drop=True)
    save_predictions_with_features(X_test_df, y_te, y_pred, target_col)

    return {"target": target_col, "best_k": best_k, "knn_accuracy": acc, 
            "knn_precision": pre, "knn_recall": rec, "knn_f1": f1, 
            "logreg_accuracy": log_acc, "logreg_precision": log_pre,
            "logreg_recall": log_rec, "logreg_f1": log_f1,}


def main():
    
    # set to True after first successful run
    use_clean_file = False  
    raw_file   = "brfss_survey_data_2024.csv"
    clean_file = "clean_brfss_data.csv"

    if use_clean_file:
        try:
            df = load_clean_data(clean_file)
            print(f"Loaded cleaned file: {clean_file}  shape={df.shape}")
        except FileNotFoundError:
            print("Cleaned file not found — building from raw data...")
            raw_df  = load_data(raw_file)
            col_map = build_column_map(raw_df)
            df      = clean_brfss_data(raw_df, col_map)
            save_clean_model_file(df, clean_file)
    else:
        raw_df  = load_data(raw_file)
        col_map = build_column_map(raw_df)
        df      = clean_brfss_data(raw_df, col_map)
        save_clean_model_file(df, clean_file)

    print("Dataset shape:", df.shape)

    targets  = ["diabetes", "hypertension", "cholesterol"]
    k_values = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    results  = []

    for target in targets:
        if target in df.columns:
            results.append(run_knn_for_target(df, target, k_values, 
                                              max_rows=10000))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("knn_all_health_outcomes_summary.csv", index=False)
    print(f"\n{'=' * 50}\nFINAL SUMMARY\n{'=' * 50}")
    pd.set_option('display.max_columns', None)
    print(summary_df)

    build_dashboard(df, targets, "brfss_dashboard.html")

if __name__ == "__main__":
    main()
    
