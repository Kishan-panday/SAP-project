import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


RANDOM_STATE = 42


def generate_synthetic_dataset(n_samples: int = 150) -> pd.DataFrame:
    """
    Generate a synthetic corporate employee health dataset.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(22, 60, n_samples)
    gender = rng.choice(["Male", "Female"], n_samples)
    department = rng.choice(
        ["IT", "HR", "Finance", "Marketing", "Operations", "Sales"],
        n_samples
    )
    work_hours = rng.integers(6, 13, n_samples)
    sleep_duration = rng.integers(4, 10, n_samples)
    stress_level = rng.integers(1, 11, n_samples)
    exercise_frequency = rng.integers(0, 6, n_samples)
    job_satisfaction = rng.integers(1, 11, n_samples)
    smoking_habit = rng.choice(["No", "Occasionally", "Yes"], n_samples, p=[0.6, 0.25, 0.15])
    alcohol_consumption = rng.choice(["No", "Occasionally", "Yes"], n_samples, p=[0.45, 0.4, 0.15])
    bmi = np.round(rng.normal(24.5, 3.8, n_samples), 1)
    wfh_frequency = rng.integers(0, 6, n_samples)  # days per week

    # Risk score logic to make target realistic
    risk_score = (
        (work_hours >= 10).astype(int) * 2
        + (sleep_duration <= 5).astype(int) * 2
        + (stress_level >= 8).astype(int) * 3
        + (exercise_frequency <= 1).astype(int) * 1
        + (job_satisfaction <= 4).astype(int) * 2
        + (bmi >= 29).astype(int) * 1
        + (smoking_habit == "Yes").astype(int) * 1
        + (alcohol_consumption == "Yes").astype(int) * 1
    )

    # Convert risk score into binary target with controlled noise
    health_risk = (risk_score + rng.integers(0, 2, n_samples) >= 5).astype(int)

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Department": department,
        "Work_Hours_Per_Day": work_hours,
        "Sleep_Duration": sleep_duration,
        "Stress_Level": stress_level,
        "Exercise_Frequency": exercise_frequency,
        "Job_Satisfaction": job_satisfaction,
        "Smoking_Habit": smoking_habit,
        "Alcohol_Consumption": alcohol_consumption,
        "BMI": bmi,
        "WFH_Frequency": wfh_frequency,
        "Health_Risk": health_risk
    })

    # Add a few missing values intentionally for realism
    for col in ["Gender", "Department", "BMI"]:
        idx = rng.choice(df.index, size=max(2, n_samples // 30), replace=False)
        df.loc[idx, col] = np.nan

    return df


def remove_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """
    Remove outliers using the IQR method for selected numeric columns.
    """
    cleaned_df = df.copy()

    for col in numeric_columns:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]

    return cleaned_df


def preprocess_features(X: pd.DataFrame):
    """
    Build a preprocessing pipeline for numeric and categorical features.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def label_encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_columns: list[str]):
    """
    Apply Label Encoding to categorical columns.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in categorical_columns:
        le = LabelEncoder()
        train_encoded[col] = train_encoded[col].astype(str)
        test_encoded[col] = test_encoded[col].astype(str)

        le.fit(pd.concat([train_encoded[col], test_encoded[col]], axis=0))
        train_encoded[col] = le.transform(train_encoded[col])
        test_encoded[col] = le.transform(test_encoded[col])

    return train_encoded, test_encoded


def evaluate_model(model_name: str, y_true, y_pred) -> dict:
    """
    Compute evaluation metrics for a model.
    """
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1-score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def run_experiment(df: pd.DataFrame, test_size: float):
    """
    Run one train-test split experiment.
    """
    X = df.drop(columns=["Health_Risk"])
    y = df["Health_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    _, numeric_cols, categorical_cols = preprocess_features(X_train)

    # Fill missing values before label encoding
    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())

    for col in categorical_cols:
        X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
        X_test[col] = X_test[col].fillna(X_train[col].mode()[0])

    # Label encode categoricals
    X_train, X_test = label_encode_categorical(X_train, X_test, categorical_cols)

    # Scale numeric columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Balance only training data
    if SMOTE_AVAILABLE and len(np.unique(y_train)) > 1:
        minority_count = y_train.value_counts().min()
        if minority_count >= 2:
            k_neighbors = max(1, min(5, minority_count - 1))
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    }

    results = []
    reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(evaluate_model(name, y_test, y_pred))
        reports[name] = classification_report(y_test, y_pred, zero_division=0)

    return pd.DataFrame(results), reports


def main():
    print("\nGenerating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=150)

    print("Original dataset shape:", df.shape)

    numeric_columns = [
        "Age", "Work_Hours_Per_Day", "Sleep_Duration", "Stress_Level",
        "Exercise_Frequency", "Job_Satisfaction", "BMI", "WFH_Frequency"
    ]

    # Remove rows that are completely unusable for target or excessive outliers
    df_clean = df.dropna(subset=["Health_Risk"]).copy()
    df_clean = remove_outliers_iqr(df_clean, numeric_columns)

    print("Dataset shape after preprocessing:", df_clean.shape)
    print("\nClass distribution:")
    print(df_clean["Health_Risk"].value_counts())

    split_settings = {
        "70:30": 0.30,
        "80:20": 0.20,
        "90:10": 0.10
    }

    all_results = []

    for split_name, test_size in split_settings.items():
        print(f"\n{'=' * 60}")
        print(f"Running experiment for split: {split_name}")
        print(f"{'=' * 60}")

        result_df, reports = run_experiment(df_clean, test_size=test_size)
        result_df.insert(0, "Split", split_name)
        all_results.append(result_df)

        print(result_df.to_string(index=False))

        best_model = result_df.sort_values(by="F1-score", ascending=False).iloc[0]["Model"]
        print(f"\nBest model for {split_name}: {best_model}")
        print("\nClassification report for best model:")
        print(reports[best_model])

    final_results = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 60)
    print("Final Summary Across Splits")
    print("=" * 60)
    print(final_results.to_string(index=False))

    final_results.to_csv("employee_health_model_results.csv", index=False)
    df_clean.to_csv("corporate_employee_health_dataset.csv", index=False)

    print("\nSaved files:")
    print("- corporate_employee_health_dataset.csv")
    print("- employee_health_model_results.csv")


if __name__ == "__main__":
    main()
