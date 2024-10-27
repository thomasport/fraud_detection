from sklearn.linear_model import lasso_path, LogisticRegression
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from itertools import cycle
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


SEED = 42


def undersampling(df: pd.DataFrame, 
                  prop=None, 
                  target="Class") -> pd.DataFrame:
    """Undersampling the majority class sample by a specific proportion. 
       If the proportion is not informed, returns majority class with 
       same proportion as minority class 


    Args:
        df (pd.DataFrame): Data with imabalanced target
        prop (float, optional): Proportion to undersample. Defaults to None.
        target (str, optional): Column name of target. Defaults to "Class".

    Returns:
        pd.DataFrame: Undersampled dataframe
    """
    c_min = df["Class"].value_counts().argmin()

    if not prop:
        n_class_min = df["Class"].value_counts().min()
        prop = n_class_min / df.shape[0]

    df_min = df[df[target] == c_min]
    df_max = df[df[target] != c_min]

    n = int(prop * df.shape[0])
    df_max_resampled = df_max.sample(n, random_state=SEED)

    return pd.concat([df_max_resampled, df_min])


def calc_metrics(
    data: pd.DataFrame, y_true="y_true", y_pred="y_pred", scores="scores"
) -> pd.DataFrame:
    """Calculate metrics from predictions

    Args:
        data (pd.DataFrame): Dataframe with predictions, true label and Amount
        y_true (str, optional): Column name of true label. Defaults to 'y_true'.
        y_pred (str, optional): Column name of predictions. Defaults to 'y_pred'.
        scores (str, optional): Column name of predicted scores. Defaults to 'scores'.

    Returns:
        pd.DataFrame: Table with performance metrics
    """
    recall = recall_score(data[y_true], data[y_pred])
    precision = precision_score(data[y_true], data[y_pred])

    f1 = f1_score(data[y_true], data[y_pred])

    auc = roc_auc_score(data[y_true], data[scores])

    money_cost = calc_cost(data)

    df_metrics = pd.DataFrame(
        {
            "recall": {"value": recall},
            "precision": {"value": precision},
            "f1": {"value": f1},
            "auc": {"value": auc},
            "Cost": {"value": money_cost},
        }
    ).style.format("{:.2f}")

    return df_metrics


def cost(df: pd.DataFrame, y_true="y_true", y_pred="y_pred"):
    """Calculate the custom cost from predictions

    Args:
        df (pd.DataFrame): Data with predictions and label
        y_true (str, optional): Column name of true label. Defaults to 'y_true'.
        y_pred (str, optional): Column name of predictions. Defaults to 'y_pred'.

    Raises:
        ValueError: Invalid Label

    Returns:
        float: Cost for specific sample
    """
    if (df[y_pred] == 0) & (df[y_true] == 0):
        return 0
    elif (df[y_pred] == 0) & (df[y_true] == 1):
        return -df["Amount"]
    elif (df[y_pred] == 1) & (df[y_true] == 0):
        return -2
    elif (df[y_pred] == 1) & (df[y_true] == 1):
        return 0.2 * df["Amount"]
    else:
        raise ValueError


def calc_cost(data: pd.DataFrame, y_true="y_true", y_pred="y_pred") -> pd.Series:
    """_summary_

    Args:
        data (pd.DataFrame): Sample with prediction, true label and amount
        y_true (str, optional): Column name of true label. Defaults to 'y_true'.
        y_pred (str, optional): Column name of predictions. Defaults to 'y_pred'.

    Returns:
        pd.Series: Cost calculated for the sample
    """

    costs_vector = data.apply(lambda x: cost(x, y_true=y_true, y_pred=y_pred), axis=1)
    return costs_vector.sum()


def roc_auc_curve(y_true: pd.Series, scores: pd.Series):
    """Generate ROC curve from predictions

    Args:
        y_true (pd.Series): Series with true label
        scores (pd.Series): Series with predicted scores

    Returns:
        plt.Figure: ROC Curve
    """
    return RocCurveDisplay.from_predictions(y_true, scores)


def conf_mat(y_true: pd.Series, y_pred: pd.Series, figsize=(5, 5)):
    """Generate Confusion Matrix

    Args:
        y_true (pd.Series): True Label
        y_pred (pd.Series): Predicted Label
        figsize (tuple, optional): Figure size. Defaults to (5,5).

    Returns:
        Figure: Confusion matrix
    """

    conf_mat = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(conf_mat, index=np.unique(y_true), columns=np.unique(y_true))

    group_counts = ["{0:0.0f}".format(value) for value in conf_mat.flatten()]

    group_percentages = [
        "{0:.2%}".format(value)
        for value in (conf_mat.T / np.sum(conf_mat, axis=1)).T.flatten()
    ]

    labels = [f"{c}\n{perc}" for c, perc in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)
    cm.index.name = "True"
    cm.columns.name = "Predicted"
    _, ax = plt.subplots(figsize=figsize)

    return sns.heatmap(cm, cmap="YlGnBu", annot=labels, fmt="", ax=ax)
