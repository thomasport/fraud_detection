import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pandas.io.formats.style import Styler
from IPython.display import display_html
from itertools import chain, cycle


def plot_time_stab(
    data: pd.DataFrame, line_y_col="bad", bar_y_col="count", time_col="time_hr"
) -> None:
    """Generate plot with temporal characteristics

    Args:
        data (pd.DataFrame): Dataframe with time column and variables aggregated
        line_y_col (str, optional): Column name of variable used in the line plot. Defaults to 'bad'.
        bar_y_col (str, optional): Column name of variable used in the bar plot. Defaults to 'count'.
        time_col (str, optional): Column name of time variable. Defaults to 'time_hr'.
    """

    fig, ax1 = plt.subplots(figsize=(18, 6))

    fig.suptitle("Time Stability")

    sns.lineplot(data=data[line_y_col], marker="o", sort=False, ax=ax1)
    ax2 = ax1.twinx()

    sns.barplot(data=data, x=time_col, y=bar_y_col, alpha=0.5, ax=ax2)


def bin_var(data: pd.DataFrame, col: str, n_bins=10) -> Styler:
    """Generate summary table of varible binned wit event rate and event ditribution for each bin

    Args:
        data (pd.DataFrame): DataFrame with variable and event column
        col (str): Column name of binned variable
        n_bins (int, optional): Number of bins. Defaults to 10.

    Returns:
        Styler: _description_
    """
    df = data.copy()
    col_binned = col + "_binned"
    cmap = sns.light_palette("red", as_cmap=True)
    df[col_binned] = pd.qcut(df[col], q=n_bins)
    df_agg = (
        df.groupby(col_binned, observed=False)
        .agg(count=(col, "count"), event=("Class", "sum"))
        .reset_index()
    )
    df_agg["event_rate"] = (df_agg.event / df_agg["count"])*100
    df_agg["event_distr"] = (df_agg.event / df_agg.event.sum())*100
    view = df_agg.style.background_gradient(
        subset=["event_rate", "event_distr"], cmap=cmap, vmax=100
    ).format("{:.3f}%",subset=["event_rate", "event_distr"])
    return view
