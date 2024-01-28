import os
import sys

sys.path.append("../")
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from soc_estimation_nn.helper import plot

timestamp_col = "Timestamp"
time_col = "Time [min]"
voltage_col = "Voltage [V]"
current_col = "Current [A]"
temperature_col = "Temperature [degC]"
capacity_col = "Capacity [Ah]"
soc_col = "SOC [-]"
ocv_col = "Open Circuit Voltage [V]"

parsed_data_directory = "../data/parsed"
processed_data_directory = "../data/processed"


def get_pOCV_SOC_interp_fn(file_path: str, x_parameter: str) -> interp1d:
    """
    Create pseudo OCV-SOC interpolation function from slow discharge data.

    Parameters
    ----------
    file_path : str
        Path to slow discharge data
    x_parameter : str
        X parameter, either "OCV" or "SOC"

    Returns
    -------
    interp1d
        Interpolation function that takes OCV/SOC as input and returns the other
    """
    if x_parameter.upper() not in ["OCV", "SOC"]:
        raise ValueError("x_parameter needs to be one of ['OCV', 'SOC']")

    df = pd.read_csv(file_path)
    df = df[df[current_col] < 0]
    df[capacity_col] = df[capacity_col] - df[capacity_col].iloc[0]
    df[soc_col] = 1 - abs(df[capacity_col] / df[capacity_col].iloc[-1])

    match x_parameter.upper():
        case "OCV":
            return interp1d(df[voltage_col], df[soc_col])

        case "SOC":
            return interp1d(df[soc_col], df[voltage_col])

    return


def estimate_soc_and_ocv(
    df: pd.DataFrame, get_soc_fn: interp1d, get_ocv_fn: interp1d
) -> pd.DataFrame:
    """
    Create a new SOC column with estimated values.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    get_soc_fn : interp1d
        OCV-SOC interpolation function

    Returns
    -------
    pd.DataFrame
        Time series data with estimated SOC values
    """

    df[capacity_col] = df[capacity_col] - df[capacity_col].iloc[0]

    final_soc = float(get_soc_fn(df[voltage_col].iloc[-1]))
    est_total_capacity = abs(df[capacity_col].iloc[-1]) / (1 - final_soc)
    df[soc_col] = 1 - abs(df[capacity_col]) / est_total_capacity
    df[ocv_col] = df[soc_col].apply(get_ocv_fn).apply(float)

    return df


def generate_and_save_plot(
    data_df: pd.DataFrame,
    save_file_path: str,
    fig_title: str = "",
) -> None:
    """
    Generate a plot for parsed raw data and saves figure as png.

    Parameters
    ----------
    data_df : pd.DataFrame
        Parsed data for plotting.
    save_file_path : str
        File path of saved figure.
    fig_title : str, optional
        Figure title, by default ''
    """

    fig, _ = plot(
        xy_data=[
            {
                "x_data": data_df[time_col],
                "y_data": data_df[voltage_col],
                "label": "Voltage",
            },
            {
                "x_data": data_df[time_col],
                "y_data": data_df[current_col],
                "label": "Current",
                "plot_num": 2,
            },
            {
                "x_data": data_df[time_col],
                "y_data": data_df[temperature_col],
                "label": "Temperature",
                "plot_num": 3,
            },
            {
                "x_data": data_df[time_col],
                "y_data": data_df[capacity_col],
                "label": "Capacity",
                "plot_num": 4,
            },
            {
                "x_data": data_df[time_col],
                "y_data": data_df[soc_col],
                "label": "SOC",
                "plot_num": 5,
            },
        ],
        x_label=time_col,
        y_label={
            1: voltage_col,
            2: current_col,
            3: temperature_col,
            4: capacity_col,
            5: soc_col,
        },
        title=fig_title,
        fig_size=(10, 12.5),
        show_plt=False,
    )

    fig.savefig(save_file_path)

    return


if __name__ == "__main__":
    temperatures = filter(
        lambda folder: "degC" in folder, os.listdir(parsed_data_directory)
    )

    for T in temperatures:
        parsed_data_T_directory = f"{parsed_data_directory}/{T}"
        processed_data_T_directory = f"{processed_data_directory}/{T}"

        if not os.path.exists(processed_data_T_directory):
            os.makedirs(processed_data_T_directory)

        C20_file_name = next(
            filter(
                lambda f: f.endswith(".csv") and "C20" in f,
                os.listdir(parsed_data_T_directory),
            )
        )
        C20_file_path = f"{parsed_data_T_directory}/{C20_file_name}"
        get_soc = get_pOCV_SOC_interp_fn(C20_file_path, x_parameter="OCV")
        get_ocv = get_pOCV_SOC_interp_fn(C20_file_path, x_parameter="SOC")

        csv_files = filter(
            lambda f: f.endswith(".csv") and "C20" not in f,
            os.listdir(parsed_data_T_directory),
        )

        for csv_file in csv_files:
            try:
                csv_file_name = csv_file.split("_parsed.csv")[0]

                df = pd.read_csv(f"{parsed_data_T_directory}/{csv_file}")
                df = estimate_soc_and_ocv(df, get_soc_fn=get_soc, get_ocv_fn=get_ocv)
                df.to_parquet(
                    f"{processed_data_T_directory}/{csv_file_name}.parquet", index=False
                )

                generate_and_save_plot(
                    data_df=df,
                    save_file_path=f"{processed_data_T_directory}/{csv_file_name}_plot.png",
                    fig_title=f"{csv_file_name} @ {T}",
                )
                plt.close()
                print(f"Processed {csv_file_name} @ {T}")

            except Exception as e:
                print(f"Error processing {csv_file} - {e}")
