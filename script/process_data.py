import sys, os
sys.path.append('../')
import pandas as pd
import os
from scipy.interpolate import interp1d
from soc_estimation_nn.helper import plot
import matplotlib.pyplot as plt


timestamp_col = 'Timestamp'
time_col = 'Time [min]'
voltage_col = 'Voltage [V]'
current_col = 'Current [A]'
temperature_col = 'Temperature [degC]'
capacity_col = 'Capacity [Ah]'
soc_col = 'SOC [-]'

parsed_data_directory = '../data/parsed'
processed_data_directory = '../data/processed'

normalization_settings = {
    voltage_col: (2.70, 4.25),
    current_col: (-20, 10),
    temperature_col: (-25, 45)
}
normalize_col_name = lambda col: f'Normalized {col.split(" [")[0]} [-]'
norm_voltage_col = normalize_col_name(voltage_col)
norm_current_col = normalize_col_name(current_col)
norm_temperature_col = normalize_col_name(temperature_col)


def get_pOCV_SOC_interp_fn(file_path: str) -> interp1d:
    """
    Create pseudo OCV-SOC interpolation function from slow discharge data.

    Parameters
    ----------
    file_path : str
        Path to slow discharge data

    Returns
    -------
    interp1d
        Interpolation function that takes OCV as input and returns SOC
    """

    df = pd.read_csv(file_path)
    df = df[df[current_col] < 0]
    df[capacity_col] = df[capacity_col] - df[capacity_col].iloc[0]
    df[soc_col] = 1 - abs(df[capacity_col] / df[capacity_col].iloc[-1])

    return interp1d(df[voltage_col], df[soc_col])


def estimate_soc(
    df: pd.DataFrame, 
    get_soc_fn: interp1d
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

    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize time series voltage, current and temperature data.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data

    Returns
    -------
    pd.DataFrame
        Time series data with normalized values
    """

    df = df[[time_col, voltage_col, current_col, temperature_col, soc_col]].copy(deep=True)

    for col, norm_range in normalization_settings.items():
        min_value, max_value = norm_range
        df[col] = (df[col] - min_value) / (max_value - min_value) * 2 - 1

    df = df.rename(columns={
        voltage_col: norm_voltage_col,
        current_col: norm_current_col,
        temperature_col: norm_temperature_col
    })

    return df


def generate_and_save_plot(
    data_df: pd.DataFrame,
    save_file_path: str,
    fig_title: str = '',
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
                'x_data': data_df[time_col],
                'y_data': data_df[norm_voltage_col],
                'label': 'Normalized Voltage',
            },
            {
                'x_data': data_df[time_col],
                'y_data': data_df[norm_current_col],
                'label': 'Normalized Current',
                'plot_num': 2
            },
            {
                'x_data': data_df[time_col],
                'y_data': data_df[norm_temperature_col],
                'label': 'Normalized Temperature',
                'plot_num': 3
            },
            {
                'x_data': data_df[time_col],
                'y_data': data_df[soc_col],
                'label': 'SOC',
                'plot_num': 4
            },
        ],
        x_label=time_col,
        y_label={
            1: norm_voltage_col, 
            2: norm_current_col, 
            3: norm_temperature_col, 
            4: soc_col
        },
        title=fig_title,
        fig_size=(10, 10),
        show_plt=False
    )

    fig.savefig(save_file_path)

    return

  
if __name__ == '__main__':

    temperatures = filter(lambda folder: 'degC' in folder, os.listdir(parsed_data_directory))

    for T in temperatures:
        parsed_data_T_directory = f'{parsed_data_directory}/{T}'
        processed_data_T_directory = f'{processed_data_directory}/{T}'

        if not os.path.exists(processed_data_T_directory):
            os.makedirs(processed_data_T_directory)

        csv_files = filter(lambda f: f.endswith('.csv'), os.listdir(parsed_data_T_directory))
        C20_file_name = next(filter(lambda f: 'C20' in f, csv_files))
        C20_file_path = f'{parsed_data_T_directory}/{C20_file_name}'
        get_soc = get_pOCV_SOC_interp_fn(C20_file_path)

        csv_files = filter(lambda f: f.endswith('.csv') and 'C20' not in f, os.listdir(parsed_data_T_directory))

        for csv_file in csv_files:

            try:
                csv_file_name = csv_file.split(".csv")[0]

                df = pd.read_csv(f'{parsed_data_T_directory}/{csv_file}')
                df = estimate_soc(df, get_soc_fn=get_soc)
                df = normalize_data(df)
                df.to_parquet(f'{processed_data_T_directory}/{csv_file_name}.parquet', index=False)

                generate_and_save_plot(
                    data_df=df, 
                    save_file_path=f'{processed_data_T_directory}/{csv_file_name}_plot.png',
                    fig_title=f'{csv_file_name}_processed @ {T}'
                )
                plt.close()
                print(f'Processed {csv_file_name}_processed @ {T}')

            except Exception as e:
                print(f'Error processing {csv_file} - {e}')
