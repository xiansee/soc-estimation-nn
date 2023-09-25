import sys, os
sys.path.append('../')
import pandas as pd
from datetime import datetime
from soc_estimation_nn.helper import plot
import matplotlib.pyplot as plt


timestamp_col = 'Timestamp'
time_col = 'Time [min]'
voltage_col = 'Voltage [V]'
current_col = 'Current [A]'
temperature_col = 'Temperature [degC]'
capacity_col = 'Capacity [Ah]'

raw_data_directory = '../data/raw'
parsed_data_directory = '../data/parsed'


def parse_raw_data(file_path: str) -> pd.DataFrame:
    """
    Parses raw data for 3Ah LG HG2 Li-ion cell for time series time, voltage, current, temperature and capacity.
    Eample raw data available at: https://data.mendeley.com/datasets/cp3473x7xv/3

    Citation for raw data:
    Kollmeyer, Philip; Vidal, Carlos; Naguib, Mina; Skells, Michael  (2020), “LG 18650HG2 Li-ion Battery Data and 
    Example Deep Neural Network xEV SOC Estimator Script”, Mendeley Data, V3, doi: 10.17632/cp3473x7xv.3

    Parameters
    ----------
    file_path : str
        File path to csv.

    Returns
    -------
    pd.DataFrame
        DataFrame of parsed data
    """

    with open(file_path) as f:
        lines = f.readlines()

    column_index = lines.index(next(filter(lambda l: 'Time Stamp' in l, lines)))
    column_line = lines[column_index].split(',')
    data_lines = [l.split(',') for l in lines[column_index + 2:]]
    
    timestamp_data = []
    for l in data_lines:
        try:
            timestamp_data.append(datetime.strptime(l[column_line.index('Time Stamp')], '%m/%d/%Y %I:%M:%S %p'))
        except ValueError:
            timestamp_data.append(datetime.strptime(l[column_line.index('Time Stamp')], '%m/%d/%Y'))

    df = pd.DataFrame({
        timestamp_col: timestamp_data,
        time_col: [(dt - timestamp_data[0]).total_seconds() / 60 for dt in timestamp_data],
        voltage_col: [float(l[column_line.index('Voltage')]) for l in data_lines],
        current_col: [float(l[column_line.index('Current')]) for l in data_lines],
        temperature_col: [float(l[column_line.index('Temperature')]) for l in data_lines],
        capacity_col: [float(l[column_line.index('Capacity')]) for l in data_lines],
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
                'y_data': data_df[voltage_col],
                'label': 'Voltage',
            },
            {
                'x_data': data_df[time_col],
                'y_data': data_df[current_col],
                'label': 'Current',
                'plot_num': 2
            },
            {
                'x_data': data_df[time_col],
                'y_data': data_df[temperature_col],
                'label': 'Temperature',
                'plot_num': 3
            },
            {
                'x_data': data_df[time_col],
                'y_data': data_df[capacity_col],
                'label': 'Capacity',
                'plot_num': 4
            },
        ],
        x_label=time_col,
        y_label={1: voltage_col, 2: current_col, 3:temperature_col, 4: capacity_col},
        title=fig_title,
        fig_size=(10, 10),
        show_plt=False
    )

    fig.savefig(save_file_path)

    return


if __name__ == '__main__':

    temperatures = filter(lambda folder: 'degC' in folder, os.listdir(raw_data_directory))

    for T in temperatures:
        raw_data_T_directory = f'{raw_data_directory}/{T}'
        parsed_data_T_directory = f'{parsed_data_directory}/{T}'

        if not os.path.exists(parsed_data_T_directory):
            os.makedirs(parsed_data_T_directory)

        for csv_file in filter(lambda f: f.endswith('.csv'), os.listdir(raw_data_T_directory)):

            try:
                csv_file_name = csv_file.split(".csv")[0]
                df = parse_raw_data(file_path=f'{raw_data_T_directory}/{csv_file_name}.csv')

                #Handles edge case
                if csv_file_name == '571_Mixed6':
                    df = df[df[time_col] > 600]
                    df[time_col] = df[time_col] - df[time_col].iloc[0]

                df.to_csv(f'{parsed_data_T_directory}/{csv_file_name}_parsed.csv', index=False)
                generate_and_save_plot(
                    data_df=df, 
                    save_file_path=f'{parsed_data_T_directory}/{csv_file_name}_plot.png',
                    fig_title=f'{csv_file_name}_parsed @ {T}'
                )
                plt.close()
                print(f'Processed: {csv_file_name}_parsed @ {T}')

            except Exception as e:
                print(f'Error processing: {csv_file} - {e}') 

