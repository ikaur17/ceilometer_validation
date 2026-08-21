import os
from pathlib import Path
import numpy as np
from glob import glob

from data_readers.ceilometer import CeilometerData
from tqdm import tqdm


def write_data_to_file(
    output_file,
    station_name,
    latitude,
    longitude,
    time,
    cbh,
):
    with open(output_file, "a") as f:
        np.savetxt(
            f,
            np.column_stack(
                (
                    station_name,
                    latitude,
                    longitude,
                    time,
                    cbh,
                )
            ),
            fmt="%s",
        )


def read_data(file) -> CeilometerData:
    """Read MORA or Cloudnet data."""

    if file.endswith("csv"):
        return CeilometerData.from_mora_file(Path(file))

    elif file.endswith("nc"):
        return CeilometerData.from_cloudnet_file(Path(file))

    else:
        raise ValueError("The format not supported!")


def handle_nans(cbh, times) -> tuple[np.ndarray, np.ndarray]:

    if cbh.shape[1] == 1:
        notnan = ~np.isnan(cbh[:, 0])
        no_nans_cbh = cbh[notnan, 0]
        times = np.datetime_as_string(times[notnan], unit="m").tolist()

    elif cbh.shape[1] > 1:

        isnan = np.isnan(cbh[:, 0])
        no_nans_cbh = cbh[:, 0]
        no_nans_cbh[isnan] = cbh[:, 1][isnan]
        times = np.datetime_as_string(times, unit="m").tolist()

    else:
        raise ValueError("Dimension not supported!")

    return no_nans_cbh, times


def do_morafile(ceilometerfile, start_day, end_day):

    cm = read_data(ceilometerfile)

    for day in np.arange(start_day, end_day + 1):

        day_string = np.datetime_as_string(day, unit="D").tolist()
        time_day = np.array(cm.time).astype("datetime64[D]")

        year = str(day)[:4]
        month = str(day)[5:7]
        output_path = os.path.join(base_path, year, month)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "cb_" + day_string + ".txt")

        daymask = time_day == day
        cbh = cm.cbh_agl[daymask, :]
        times = cm.time[daymask]

        if len(cbh) == 0:
            continue

        cbh, times = handle_nans(cbh, times)
        cbh += cm.surface_height

        n_entries = len(cbh)

        write_data_to_file(
            output_file,
            np.repeat(cm.station_name.replace(" ", "_"), n_entries),
            np.repeat(cm.latitude, n_entries),
            np.repeat(cm.longitude, n_entries),
            times,
            cbh,
        )


def do_cloudnetfile(ceilometerfile):

    cm = read_data(ceilometerfile)

    day_string = np.datetime_as_string(cm.time[0], unit="D").tolist()

    year = str(day_string)[:4]
    month = str(day_string)[5:7]
    output_path = os.path.join(base_path, year, month)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "cb_" + day_string + ".txt")

    cbh = cm.cbh_agl
    times = cm.time
    cbh, times = handle_nans(cbh, times)
    cbh += cm.surface_height

    n_entries = len(cbh)

    write_data_to_file(
        output_file,
        np.repeat(cm.station_name.replace(" ", "_"), n_entries),
        np.repeat(cm.latitude, n_entries),
        np.repeat(cm.longitude, n_entries),
        times,
        cbh,
    )


morafiles = glob(
    "/home/sm_indka/data/Celiometer/2010/mora_data_level_Y/csv_files/CloudBase*csv"
)
cloudnetfiles = glob("/home/sm_indka/data/Celiometer/2010/cloudnet/*nc")

base_path = Path("/home/sm_indka/data/Celiometer/2010/mora_data_level_Y/2010_cloudnet_mora/")

year = 2010
start_day = np.datetime64(f"{year}-01-01", "D")
end_day = np.datetime64(f"{year}-12-31", "D")

for morafile in tqdm(morafiles):
    do_morafile(morafile, start_day, end_day)

for cloudnetfile in tqdm(cloudnetfiles):
    do_cloudnetfile(cloudnetfile)
