import os
from pathlib import Path
import numpy as np
from glob import glob

from data_readers.ceilometer import CeilometerData


cloudnetfiles = glob("/home/sm_indka/data/Celiometer/2010/cloudnet/*nc")
base_path = Path("/home/sm_indka/data/Celiometer/2010/mora_data_level_Y/")
year = 2010
start_day = np.datetime64(f"{year}-01-01", "D")
end_day = np.datetime64(f"{year}-12-31", "D")

for cloudnetfile in cloudnetfiles:
    print(cloudnetfile)
    cm = CeilometerData.from_cloudnet_file(Path(cloudnetfile))
    for day in np.arange(start_day, end_day + 1):

        day_string = np.datetime_as_string(day, unit="D").tolist()
        time_day = np.array(cm.time).astype("datetime64[D]")

        year = str(day)[:4]
        month = str(day)[5:7]
        output_path = os.path.join(base_path, year, month)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "cb_" + day_string + ".txt")

        cbh = cm.cbh_agl[:, 0][time_day == day]
        if len(cbh) == 0:
            continue
        notnan = ~np.isnan(cbh)
        time = np.datetime_as_string(cm.time[time_day == day][notnan])
        cbh = cbh[notnan] + cm.surface_height
        n_entries = len(cbh)

        with open(output_file, "a") as f:
            np.savetxt(
                f,
                np.column_stack(
                    (
                        np.repeat(cm.station_name.replace(" ", "_"), n_entries),
                        np.repeat(cm.latitude, n_entries),
                        np.repeat(cm.longitude, n_entries),
                        np.datetime_as_string(
                            cm.time[time_day == day][notnan], unit="m"
                        ).tolist(),
                        cbh,
                    )
                ),
                fmt="%s",
            )
