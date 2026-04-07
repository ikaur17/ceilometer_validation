from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@dataclass
class CeilometerData:
    latitude: float
    longitude: float
    station_name: str
    surface_height: float
    time: np.ndarray
    cbh_amsl: np.ndarray
    cth_amsl: Optional[np.ndarray]

    @classmethod
    def from_cloudnet_file(cls, file: Path) -> CeilometerData:
        with xr.open_dataset(file.as_posix()) as ds:
            return cls(
                ds.latitude.data,
                ds.longitude.data,
                ds.attrs["location"],
                ds.altitude.data,
                ds.time.data,
                ds.cloud_base_height_amsl.data,
                ds.cloud_top_height_amsl.data,
            )

    def from_mora_file(cls, csvfile: Path) -> CeilometerData:
        with pd.read_csv(csvfile.as_posix(), delimiter=";", skiprows=9) as df:
            df_subset = df[df["Level from"] == 1.0]  # only lowest level
            station_meta_deta = pd.read_csv(
                csvfile.as_posix(),
                delimiter=";",
                skiprows=range(2),
                nrows=1,
            )
            return cls(
                station_meta_deta.iloc[0]["Latitude"],
                station_meta_deta.iloc[0]["Latitude"],
                station_meta_deta.iloc[0]["Station name"],
                station_meta_deta.iloc[0]["Height"],
                get_mora_time(df_subset),
                df_subset["Database Value"].to_numpy(),
                None,
            )


def get_mora_time(df):
    return np.array(
        [
            convert_time_to_ns(time_string, offset)
            for time_string, offset in zip(df["TimeTick"], df["Offset"])
        ]
    )


def extract_offset_in_timedelta(offset_string: str) -> timedelta:
    number = int(offset_string[2:-1])
    unit = offset_string[-1]
    if unit == "S":
        return timedelta(seconds=number)
    if unit == "M":
        return timedelta(minutes=number)
    else:
        raise ValueError(f"This offset is not supported {offset_string}")


def convert_time_to_ns(time_string: str, offset: str) -> np.ndarray:
    return np.datetime64(
        datetime.strptime(time_string, "%Y-%m-%d %H:%M")
        - extract_offset_in_timedelta(offset),
        "ns",
    )
