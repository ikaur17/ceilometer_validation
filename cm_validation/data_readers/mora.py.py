from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


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


def read_csv(csvfile: Path):
    df = pd.read_csv(csvfile.as_posix(), delimiter=";", skiprows=9)
    df_subset = df[df["Level from"] == 1.0]  # only lowest level
    station_meta_deta = pd.read_csv(
        csvfile.as_posix(),
        delimiter=";",
        skiprows=range(2),
        nrows=1,
    )
    latitude = station_meta_deta.iloc[0]["Latitude"]
    longitude = station_meta_deta.iloc[0]["Latitude"]
    surface_height = station_meta_deta.iloc[0]["Height"]
    station_name = station_meta_deta.iloc[0]["Station name"]
    time = np.array(
        [
            convert_time_to_ns(time_string, offset)
            for time_string, offset in zip(df["TimeTick"], df["Offset"])
        ]
    )
