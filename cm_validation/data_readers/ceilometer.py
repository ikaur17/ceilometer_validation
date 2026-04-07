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
    cbh_agl: np.ndarray
    cth_agl: np.ndarray
    multilayer: np.ndarray
    cloudamount: np.ndarray
    precip: np.ndarray

    @classmethod
    def from_cloudnet_file(cls, ncfile: Path) -> "CeilometerData":
        with xr.open_dataset(ncfile.as_posix()) as ds:
            return cls(
                ds.latitude.data,
                ds.longitude.data,
                ds.attrs["location"],
                ds.altitude.data,
                ds.time.data,
                ds.cloud_base_height_agl.data.reshape(-1, 1),
                ds.cloud_top_height_agl.data.reshape(-1, 1),
                np.ones_like(ds.cloud_base_height_agl.data) * -999.9,
                np.ones_like(ds.cloud_base_height_agl.data) * -999.9,
                np.ones_like(ds.cloud_base_height_agl.data) * -999.9,
            )

    @classmethod
    def from_mora_file(
        cls, cbfile: Path, cloudamountfile: Optional[Path], precipfile: Optional[Path]
    ) -> "CeilometerData":
        df_subset, station_meta_data = get_cloudbase_data(cbfile)
        if cloudamountfile is None:
            cloudamount = (-999.9 * np.ones(df_subset["time"].to_numpy().shape),)
        else:
            ca_subset = get_cloudamount_data(cloudamountfile)
            df_subset = ca_subset.merge(df_subset, how="left", on="time")
            cloudamount = df_subset["cloudamount"].to_numpy()
        if precipfile is None:
            precip = (-999.9 * np.ones(df_subset["time"].to_numpy().shape),)
        else:
            precip = get_precip_data(precipfile)
            df_subset = precip.merge(df_subset, how="right", on="time")
            precip = df_subset["precip"].to_numpy()

        return cls(
            station_meta_data.iloc[0]["Latitude"],
            station_meta_data.iloc[0]["Longitude"],
            station_meta_data.iloc[0]["Station name"],
            station_meta_data.iloc[0]["Height"],
            df_subset["time"].to_numpy(),
            df_subset[
                [
                    "height_level_1",
                    "height_level_2",
                    "height_level_3",
                    "height_level_4",
                ]
            ].to_numpy(),
            -999.9 * np.ones(df_subset["time"].to_numpy().shape),
            df_subset["multilayer"].to_numpy(),
            cloudamount,
            precip,
        )


def get_cloudamount_data(cafile: Path) -> pd.DataFrame:
    ca = pd.read_csv(cafile.as_posix(), delimiter=";", skiprows=9)
    ca.loc[:, "time"] = get_mora_time(ca)
    ca_subset = ca[["time", "Database Value"]]
    ca_subset.rename(columns={"Database Value": "cloudamount"}, inplace=True)

    return ca_subset


def get_precip_data(precipfile: Path) -> pd.DataFrame:
    pc = pd.read_csv(precipfile.as_posix(), delimiter=";", skiprows=9)
    pc["time"] = get_mora_time(pc)
    pc1 = pc[["time", "Database Value"]]
    pc1.rename(columns={"Database Value": "precip"}, inplace=True)
    pc1.set_index("time", inplace=True)
    return pc1.resample("5min").sum()


def get_cloudbase_data(cbfile: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df = pd.read_csv(cbfile.as_posix(), delimiter=";", skiprows=9)
    df_subset = consolidate_info_on_multiple_layers(df)
    multilayer = get_multilayer_flag(df)
    station_meta_data = pd.read_csv(
        cbfile.as_posix(),
        delimiter=";",
        skiprows=range(2),
        nrows=1,
    )
    df_subset.loc[:, "multilayer"] = multilayer
    return df_subset, station_meta_data

def consolidate_info_on_multiple_layers(df):
    subsets = {}
    for level in range(0, 5):
        subset = df[df["Level from"] == float(level)].copy()
        subset.loc[:, "time"] = get_mora_time(subset)
        subset = subset[["time", "Database Value"]] 
        subset.rename(columns={"Database Value": f"height_level_{level}"}, inplace=True)
        subsets[level] = subset

    merged = subsets[0]
    for level in range(1, 5):
        merged = merged.merge(subsets[level], on="time", how="outer")

    merged.sort_values("time")
    merged["height_level_1"] = merged["height_level_0"].combine_first(merged["height_level_1"])       

    return merged


def get_multilayer_flag(df):
    df_subset = df[df["Level from"] == 1.0]  # lowest level
    time1 = get_mora_time(df_subset)
    df_subset.loc[:, "time"] = time1
    df_subset_2 = df[df["Level from"] == 2.0]  # second level
    time2 = get_mora_time(df_subset_2)
    df_subset_2.loc[:, "time"] = time2
    return df_subset["time"].isin(df_subset_2["time"])


def get_mora_time(df: pd.DataFrame) -> np.ndarray:
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
