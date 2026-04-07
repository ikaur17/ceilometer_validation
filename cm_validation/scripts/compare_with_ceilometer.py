# %%
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional
import re
import xarray as xr
import numpy as np
import glob
import pyresample
import pickle
from data_readers.ceilometer import CeilometerData
from data_readers.config import MORA_STATIONS, CLOUDNET_STATIONS, CMPATH, CBHPATH


# %%
def check_if_in_domain(
    clat: float, clon: float, slats: np.ndarray, slons: np.ndarray
) -> bool:
    if (
        np.nanmin(slons) < clon
        and np.nanmax(slons) > clon
        and np.nanmin(slats) < clat
        and np.nanmax(slats) > clat
    ):
        return True
    return False


def get_nearest_index_in_space(
    cm_lat: float,
    cm_lon: float,
    sat_lats: np.ndarray,
    sat_lons: np.ndarray,
    buffer_deg=1.0,
) -> Optional[tuple[int, int]]:
    lat_mask = (sat_lats >= cm_lat - buffer_deg) & (sat_lats <= cm_lat + buffer_deg)
    lon_mask = (sat_lons >= cm_lon - buffer_deg) & (sat_lons <= cm_lon + buffer_deg)
    subset_mask = lat_mask & lon_mask

    if np.all(~subset_mask):
        return None
    else:
        flat_idx_in_subset = calculate_nearest_index_in_space(
            np.array([cm_lat]),
            np.array([cm_lon]),
            sat_lats[subset_mask],
            sat_lons[subset_mask],
        )
        if flat_idx_in_subset is not None:
            subset_indices = np.where(subset_mask)
            return (
                subset_indices[0][flat_idx_in_subset],
                subset_indices[1][flat_idx_in_subset],
            )
        return None


def calculate_nearest_index_in_space(
    cm_lat: np.ndarray,
    cm_lon: np.ndarray,
    sat_lats: np.ndarray,
    sat_lons: np.ndarray,
    radius: float = 500.0,
    nneighbours: int = 1,
) -> Optional[tuple[int, int]]:
    grid = pyresample.geometry.SwathDefinition(lats=sat_lats, lons=sat_lons)
    swath = pyresample.geometry.SwathDefinition(lons=[cm_lon], lats=[cm_lat])

    _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
        source_geo_def=grid,
        target_geo_def=swath,
        radius_of_influence=radius,
        neighbours=nneighbours,
    )
    if np.all(np.isfinite(distance_array)):
        if distance_array[0] < 500:
            print(distance_array[0])
            return index_array[0]
        return None
    else:
        return None


def find_nearest_index_in_time(cm_time: np.ndarray, sat_time: np.ndarray) -> int:
    diff = np.abs(cm_time.astype("int64") - sat_time.astype("int64"))
    index = np.argmin(diff)
    if diff[index] < 60 * 2 * 1e9:
        return index
    else:
        return None


def get_cth_data(file: str, indices: tuple[int, int]) -> np.ndarray:
    cthfile = os.path.join(
        CBHPATH,
        os.path.basename(file).replace("CBH", "CTTH"),
    )
    ii, jj = indices
    with xr.open_dataset(cthfile) as cth:
        return float(cth.ctth_alti[0].data[ii, jj])


def get_matched_ceilometer_data(
    ds: xr.Dataset, cm: xr.Dataset, indices: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scan_index, _ = indices
    time_scan = np.linspace(
        ds.time_bnds.data[0, 0].astype("int64"),
        ds.time_bnds.data[0, 1].astype("int64"),
        len(ds.ny.data),
    ).astype("datetime64[ns]")
    index = find_nearest_index_in_time(cm.time, time_scan[scan_index])
    if index is None:
        return None
    else:
        return (
            cm.cbh_agl[index],
            cm.cth_agl[index],
            cm.multilayer[index],
        )


def get_sunzenith_angle(file: str, indices: tuple[int, int]) -> np.ndarray:
    imagerfile = glob.glob(
        os.path.join(
            "/nobackup/smhid20/users/sm_indka/data/pps/import/IMAGER_data/",
            "*" + os.path.basename(file)[15:],
        )
    )[0]
    ii, jj = indices
    with xr.open_dataset(imagerfile) as im:
        return im.sunzenith.data[0, ii, jj]


def get_sat_name(file: str) -> str:
    return re.split(r"_", os.path.basename(file))[3]


def extract_matched_data_from_file(file: str, cm: CeilometerData) -> dict:
    with xr.open_dataset(file) as ds:
        if not check_if_in_domain(cm.latitude, cm.longitude, ds.lat.data, ds.lon.data):
            return None
        indices = get_nearest_index_in_space(
            cm.latitude, cm.longitude, ds.lat.data, ds.lon.data
        )
        if indices is None:
            return None
        else:
            ii, jj = indices
            print(
                f"{file} matches with {cm.station_name}, {indices}, {cm.latitude}, {cm.longitude}, {ds.lat.data[ii, jj], ds.lon.data[ii, jj]}"
            )
            result = get_matched_ceilometer_data(ds, cm, indices)
            if result is None:
                cbh_cm, cth_cm, multilayer = np.ones([4]) * -999.9, -999.9, False
            else:
                cbh_cm, _, multilayer = result
                print(ds.cbh_alti[0].data[ii, jj], cbh_cm)
                return {
                    "cbh_sat": float(ds.cbh_alti[0].data[ii, jj]),
                    "cbh_sat_err": float(ds.cbh_alti_err[0].data[ii, jj]),
                    "cth": get_cth_data(file, indices),
                    "cbh_cm": cbh_cm,
                    "z_cm": cm.surface_height,
                    "z_sat": ds.surface_alti[0].data[ii, jj],
                    "sunzenith": get_sunzenith_angle(file, indices),
                    "time": ds.time.data[0],
                    "sat": get_sat_name(file),
                    "multilayer": multilayer,
                }


def run_process_for_one_day(cm, cbhfiles) -> list:
    return [
        data
        for file in cbhfiles[:]
        if (data := extract_matched_data_from_file(file, cm)) is not None
    ]


def prepare_cm_data(station, date):
    if station in CLOUDNET_STATIONS:
        cmfile = Path(os.path.join(CMPATH, f"{date}_{station}_classification.nc"))
        return CeilometerData.from_cloudnet_file(cmfile)
    if station in MORA_STATIONS.keys():
        cmfile = Path(os.path.join(CMPATH, f"CloudBase_data_{station}.csv"))
        return CeilometerData.from_mora_file(cmfile)


def run_process(station, dates):
    print(f"doing station {station}")
    matchups_dict_list = [
        run_process_for_one_day(
            prepare_cm_data(station, date),
            glob.glob(os.path.join(CBHPATH, f"*CBH*{date}*nc")),
        )
        for date in dates
    ]

    merged_dict = defaultdict(list)
    print(matchups_dict_list)
    for matchup_one_day in matchups_dict_list:
        for d in matchup_one_day:
            for key, value in d.items():
                merged_dict[key].append(value)

    with open(f"collocated_data_{station}_cbh_level.pickle", "wb") as f:
        pickle.dump(merged_dict, f)


if __name__ == "__main__":
    # CLOUDNET_STATIONS = [
    #     # "bucharest",
    #     # "cabauw",
    #     # "cluj",
    #     # "galati",
    #     # "granada",
    #     "hyytiala",
    #     # "juelich",
    #     # "leipzig",
    #     # "limassol",
    #     # "lindenberg",
    #     # "munich",
    #     "norunda",
    #     "ny-alesund",
    #     # "palaiseau",
    #     # "payerne",
    #     # "potenza",
    # ]
    # MORA_STATIONS = []
    CMPATH = "/home/sm_indka/data/Celiometer/"
    CBHPATH = "/nobackup/smhid20/users/sm_indka/data/pps/export/"
    # CBHPATH = "/nobackup/smhid20/proj/foua/data/NWCSAF/CBH_FMI_MAR25/FMI_CBH_PPS"
    DATES = np.arange(20250301, 20250318, 1).astype("str")

    for station in CLOUDNET_STATIONS:
        print(f"Doing {station}")
        run_process(station, DATES)

    for station in MORA_STATIONS.keys():
        print(f"Doing {station}")
        run_process(station, DATES)

    # def plot_data(station, data):
    #     fig, ax = plt.subplots(1, 1, figsize=[12, 6])

    #     lidarfiles = []
    #     for date in dates:
    #         lidarfiles.append(
    #             f"/home/sm_indka/data/Celiometer/{date}_{station}_classification.nc"
    #         )
    #     with xr.open_mfdataset(lidarfiles, combine="by_coords") as cm:
    #         # # norm = mcolors.LogNorm(vmin=1e-7, vmax=1e-4)
    #         # # with xr.open_dataset(betafile) as ds:
    #         # #     ds.beta_smooth.T.plot.pcolormesh(ax=ax, norm=norm)
    #         # with xr.open_dataset(lidarfile) as cm:
    #         cm = cm.resample(time="5min").mean()
    #         ax.scatter(
    #             cm.time.data,
    #             cm.cloud_base_height_amsl.data,
    #             s=3,
    #             c="g",
    #             label="CBH Lidar",
    #         )
    #         ax.scatter(
    #             cm.time.data,
    #             cm.cloud_top_height_amsl.data,
    #             s=3,
    #             c="b",
    #             label="CTH Lidar",
    #         )
    #         ax.scatter(
    #             data[station]["time"],
    #             data[station]["cbh_sat"],
    #             s=10,
    #             c="r",
    #             label="CBH PPS",
    #         )
    #         ax.scatter(
    #             data[station]["time"],
    #             data[station]["cth_sat"],
    #             s=10,
    #             c="k",
    #             label="CTH PPS",
    #         )
    #         # ax.set_ylim([0, 8000])
    #     cm.close()
    #     ax.legend()
    #     ax.set_ylabel("Height [m]")
    #     ax.set_xlabel("Time")
    #     fig.suptitle(station)
    #     fig.savefig(f"{station}_all_amsl.png")
