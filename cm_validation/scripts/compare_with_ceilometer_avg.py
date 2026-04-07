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
from pyresample import geometry, kd_tree
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


def spatial_average(
    source_lons,
    source_lats,
    source_values,
    center_lon,
    center_lat,
    radius=15000,
):
    # 1. Define source and target locations
    source_def = geometry.SwathDefinition(lons=source_lons, lats=source_lats)
    target_def = geometry.SwathDefinition(lons=[center_lon], lats=[center_lat])

    # 2. Get neighbor info: all pixels within radius
    valid_input_idx, valid_output_idx, index_array, distance_array = (
        kd_tree.get_neighbour_info(
            source_geo_def=source_def,
            target_geo_def=target_def,
            radius_of_influence=radius,
            neighbours=10000,  # Large enough to catch all in radius
            epsilon=0,  # Strict radius
        )
    )
    return index_array, distance_array


def percent_precip(data):
    rainy = data > 0
    return calculate_percent_true(rainy)


def calculate_percent_true(data):
    return (np.sum(data) / len(data)) * 100


def percent_multilayer(data):
    return calculate_percent_true(data)


def get_matched_ceilometer_data_avg(
    ds: xr.Dataset, cm, indices: tuple[int, int], window_minutes=30
) -> tuple[float, float, float, float, float, float]:
    scan_index, _ = indices

    time_scan = np.linspace(
        ds.time_bnds.data[0, 0].astype("int64"),
        ds.time_bnds.data[0, 1].astype("int64"),
        len(ds.ny.data),
    ).astype("datetime64[ns]")
    t_center = time_scan[scan_index]

    time_diff = (cm.time - t_center).astype("timedelta64[m]").astype(float)

    # Filter within ±window
    time_mask = np.abs(time_diff) <= window_minutes
    if not np.any(time_mask):
        return None
    print(f"min time diff is {np.min(time_diff)}")
    nanmask = ~np.isnan(cm.cbh_agl[:, 0][time_mask])
    cbh_vals = cm.cbh_agl[:, 0][time_mask][nanmask]
    cth_vals = cm.cth_agl[time_mask][nanmask]
    time_diff = time_diff[time_mask][nanmask]
    print(cth_vals.shape, cm.multilayer.shape, cm.cbh_agl.shape)
    if np.std(cbh_vals) < 1000.0:
        return (
            np.nanmean(cbh_vals),
            np.nanmean(cth_vals),
            percent_multilayer(cm.multilayer[time_mask][nanmask]),
            np.nanmean(cm.cloudamount[time_mask][nanmask]),
            np.max(cm.cloudamount[time_mask][nanmask]),
            np.min(cm.cloudamount[time_mask][nanmask]),
            percent_precip(cm.precip[time_mask][nanmask]),
        )
    else:
        return -5000.0, -5000.0, -5000, -5000.0, -5000.0, -5000.0, -5000.0


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
    radius: float = 5000.0,
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
            return index_array[0]
        return None
    else:
        return None


def find_nearest_index_in_time(cm_time: np.ndarray, sat_time: np.ndarray) -> int:
    diff = np.abs(cm_time.astype("int64") - sat_time.astype("int64"))
    index = np.argmin(diff)
    return index


def get_cth_data(file: str, indices: tuple[int, int]) -> np.ndarray:
    cthfile = os.path.join(
        CBHPATH,
        os.path.basename(file).replace("CBH", "CTTH"),
    )
    ii, jj = indices
    with xr.open_dataset(cthfile) as cth:
        return float(cth.ctth_alti[0].data[ii, jj])


def get_satzenith_angle(file: str, indices: tuple[int, int]) -> np.ndarray:
    imagerfile = glob.glob(
        os.path.join(
            "/nobackup/smhid20/users/sm_indka/data/pps/import/IMAGER_data/",
            "*" + os.path.basename(file)[15:],
        )
    )[0]
    ii, jj = indices
    with xr.open_dataset(imagerfile) as im:
        return im.satzenith.data[0, ii, jj]


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
        index_array, distance_array = spatial_average(
            source_lons=ds.lon.data,
            source_lats=ds.lat.data,
            source_values=ds.cbh_alti.data[0],
            center_lon=cm.longitude,
            center_lat=cm.latitude,
            radius=5000,  # 5 km
        )

        flat_values = ds.cbh_alti.data[0].flatten()
        dist_mask = np.isfinite(distance_array)
        valid_values = flat_values[index_array[dist_mask]]
        cbh_err = ds.cbh_alti_err.data[0].flatten()[index_array[dist_mask]]
        z_sat = ds.surface_alti[0].data.flatten()[index_array[dist_mask]]
        mask = np.isfinite(valid_values)

        # indices = get_nearest_index_in_space(
        #     cm.latitude, cm.longitude, ds.lat.data, ds.lon.data
        # )
        # if indices is None:
        #     return None
        # else:
        if np.sum(~mask) == 0:
            return None
        else:
            valid_values = valid_values[mask]
            z_sat = np.mean(z_sat[mask])
            cbh_sat = np.mean(valid_values)
            cbh_sat_err = np.median(cbh_err[mask])

            # sigma = 7500.0
            # weights = np.exp(-0.5 * (valid_distances / sigma) ** 2)
            # weights /= weights.sum()
            # cbh_sat = np.sum(weights * valid_values)

            indices = np.unravel_index(index_array[0][0], ds.cbh_alti.data[0].shape)
            ii, jj = indices

            result = get_matched_ceilometer_data_avg(
                ds,
                cm,
                indices,
                window_minutes=20,
            )
            if result is None:
                cbh_cm = cth_cm = multilayer = ca_mean = ca_max = ca_min = (
                    precip
                ) = -999.9
            else:
                cbh_cm, cth_cm, multilayer, ca_mean, ca_max, ca_min, precip = result
            return {
                "cbh_sat": float(cbh_sat),
                "cbh_sat_err": cbh_sat_err,
                "cth": get_cth_data(file, indices),
                "cbh_cm": cbh_cm,
                "z_cm": cm.surface_height,
                "z_sat": z_sat,
                "sunzenith": get_sunzenith_angle(file, indices),
                "satzenith": get_satzenith_angle(file, indices),
                "time": ds.time.data[0],
                "sat": get_sat_name(file),
                "multilayer": multilayer,
                "ca_mean": ca_mean,
                "ca_max": ca_max,
                "ca_min": ca_min,
                "precip": precip,
            }


def run_process_for_one_day(cm, cbhfiles) -> list:
    return [
        data
        for file in cbhfiles[:]
        if (data := extract_matched_data_from_file(file, cm)) is not None
    ]


def prepare_cm_data(station, date):
    if station in CLOUDNET_STATIONS:
        try:
            cmfile = Path(os.path.join(CMPATH, f"{date}_{station}_classification.nc"))
            return CeilometerData.from_cloudnet_file(cmfile)
        except:
            return None
    if station in MORA_STATIONS.keys():
        cbfile = Path(os.path.join(CMPATH, f"CloudBase_data_{station}.csv"))
        cafile = Path(os.path.join(CMPATH, f"TotalCloudAmount_data_{station}.csv"))
        precipfile = Path(os.path.join(CMPATH, f"Precipitation_data_{station}.csv"))
        cm = CeilometerData.from_mora_file(cbfile, cafile, precipfile)
        return cm


def run_process(station, dates):
    print(f"doing station {station}")
    outfilename = f"collocated_data_{station}_avg_all_5km_with_precip_finetuned.pickle"

    if Path(outfilename).exists():
        print(f"{outfilename} already exists — skipping processing.")
        return

    matchups_dict_list = [
        run_process_for_one_day(
            cm_data,
            glob.glob(os.path.join(CBHPATH, f"*CBH*{date}*nc")),
        )
        for date in dates
        if (cm_data := prepare_cm_data(station, date)) is not None
    ]

    merged_dict = defaultdict(list)
    print(matchups_dict_list)
    for matchup_one_day in matchups_dict_list:
        for d in matchup_one_day:
            for key, value in d.items():
                merged_dict[key].append(value)

    with open(outfilename, "wb") as f:
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
