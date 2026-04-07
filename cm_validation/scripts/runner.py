import sys
from compare_with_ceilometer import run_process
from data_readers.config import CLOUDNET_STATIONS, MORA_STATIONS, DATES

#stations = CLOUDNET_STATIONS + list(MORA_STATIONS.keys())
stations = list(MORA_STATIONS.keys()) + CLOUDNET_STATIONS

station = stations[int(sys.argv[1])]
run_process(station, DATES)

