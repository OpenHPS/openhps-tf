# Process the raw fingerprint data
# This script will perform the following tasks
# on **/raw/*.csv:
# - Aggregate results for each datapoint and orientation
# - Add the start and end timestamp of the aggregated results
#
# This script is provided as extra context to the processed results
# different aggregations are possible.

import os, json
import pandas as pd
import numpy as np

## Replace with "../test" for test data
dataset_dir = "../train"
# Range of the aggregation (in seconds). -1 will ignore this setting.
duration = -1

dir_path = os.path.dirname(os.path.realpath(__file__))
raw_dir = os.path.join(dir_path, dataset_dir, "raw")
processed_dir  = os.path.join(dir_path, dataset_dir, "aggregated")

# Load raw data
wlan_fingerprints = pd.read_csv(os.path.join(raw_dir, "wlan_fingerprints.csv"))
ble_fingerprints = pd.read_csv(os.path.join(raw_dir, "ble_fingerprints.csv"))
imu_fingerprints = pd.read_csv(os.path.join(raw_dir, "imu_fingerprints.csv"))

# Aggregate result
wlan_fingerprints_grouped = wlan_fingerprints.replace(100, np.nan).groupby(by=['X', 'Y', 'ORIENTATION'], as_index=False)
ble_fingerprints_grouped = ble_fingerprints.replace(100, np.nan).groupby(by=['X', 'Y', 'ORIENTATION'], as_index=False)
imu_fingerprints_grouped = imu_fingerprints.groupby(by=['X', 'Y', 'ORIENTATION'], as_index=False)

if duration > 0:
    start = imu_fingerprints_grouped['TIMESTAMP'].min()
    end_max = imu_fingerprints_grouped['TIMESTAMP'].max()
    end = start
    end['TIMESTAMP'] = start['TIMESTAMP'] + (duration * 1000)
    for i,point in end.iterrows():
        start_point = start[(start['X'] == point['X']) &
            (start['Y'] == point['Y']) &
            (start['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        end_point = end[(end['X'] == point['X']) &
            (end['Y'] == point['Y']) &
            (end['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        end_max_point = end_max[(end_max['X'] == point['X']) &
            (end_max['Y'] == point['Y']) &
            (end_max['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        wlan_fingerprints = wlan_fingerprints[
            (wlan_fingerprints['TIMESTAMP'] <= end_point) |
            (wlan_fingerprints['TIMESTAMP'] > end_max_point)
        ]
wlan_fingerprints_grouped = wlan_fingerprints.replace(100, np.nan).groupby(by=['X', 'Y', 'ORIENTATION'], as_index=False)
wlan_fingerprints = wlan_fingerprints_grouped.agg(lambda x: x.mean(skipna=True)).fillna(100)
wlan_fingerprints['TIMESTAMP'] = round(wlan_fingerprints['TIMESTAMP'])
wlan_fingerprints['TIMESTAMP'] = wlan_fingerprints['TIMESTAMP'].astype(np.uint64)

if duration > 0:
    for i,point in end.iterrows():
        start_point = start[(start['X'] == point['X']) &
            (start['Y'] == point['Y']) &
            (start['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        end_point = end[(end['X'] == point['X']) &
            (end['Y'] == point['Y']) &
            (end['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        end_max_point = end_max[(end_max['X'] == point['X']) &
            (end_max['Y'] == point['Y']) &
            (end_max['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        ble_fingerprints = ble_fingerprints[
            (ble_fingerprints['TIMESTAMP'] <= end_point) |
            (ble_fingerprints['TIMESTAMP'] > end_max_point)
        ]
ble_fingerprints_grouped = ble_fingerprints.replace(100, np.nan).groupby(by=['X', 'Y', 'ORIENTATION'], as_index=False)
ble_fingerprints = ble_fingerprints_grouped.agg(lambda x: x.mean(skipna=True)).fillna(100)
ble_fingerprints['TIMESTAMP'] = round(ble_fingerprints['TIMESTAMP'])
ble_fingerprints['TIMESTAMP'] = ble_fingerprints['TIMESTAMP'].astype(np.uint64)

if duration > 0:
    for i,point in end.iterrows():
        start_point = start[(start['X'] == point['X']) &
            (start['Y'] == point['Y']) &
            (start['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        end_point = end[(end['X'] == point['X']) &
            (end['Y'] == point['Y']) &
            (end['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        end_max_point = end_max[(end_max['X'] == point['X']) &
            (end_max['Y'] == point['Y']) &
            (end_max['ORIENTATION'] == point['ORIENTATION'])]['TIMESTAMP'].iloc[0]
        imu_fingerprints = imu_fingerprints[
            (imu_fingerprints['TIMESTAMP'] <= end_point) |
            (imu_fingerprints['TIMESTAMP'] > end_max_point)
        ]
imu_fingerprints_grouped = imu_fingerprints.replace(100, np.nan).groupby(by=['X', 'Y', 'ORIENTATION'], as_index=False)
imu_fingerprints = imu_fingerprints_grouped.mean()
imu_fingerprints['TIMESTAMP'] = round(imu_fingerprints['TIMESTAMP'])
imu_fingerprints['TIMESTAMP'] = imu_fingerprints['TIMESTAMP'].astype(np.uint64)

# Create combined results
combined = wlan_fingerprints.merge(ble_fingerprints, on=['X', 'Y', 'ORIENTATION'], how='left', suffixes=('_WLAN', '_BLE')).fillna(100)

# Save processed data
wlan_fingerprints.to_csv(os.path.join(processed_dir, "wlan_fingerprints.csv"), index=False)
ble_fingerprints.to_csv(os.path.join(processed_dir, "ble_fingerprints.csv"), index=False)
imu_fingerprints.to_csv(os.path.join(processed_dir, "imu_fingerprints.csv"), index=False)
combined.to_csv(os.path.join(processed_dir, "wlan-ble_fingerprints.csv"), index=False)
