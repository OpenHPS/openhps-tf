import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
wlan_aps = pd.read_csv(os.path.join(dir_path, "../wlan_aps.csv"))
wlan_fingerprints_train = pd.read_csv(os.path.join(dir_path, "../train/raw/wlan_fingerprints.csv"))
ble_fingerprints_train = pd.read_csv(os.path.join(dir_path, "../train/raw/ble_fingerprints.csv"))
wlan_fingerprints_test = pd.read_csv(os.path.join(dir_path, "../test/raw/wlan_fingerprints.csv"))
ble_fingerprints_test = pd.read_csv(os.path.join(dir_path, "../test/raw/ble_fingerprints.csv"))

print("Total detected WLAN access points: ", len(wlan_aps))
print("Total stable WLAN access points: ", len(wlan_aps[wlan_aps['STABLE'] == True]))
print("Training WLAN data", len(wlan_fingerprints_train), f"(~{len(wlan_fingerprints_train) / 440} per fingerprint)")
print("Training BLE advertisements", len(ble_fingerprints_train), f"(~{len(ble_fingerprints_train) / 440} per fingerprint)")
print("Test WLAN data", len(wlan_fingerprints_test), f"(~{len(wlan_fingerprints_test) / 120} per fingerprint)")
print("Test BLE advertisements", len(ble_fingerprints_test), f"(~{len(ble_fingerprints_test) / 120} per fingerprint)")
