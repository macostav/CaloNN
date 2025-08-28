import pandas as pd
import pickle

"""
Generates a dictionary from SiPM ID to its information for easier access.
"""

if __name__ == "__main__":
    geometry_path = "~/miguel_sims/LXe_Calo_sensor_positions_rIn_15_opening_angle_35.csv"

    # Read CSV
    geom = pd.read_csv(
        geometry_path,
        comment="#",
        sep=",",
        names=["volumeID", "type", "surface", "x", "y", "z", "r", "theta", "phi"]
    )

    # Create mapping: sipm_id -> {info in dictionary}
    sipm_map = geom.set_index("volumeID").to_dict(orient="index")

    # Save mapping as pickle
    with open("sipm_info_mapping.pkl", "wb") as f:
        pickle.dump(sipm_map, f)
    