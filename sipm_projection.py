import ROOT
import pandas as pd

"""
Project SiPM locations and such.
"""

if __name__ == "__main__":
    geometry_file = "~/miguel_sims/LXe_Calo_sensor_positions_rln_15_opening_angle_35.csv"

    pd.read_csv(geometry_file, comment = '#')