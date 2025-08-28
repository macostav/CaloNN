import ROOT
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
We extract the data from the PIONEER simulation ROOT files. We project the different surfaces into a 2D grid to create hit maps
for each surface. Ultimately, we aim to put all the hitmaps into a single image that we will then feed into the CNN.
"""

if __name__ == "__main__":
    # Loading SiPM info
    with open('sipm_info/sipm_id_index.pkl', 'rb') as f:
        index_map = pickle.load(f)

    sipm_id_to_index = index_map['sipm_id_to_index']
    n_unique_sipms = index_map['n_unique_sipms'] # Should be 1891

    with open('sipm_info/sipm_info_mapping.pkl', 'rb') as f:
        sipm_info = pickle.load(f)
    
    # ROOT data extraction
    folder = "/data_macosta/CaloSims"
    particle_type = "gamma" # gamma or positron

    # Build TChain with all trees in simulation
    chain = ROOT.TChain("sim")
    for i in range(32):
        chain.Add(f"{folder}/{particle_type}_-{i:02d}.root")

    # Hitmap parameters
    # TODO I think what Ben has in poly_clustering.C is probably a better implementation of this binning logic
    n_theta_bins = 30
    n_phi_bins = 60
    n_surfaces = 4
    event_data = []

    # Precompute SiPM bin indices
    sipm_bins = {}
    for sipm_id, info in sipm_info.items():
        theta = info["theta"]
        phi = info["phi"]
        
        theta_idx = int(theta / np.pi * n_theta_bins)
        phi_idx = int(phi / (2 * np.pi) * n_phi_bins)
        
        # Clip to valid range
        theta_idx = min(max(theta_idx, 0), n_theta_bins - 1)
        phi_idx = min(max(phi_idx, 0), n_phi_bins - 1)
        
        sipm_bins[sipm_id] = (theta_idx, phi_idx, info["surface"])

    # Process events
    n_events = chain.GetEntries()
    print(f"Processing {n_events} events...")

    for i_event in range(n_events):
        chain.GetEntry(i_event)
        sipm = chain.sipm
        nsipms = sipm.GetEntriesFast()
        
        # 4 surfaces, each n_theta x n_phi
        hitmaps = [np.zeros((n_theta_bins, n_phi_bins), dtype=float) for _ in range(n_surfaces)]
        total_hits_in_event = 0
        
        # First pass: collect raw hit counts
        sipm_hits = {}
        for i in range(nsipms):
            sensor = sipm.At(i)
            sipm_id = sensor.GetID()
            nhits = sensor.GetNHits()
            sipm_hits[sipm_id] = nhits
            total_hits_in_event += nhits
        
        if total_hits_in_event > 0:
            # Normalize and assign to the correct 2D bin
            for sipm_id, nhits in sipm_hits.items():
                if sipm_id not in sipm_bins:
                    continue
                theta_idx, phi_idx, surface = sipm_bins[sipm_id]
                hitmaps[int(surface)][theta_idx, phi_idx] += nhits / total_hits_in_event

        event_data.append(np.stack(hitmaps))
        
        # Saving
        event_array = np.array(event_data, dtype=np.float32)  # should have shape: (n_events, 4, n_theta_bins, n_phi_bins)
        np.save("gamma_data.npy", event_array)
