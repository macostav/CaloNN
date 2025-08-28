import ROOT
import pickle
import numpy as np

"""
We extract the data from the PIONEER simulation ROOT files. We save the data as a numpy array.
"""

if __name__ == "__main__":
    # Loading SiPM ID mapping
    with open('sipm_id_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    sipm_id_to_index = mapping['sipm_id_to_index']
    n_unique_sipms = mapping['n_unique_sipms'] # Should be 1891
    
    # ROOT data extraction
    folder = "/data_macosta/CaloSims"
    particle_type = "positron" # gamma or positron

    # Build TChain with all trees in simulation
    chain = ROOT.TChain("sim")
    for i in range(32):
        chain.Add(f"{folder}/{particle_type}_-{i:02d}.root")

    n_events = chain.GetEntries()
    print(f"Processing {n_events} events...")

    event_data = []
    # Iterating over all events
    for i_event in range(n_events):
        chain.GetEntry(i_event)

        sipm = chain.sipm
        nsipms = sipm.GetEntriesFast() # Number of SiPMs with hits in this event

        sipm_features = np.zeros(n_unique_sipms) # Array of zeros for all SiPMs; this is the data we want to save for this event

        # Collect the hit data for this event
        event_hit_data = {}
        total_hits_in_event = 0
        
        for i in range(nsipms):
            sensor = sipm.At(i)
            sipm_id = sensor.GetID()
            
            nhits = sensor.GetNHits() # Get number of hits for this SiPM
            
            event_hit_data[sipm_id] = nhits # Store the number of hits for this SiPM
            total_hits_in_event += nhits

        # Normalize the hit counts by total hits
        if total_hits_in_event > 0:
            for sipm_id, nhits in event_hit_data.items():
                if sipm_id in event_hit_data: # If this SiPM had hits
                    array_index = sipm_id_to_index[sipm_id]
                    sipm_features[array_index] = nhits / total_hits_in_event # Normalized hits

        event_data.append(sipm_features.copy())

        if i_event % 5000 == 0:
            print(f"Event {i_event}.")
    
    # Saving
    X = np.array(event_data) # One array for all 50,000 events. Will have to separate later into multiple sets
    np.save(f"{particle_type}_cnn_data.npy", X)
