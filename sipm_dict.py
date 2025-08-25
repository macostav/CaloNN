import ROOT
import pickle
import json
import numpy as np

"""
Generates a dictionary of SiPM IDs and mapping between SiPM ID and array index. This will be
useful to when we are prepating the data for the NN.
"""

if __name__ == "__main__":
    folder = "/data_macosta/CaloSims"
    
    # Build TChain with all trees in simulation
    chain = ROOT.TChain("sim")
    for i in range(32):
        chain.Add(f"{folder}/gamma_-{i:02d}.root")

    n_events = chain.GetEntries()
    print(f"Processing {n_events} events...")

    # Collect ALL unique SiPM IDs across all events
    all_sipm_ids = set()

    for i_event in range(n_events):
        chain.GetEntry(i_event)
        
        sipm = chain.sipm
        nsipms = sipm.GetEntriesFast()
        
        # Collect all IDs that appear in this event
        for i in range(nsipms):
            sensor = sipm.At(i)
            id = sensor.GetID()
            all_sipm_ids.add(id)
        
        if i_event % 5000 == 0:
            print(f"Event {i_event}.")

    # Convert to sorted list for consistent ordering
    all_sipm_ids = sorted(list(all_sipm_ids))
    n_unique_sipms = len(all_sipm_ids)

    print(f"\nFound {n_unique_sipms} unique SiPM IDs")

    # Create mapping from SiPM ID to array index
    sipm_id_to_index = {sipm_id: idx for idx, sipm_id in enumerate(all_sipm_ids)}
    index_to_sipm_id = {idx: sipm_id for sipm_id, idx in sipm_id_to_index.items()}

    # Save the mappings
    # Save as pickle (Python-specific, preserves exact data types)
    with open('sipm_id_mapping.pkl', 'wb') as f:
        pickle.dump({
            'sipm_id_to_index': sipm_id_to_index,
            'index_to_sipm_id': index_to_sipm_id,
            'all_sipm_ids': all_sipm_ids,
            'n_unique_sipms': n_unique_sipms
        }, f)

    # Save as JSON (human-readable)
    with open('sipm_id_mapping.json', 'w') as f:
        json.dump({
            'sipm_id_to_index': {str(k): v for k, v in sipm_id_to_index.items()},
            'index_to_simp_id': {str(k): v for k, v in index_to_sipm_id.items()},
            'all_simp_ids': all_sipm_ids,
            'n_unique_sipms': n_unique_sipms
        }, f, indent=2)

    # Save just the list of IDs as numpy array
    np.save('all_simp_ids.npy', np.array(all_sipm_ids))

    print(f"\nSaved mappings:")
    print(f"  - simp_id_mapping.pkl")
    print(f"  - sipm_id_mapping.json") 
    print(f"  - all_sipm_ids.npy")
    print(f"\nDictionary generation complete!")