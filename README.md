# CaloNN
This repository contains a simple NN that is used to differentiate events where a gamma particle and a positron enter the PIONEER calorimeter. The parameter that is being evaluated is the proportion of hits that each SiPM gets in an event. The logic for that is that positrons initiate their shower almost as soon as they enter the calorimeter, whereas gammas initiate this process deeper into the calorimeter. This causes different hit patterns in the SiPMs. We hope to use these patterns to tell these events apart.

## 1d_data
This directory contains work done with 1d data. What this means is that the data being used by the NN is an array containing the proportion of hits that each SiPM got in an event. The array is ordered by SiPM ID. As an example, consider a simple case where we have 3 SiPMs with IDs: $\{1,2,3\}$. The data being fed is an array of the form:

$$\biggl[\frac{\text{hits in 1}}{\text{total hits}}, \frac{\text{hits in 2}}{\text{total hits}}, \frac{\text{hits in 3}}{\text{total hits}} \biggr]$$

## 2d_data
The data being used here is the same as before, but the presentation is a bit different. In this case, we want to project the geometry of the SiPMs into a 2d grid to feed to the NN. This approach has more potential than the previous one, as projecting the SiPMs into a 2D grid can give the NN more information about the actual hit patterns that can be observed with each event.

## sipm_info
This folder contains useful mappings to easily access information from the sipms. These mappings include mapping the SiPM ID to an index in an array (and viceversa), as well as a mapping from SiPM ID to geometry.