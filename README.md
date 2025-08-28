# CaloNN
This repository contains a simple NN that is used to differentiate events where a gamma particle and a positron enter the PIONEER calorimeter. The parameter that is being evaluated is the proportion of hits that each SiPM gets in an event.

## 1d_data
This directory contains work done with 1d data. What this means is that the data being used by the NN is an array containing the proportion of hits that each SiPM got in an event. The array is ordered by SiPM ID.

## 2d_data
The data being used here is the same as before, but the presentation is a bit different. In this case, we want to project the geometry of the SiPMs into a 2d grid to feed to the NN. This approach has more potential than the previous one, as projecting the SiPMs into a 2D grid can give the NN more information about the actual hit patterns that can be observed with each event.