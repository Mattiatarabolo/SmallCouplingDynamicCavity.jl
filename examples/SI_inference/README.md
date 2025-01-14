# Adapted Dataset for real contact networks

This repository contains an adapted version of the dataset published as:  
> Génois, M., & Barrat, A. (2017). SocioPatterns datasets [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2540795

## Modifications

The original contact data are collected over a period of several days, with a temporal resolution of 20 seconds. This allows for data aggregation over coarse-grained time windows of a preferred size $\tau_w$. In this study, time windows with sizes $\tau_w$ ranging from 3 hours to a day are considered, resulting in $T$ time steps ranging from a minimum of 12 to a maximum of 36 steps. 

During the coarse-graining procedure:
- The number $c_{ij}^t$ of contacts between individuals $i$ and $j$ occurring in a time window $t$ of size $\tau_w$ is computed.
- This value is used to estimate the infection probability $\lambda_{ij}^t$ between the two individuals at time step $t$ as:
  \[
  \lambda_{ij}^t = 1 - (1 - \gamma)^{c_{ij}^t}
  \]
  where $\gamma$ is a common parameter describing the infectiousness of a single contact.

These modifications are tailored for studying the dynamics of infection spread in coarse-grained temporal networks.

## License

This adaptation is published under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
