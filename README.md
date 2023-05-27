# Non-stationary spatio-temporal point process modeling for high-resolution COVID-19 data

Most COVID-19 studies commonly report figures of the overall infection at a state- or county-level. This aggregation tends to miss out on fine details of virus propagation. In this paper, we analyze a high-resolution COVID-19 dataset in Cali, Colombia, that records the precise time and location of every confirmed case. We develop a non-stationary spatio-temporal point process equipped with a neural network-based kernel to capture the heterogeneous correlations among COVID-19 cases. The kernel is carefully crafted to enhance expressiveness while maintaining model interpretability. We also incorporate some exogenous influences imposed by city landmarks. Our approach outperforms the state-of-the-art in forecasting new COVID-19 cases with the capability to offer vital insights into the spatio-temporal interaction between individuals concerning the disease spread in a metropolis.


## Model

### Neural network architecture

An illustration of a deep neural network that maps an arbitrary location $s$ to a spatial kernel, consisting of a feature function $\kappa_s$ (represented through focus points) and weight $w_s$.

![](https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/NN_Illustration.png)

### Non-stationary neural kernel

An example of the non-stationary spatial kernel with two feature functions evaluating at location $s$ (the center of the box), i.e., $\upsilon(s, s') = \left< \phi_s, \phi_{s'}\right>,~\forall s' \in \mathcal{S}$, where $\phi_s = \kappa_s^{(1)} + \kappa_s^{(2)}$. 
Two purple boxes in the middle indicate the cross-correlated terms ($\kappa_s^{(1)} \cdot \kappa_{s'}^{(2)}$ and $\kappa_s^{(2)} \cdot \kappa_{s'}^{(1)}$); the red and blue boxes indicate the self-correlated terms ($\kappa_s^{(1)} \cdot \kappa_{s'}^{(1)}$ and $\kappa_s^{(2)} \cdot \kappa_{s'}^{(2)}$).

![](https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/Spatial_Kernel_Illustration.png)


## Results

Animations below show (1) Snapshot of confirmed COVID-19 cases at the week of July 12, 2020. Each dot represents the location of a confirmed case.  (2) Evaluation of the spatial kernel $\upsilon(s, \cdot)$ with $s$ fixed at city airport, which intuitively show the spatial influence of the COVID-19 cases reported at airport. (3) Predicted conditional intensity at June 28.

<table>
  <tr>
    <th> Data </th>
    <th> Kernel </th>
    <th> Prediction </th>
  </tr>
  <tr>
    <td> <img src="https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/DP_18.png"  alt="1" width = 360px height = 360px ></td>
    <td> <img src="https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/Spatial_correlation_exo_1.png"  alt="2" width = 360px height = 360px ></td>
    <td> <img src="https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/Intensity_16.png"  alt="3" width = 360px height = 360px ></td>
  </tr>
</table>

<!-- Data     | Kernel    | Prediction
:---------------------------:|:---------------------------:|:----------------------------:
![](https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/Spatial_correlation_exo_1.png) |  ![](https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/Spatial_correlation_exo_1.png) | ![](https://github.com/McDaniel7/COVID-Cali-Colombia/blob/main/Results/Spatial_correlation_exo_1.png) -->



## Scripts

`kernel.py` defines the non-stationary neural-network-based spatio-temporal kernel.

`exogenous_promotion.py` defines the exogenous promotion of city landmarks to the spread of COVID-19.

`point_process.py` defines the spatio-temporal point process model with non-stationary neural kernel.

`train.py` is the main function to run the experiments, including data loading and model training.


## Citation
```
@article{dong2021non,
    author = {Dong, Zheng and Zhu, Shixiang and Xie, Yao and Mateu, Jorge and Rodríguez-Cortés, Francisco J},
    title = "{Non-stationary spatio-temporal point process modeling for high-resolution COVID-19 data}",
    journal = {Journal of the Royal Statistical Society Series C: Applied Statistics},
    year = {2023},
    month = {03},
    issn = {0035-9254},
    doi = {10.1093/jrsssc/qlad013},
    url = {https://doi.org/10.1093/jrsssc/qlad013},
    note = {qlad013},
    eprint = {https://academic.oup.com/jrsssc/advance-article-pdf/doi/10.1093/jrsssc/qlad013/49622936/qlad013\_supplementary\_data.pdf},
}
```
