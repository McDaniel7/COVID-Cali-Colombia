# Non-stationary spatio-temporal point process modeling for high-resolution COVID-19 data

Most COVID-19 studies commonly report figures of the overall infection at a state- or county-level. This aggregation tends to miss out on fine details of virus propagation. In this paper, we analyze a high-resolution COVID-19 dataset in Cali, Colombia, that records the precise time and location of every confirmed case. We develop a non-stationary spatio-temporal point process equipped with a neural network-based kernel to capture the heterogeneous correlations among COVID-19 cases. The kernel is carefully crafted to enhance expressiveness while maintaining model interpretability. We also incorporate some exogenous influences imposed by city landmarks. Our approach outperforms the state-of-the-art in forecasting new COVID-19 cases with the capability to offer vital insights into the spatio-temporal interaction between individuals concerning the disease spread in a metropolis.


## Model

### Neural network architecture


### Non-stationary neural kernel



## Results

Animations below show (1) The evolvement of beat workload distribution from 2014 to 2019 where year 2018 and 2019 are predicted by our model; (2,3) Comparisons of zone workload distribution between existing plan and redesigned plan at 2018 and 2019

Beat distribution over years     | Design comparison for 2018    | Design comparison for 2019
:----------------------------:|:----------------------------:|:----------------------------:
![]()  |  ![]() | ![]()



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
