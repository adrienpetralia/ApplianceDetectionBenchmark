<h1 align="center"> Appliance Detection Benchmark </h1>

<p align="center">
    <img width="400" src="https://github.com/adrienpetralia/ApplianceDetectionBenchmark/blob/main/ressources/Intro.png" alt="Intro image">
</p>

<h2 align="center">Appliance Detection Using Very Low-Frequency Smart Meter Time Series (ACM e-Energy '23) </h2>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/adrienpetralia/ApplianceDetectionBenchmark"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/adrienpetralia/ApplianceDetectionBenchmark">
</p>
</div>

### Abstract

In recent years, smart meters have been widely adopted by electricity suppliers to improve the management of the smart grid system. 
These meters usually collect energy consumption data at a very low frequency (every 30min), enabling utilities to bill customers more accurately. 
To provide more personalized recommendations, the next step is to detect the appliances owned by customers, which is a challenging problem, due to the very-low meter reading frequency.
Even though the appliance detection problem can be cast as a time series classification problem, with many such classifiers having been proposed in the literature, no study has applied and compared them on this specific problem.
This paper presents an in-depth evaluation and comparison of state-of-the-art time series classifiers applied to detecting the presence/absence of diverse appliances in very low-frequency smart meter data. 
We report results with five real datasets. 
We first study the impact of the detection quality of 13 different appliances using 30min sampled data, and we subsequently propose an analysis of the possible detection performance gain by using a higher meter reading frequency. 
The results indicate that the performance of current time series classifiers varies significantly. 
Some of them, namely deep learning-based classifiers, provide promising results in terms of accuracy (especially for certain appliances), even using 30min sampled data, and are scalable to the large smart meter time series collections of energy consumption data currently available to electricity suppliers.
Nevertheless, our study shows that more work is needed in this area to further improve the accuracy of the proposed solutions. 

### References
> Adrien Petralia, Philippe Charpentier, Paul Boniol, and Themis Palpanas. 2023. 
> Appliance Detection Using Very Low-Frequency Smart Meter Time Series. 
> In The 14th ACM International Conference on Future Energy Systems (e-Energy ’23), June 20–23, 2023, Orlando, FL, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3575813.3595198

```bibtex
@inproceedings{10.1145/3575813.3595198,
  author    = {Adrien Petralia and
               Philippe Charpentier and
               Paul Boniol and
               Themis Palpanas},
  title     = {Appliance Detection Using Very Low-Frequency Smart Meter Time Series},
  booktitle = {{ACM International Conference on Future Energy Systems (e-Energy)}},
  year      = {2023}
}
```

Access to the Preprint version of this article [here](https://www.researchgate.net/publication/370654257_Appliance_Detection_Using_Very_Low-Frequency_Smart_Meter_Time_Series).

### Contributors

- Adrien Petralia (EDF R&D - Université Paris Cité)
- Paul Boniol (Université Paris Cité - ENS Paris Saclay)

### Prerequisites 

Python version : <code> >= Python 3.7 </code>

Overall, the required python packages are listed as follows:

<ul>
    <li><a href="https://numpy.org/">numpy</a></li>
    <li><a href="https://pandas.pydata.org/">pandas</a></li>
    <li><a href="https://scikit-learn.org/stable/">sklearn</a></li>
    <li><a href="https://imbalanced-learn.org/stable/">imbalanced-learn</a></li>
    <li><a href="https://pytorch.org/docs/1.8.1/">pytorch==1.8.1</a></li>
    <li><a href="https://pypi.org/project/torchinfo/0.0.1/">torchinfo</a></li>
    <li><a href="https://scipy.org/">scipy</a></li>
    <li><a href="http://www.sktime.net/en/latest/">sktime</a></li>
    <li><a href="https://matplotlib.org/">matplotlib</a></li>
</ul>

### Installation

Use pip to install all the required libraries listed in the requirements.txt file.

<code> pip install -r requirements.txt </code>

### Data
The data used in this project comes multiple sources:

<ul>
  <li>CER smart meter dataset from the ISSDA archive.</li>
  <li>REFIT smart meter dataset.</li>
  <li>UKDALE smart meter dataset.</li>
  <li>Private smart meter dataset provide by EDF (Electricité De France).</li>
</ul> 

You may find more information on how to access the datasets in the [data](https://github.com/adrienpetralia/ApplianceDetectionBenchmark/tree/main/data) folder.

The following table summarzies some statistics of the abovementioned datasets:

| Datasets | number of TS | 1-min sampled  TS length | 10-min sampled  TS length | 15-min sampled  TS length | 30-min sampled  TS length |
|----------|--------------|--------------------------|---------------------------|---------------------------|---------------------------|
| REFIT    | 9091         | 1440                     | 144                       | 96                        | 48                        |
| UKDALE   | 4767         | 1440                     | 144                       | 96                        | 48                        |
| CER      | 4225         | /                        | /                         | /                         | 25728                     |
| EDF 1    | 2611         | /                        | /                         | /                         | 17520                     |
| EDF 2    | 1553         | /                        | 26208                     | 17427                     | 8736                      |


The following table summarizes the selected appliance detection cases through the five datasets; for each case, the table summarizes the number of time series available (♯TS)
and the imbalance degree of the test set for the case (IB Ratio). A slash indicate that no data are available for this case/dataset.

| Appliance case    | REFIT (#TS, IB ratio) | UKDALE (#TS, IB ratio) | CER (#TS, IB ratio) | EDF 1 (#TS, IB ratio) | EDF 2 (#TS, IB ratio) |
|-------------------|-----------------------|------------------------|---------------------|-----------------------|-----------------------|
| Desktop Computer  | 5190 (0.56)           | /                      | 3286 (0.47)         | 1402 (0.38)           | 3740 (0.62)           |
| Television        | 1134 (0.92)           | /                      | /                   | /                     | /                     |
| Cooker            | /                     | /                      | 1682 (0.76)         | /                     | /                     |
| Kettle            | 4790 (0.72)           | 1222 (0.84)            | /                   | /                     | /                     |
| Microwave         | 7434 (0.55)           | 1678 (0.77)            | /                   | 342 (0.91)            | /                     |
| Electric Oven     | /                     | /                      | /                   | 510 (0.85)            | 1152 (0.91)           |
| Dishwasher        | 7798 (0.44)           | 2378 (0.32)            | 2350 (0.66)         | 224 (0.93)            | 2846 (0.75)           |
| Tumble Dryer      | 3466 (0.22)           | /                      | 2214 (0.68)         | 1534 (0.41)           | 3470 (0.42)           |
| Washing Machine   | 7422 (0.54)           | 2380 (0.38)            | /                   | /                     | /                     |
| Water Heater      | /                     | /                      | 3070 (0.56)         | 1336 (0.66)           | 548 (0.86)            |
| Electric Heater   | /                     | /                      | 1348 (0.19)         | 1624 (0.58)           | 1538 (0.56)           |
| Convector         | /                     | /                      | /                   | 506 (0.69)            | /                     |
| Electric Vehicule | /                     | /                      | /                   | 140 (0.3)             | /                     |
