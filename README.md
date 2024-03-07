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

## Abstract

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

## References
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

## Contributors

- Adrien Petralia (EDF R&D - Université Paris Cité)
- Paul Boniol (Université Paris Cité - ENS Paris Saclay)

## Prerequisites 

Python version : <code> >= Python 3.7 </code>

Overall, the required python packages are listed as follows:

<ul>
    <li><a href="https://numpy.org/">numpy</a></li>
    <li><a href="https://pandas.pydata.org/">pandas</a></li>
    <li><a href="https://scikit-learn.org/stable/">scikit-learn</a></li>
    <li><a href="https://imbalanced-learn.org/stable/">imbalanced-learn</a></li>
    <li><a href="https://pytorch.org/docs/1.13.1/">torch==1.13.1</a></li>
    <li><a href="https://pypi.org/project/torchinfo/0.0.1/">torchinfo</a></li>
    <li><a href="https://scipy.org/">scipy</a></li>
    <li><a href="http://www.sktime.net/en/latest/">sktime</a></li>
    <li><a href="https://matplotlib.org/">matplotlib</a></li>
</ul>

## Installation

Use pip to install all the required libraries listed in the requirements.txt file.

``` 
pip install -r requirements.txt 
```

## Data
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


The following table summarizes the selected appliance detection cases through the five datasets; for each case, the table summarizes the number of time series available (♯TS) and the imbalance degree of the test set for the case (IB Ratio). 
A slash indicate that no data are available for this case/dataset.

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


## Results

In the following table, we summarize our benchmark evaluation for each appliance detection case. The classification methods used in our benchmark are listed in the following taxonomy (only the methods in blue were experimentally evaluated):

<p align="center">
    <img width="600" src="https://github.com/adrienpetralia/ApplianceDetectionBenchmark/blob/main/ressources/taxonomy.png" alt="Taxonomy of classification methods">
</p>

### 30min accuracy detection results

#### Desktop Computer Detection Accuracy (F1-Macro score) 

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| CER      | 0.618   | 0.617      | 0.606  | 0.602   | 0.614  | 0.530     | 0.608         | 0.516 | 0.580 | 0.586 | 0.491    | 0.579     |
| EDF 1    | 0.571   | 0.564      | 0.570  | 0.489   | 0.560  | 0.459     | 0.555         | 0.491 | 0.533 | 0.543 | 0.469    | 0.528     |
| EDF 2    | 0.603   | 0.576      | 0.582  | 0.579   | 0.620  | 0.514     | 0.601         | 0.519 | 0.570 | 0.592 | 0.520    | 0.571     |
| REFIT    | 0.697   | 0.683      | 0.674  | 0.715   | 0.740  | /         | 0.623         | 0.542 | 0.525 | 0.600 | 0.548    | 0.635     |

#### Television Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| REFIT    | 0.656   | 0.647      | 0.645  | 0.695   | 0.699  | /         | 0.718         | 0.485 | 0.737 | 0.664 | 0.513    | 0.646     |

#### Cooker Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| CER      | 0.680   | 0.673      | 0.676  | 0.661   | 0.689  | 0.541     | 0.710         | 0.526 | 0.566 | 0.584 | 0.440    | 0.613     |

#### Kettle Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| REFIT    | 0.368   | 0.376      | 0.381  | 0.522   | 0.477  | /         | 0.415         | 0.536 | 0.359 | 0.428 | 0.421    | 0.428     |
| UKDALE   | 0.540   | 0.502      | 0.522  | 0.428   | 0.432  | /         | 0.583         | 0.504 | 0.353 | 0.442 | 0.446    | 0.475     |

#### Microwave Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| REFIT    | 0.656   | 0.598      | 0.588  | 0.745   | 0.679  | /         | 0.673         | 0.563 | 0.540 | 0.717 | 0.529    | 0.629     |
| UKDALE   | 0.446   | 0.498      | 0.460  | 0.532   | 0.526  | /         | 0.541         | 0.435 | 0.459 | 0.430 | 0.378    | 0.471     |
| EDF 1    | 0.480   | 0.471      | 0.475  | 0.534   | 0.510  | 0.409     | 0.474         | 0.454 | 0.400 | 0.429 | 0.457    | 0.463     |

#### Electric Oven Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| EDF 1    | 0.513   | 0.498      | 0.499  | 0.512   | 0.512  | 0.472     | 0.523         | 0.506 | 0.429 | 0.497 | 0.437    | 0.491     |
| EDF 2    | 0.557   | 0.584      | 0.553  | 0.571   | 0.562  | 0.560     | 0.576         | 0.495 | 0.459 | 0.491 | 0.397    | 0.528     |

#### Dishwasher Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| REFIT    | 0.650   | 0.599      | 0.619  | 0.580   | 0.605  | /         | 0.590         | 0.557 | 0.519 | 0.584 | 0.515    | 0.582     |
| UKDALE   | 0.458   | 0.465      | 0.465  | 0.419   | 0.380  | /         | 0.384         | 0.399 | 0.429 | 0.554 | 0.525    | 0.448     |
| CER      | 0.699   | 0.720      | 0.700  | 0.730   | 0.728  | 0.594     | 0.737         | 0.586 | 0.609 | 0.648 | 0.488    | 0.658     |
| EDF 1    | 0.454   | 0.441      | 0.450  | 0.528   | 0.522  | 0.383     | 0.535         | 0.430 | 0.418 | 0.421 | 0.211    | 0.436     |
| EDF 2    | 0.753   | 0.760      | 0.741  | 0.799   | 0.801  | 0.585     | 0.835         | 0.596 | 0.603 | 0.600 | 0.512    | 0.690     |

#### Tumble Dryer Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| REFIT    | 0.493   | 0.503      | 0.502  | 0.468   | 0.448  | /         | 0.441         | 0.506 | 0.416 | 0.434 | 0.461    | 0.467     |
| CER      | 0.634   | 0.641      | 0.628  | 0.606   | 0.612  | 0.550     | 0.623         | 0.549 | 0.578 | 0.602 | 0.474    | 0.591     |
| EDF 1    | 0.619   | 0.578      | 0.607  | 0.624   | 0.607  | 0.475     | 0.636         | 0.550 | 0.537 | 0.563 | 0.487    | 0.571     |
| EDF 2    | 0.733   | 0.714      | 0.714  | 0.757   | 0.769  | 0.475     | 0.769         | 0.560 | 0.593 | 0.681 | 0.493    | 0.660     |

#### Waching Machine Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| REFIT    | 0.605   | 0.572      | 0.592  | 0.581   | 0.586  | /         | 0.614         | 0.520 | 0.562 | 0.557 | 0.529    | 0.572     |
| UKDALE   | 0.475   | 0.505      | 0.478  | 0.535   | 0.530  | /         | 0.454         | 0.408 | 0.581 | 0.549 | 0.509    | 0.502     |

#### Water Heater Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| CER      | 0.625   | 0.613      | 0.613  | 0.610   | 0.612  | 0.465     | 0.637         | 0.527 | 0.596 | 0.584 | 0.462    | 0.577     |
| EDF 1    | 0.835   | 0.821      | 0.827  | 0.814   | 0.828  | 0.768     | 0.841         | 0.670 | 0.713 | 0.805 | 0.591    | 0.774     |
| EDF 2    | 0.733   | 0.685      | 0.724  | 0.731   | 0.685  | 0.591     | 0.759         | 0.658 | 0.580 | 0.666 | 0.617    | 0.675     |

#### Electric Heater Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| CER      | 0.522   | 0.532      | 0.514  | 0.533   | 0.508  | 0.477     | 0.565         | 0.459 | 0.492 | 0.527 | 0.397    | 0.502     |
| EDF 1    | 0.784   | 0.783      | 0.789  | 0.777   | 0.778  | 0.713     | 0.800         | 0.643 | 0.758 | 0.777 | 0.638    | 0.749     |
| EDF 2    | 0.591   | 0.566      | 0.578  | 0.626   | 0.637  | 0.527     | 0.648         | 0.497 | 0.591 | 0.605 | 0.451    | 0.574     |

#### Convector/Heat Pump Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| EDF 1    | 0.632   | 0.622      | 0.631  | 0.597   | 0.638  | 0.534     | 0.651         | 0.539 | 0.556 | 0.625 | 0.467    | 0.590     |

#### Electric Vehicule Detection Accuracy (F1-Macro score)

| Datasets | Arsenal | MiniRocket | Rocket | ConvNet | ResNet | ResNetAtt | InceptionTime | BOSS  | TSF   | RISE  | KNNeucli | Avg score |
|----------|---------|------------|--------|---------|--------|-----------|---------------|-------|-------|-------|----------|-----------|
| EDF 1    | 0.689   | 0.730      | 0.670  | 0.681   | 0.699  | 0.553     | 0.720         | 0.541 | 0.456 | 0.725 | 0.556    | 0.638     |
