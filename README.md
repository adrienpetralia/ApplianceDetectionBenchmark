<h1 align="center"> Appliance Detection Benchmark </h1>

<p align="center">
    <img width="450" src="https://github.com/adrienpetralia/ApplianceDetectionBenchmark/blob/master/ressources/Intro.png" alt="Intro image">
</p>

<h2 align="center">Appliance Detection Using Very Low-Frequency Smart Meter Time Series (ACM e-Energy '23) </h2>

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

To download the preprocessed subsample of the CER dataset (data/labels) :

<code>cd data</code>

Copy and paste the data from the link drive folder indicate in the README.md file.
