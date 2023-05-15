#!/bin/bash
declare -a all_case=("cooker_case" "dishwasher_case" "desktopcomputer_case" "ecs_case" "heater_case" "laptopcomputer_case" "tumbledryer_case" "tv_greater21inch_case" "tv_less21inch_case")
declare -a classifiers=("Arsenal" "Rocket" "Minirocket" "TimeSeriesForest" "Rise" "DrCIF" "cBOSS" "BOSS" "KNNeucli" "KNNdtw" "ResNet" "Inception" "ConvNet" "ConvResNetAttention")

for case in ${all_case[@]}; do
  for str in ${sk_classifiers[@]}; do
    sbatch CER_Benchmark.py $str $case
  done
done