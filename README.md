# Overview

This repository computes False Acceptance Rate (FAR), False Rejection Rate (FRR) and Equal Error Rate (EER) given similarity
scores of a biometric system and the actual label of the sample (genuine or imposter). It also plots some visualizations
such as histogram of genuine (blue) and imposter (red) distributions with a given bin width, EER along with FAR and FRR,
and ROC curve.


## Installation

1. Git clone this repo: `git clone https://github.com/nathanlem1/Compute_EER.git`
2. Install dependencies by `pip install -r requirements.txt` to have the same environment configuration as the one we used. 

## Run
```bash
$ python compute_eer.py
```

