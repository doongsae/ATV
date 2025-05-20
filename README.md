# ATV
Code repositories for __ATV (Adaptive Task Vectors).__

### Prepare datasets
   ```bash
   ./scripts/prepare_dataset.sh
   ```

### Train ATV model
   ```bash
   ./scripts/ATV_training.sh
   ```
Running the above script trains the model on all 20 in-domain datasets. After training, evaluation is performed on the test samples from all in-domain datasets.


### Evaluate all datasets
   ```bash
   ./scripts/ATV_evaluate.sh
   ```
Running the above script enables evaluation of performance on each individual dataset within the full collection.

### Analyze results
   ```bash
   python ATV_analysis.py
   ```
This script enables evaluation of performance for each category.


<br/>

## Acknowledge
This repository is built on top of the [ELICIT](https://github.com/LINs-lab/ELICIT) project. We thank the authors for sharing the source and their work itself.