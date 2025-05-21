# ATV
Code repositories for __ATV (Adaptive Task Vectors).__

## Requirements

To run this code, installation of [RAPIDS](https://docs.rapids.ai/install/) is required.  
We recommend installing it via Conda as follows:

```bash
conda create -n ATV -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.04 python=3.10 'cuda-version>=12.0,<=12.8'
```

After completing the above installation, please install the required Python packages with the following command:

```bash
pip install -r requirements.txt

conda install -c conda-forge faiss-gpu

pip install git+https://github.com/davidbau/baukit
```


## Run code
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
Please make sure to modify the `result_dirs` variable in `ATV_analysis.py` to match the path to your result directory.

#### Analyze unseen task
   ```bash
   python ATV_analysis.py
   ```
For unseen data, run `ATV_unseen.py` to perform the evaluation.
As above, make sure to set the correct paths accordingly.

<br/>

## Acknowledge
This repository is built on top of the [ELICIT](https://github.com/LINs-lab/ELICIT) project. We thank the authors for sharing the source and their work itself.