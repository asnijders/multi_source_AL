# Active Learning

* **Note**: We intend to provide a more comprehensively documented repository soon at https://github.com/asnijders/multi_source_AL.

This repository contains the necessary code to run the experiments as presented in the paper "Investigating Multi-source Active Learning for Natural Language Inference".

For an overview of required dependencies, see:
`resources/environment/requirements.txt`

The used datasets can be downloaded by running `scripts/download.sh`. 
Note that at time of writing, *WANLI* was not yet pubicly available, so this script will only download data for *SNLI*, *MNLI* and *ANLI*.

The experiments for this work were performed using the LISA GPU cluster. Therefore, the code has been structured to accomodate a batch-processing workflow.
Example jobs for single- and multi-source active learning experiments as presented in this work can be found under`jobs/active_array_single.job` and `jobs/active_array.job`, respectively. 

For an overview of used script arguments, please see the dedicated argparse section in `main.py`.

Code to produce datamaps and assorted experiments can be found under `data_analysis/analysis.ipynb`.
