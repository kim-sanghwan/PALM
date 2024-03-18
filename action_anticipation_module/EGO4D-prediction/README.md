This repository is a direct adaptation from the Ego4D git page. Please consider referring their datatset.

## How to run code 

1. Install conda environment based on environment.yml

2. Download LLM of your interest and update get_generator(text_generation_model, device) function in utils/prompt_utils.py
We leveraged llama-2-7b. Please refer to official Llama2 gihub https://github.com/meta-llama/llama

3. We assume you already have the results of action recognition and image captioning models as json files. You should also have Ego4D video clips and annotations following below instruction

Run bash llama7B_run.sh  


# Long-Term Action Anticipation

This README reports information on how to train and test the baseline model for the Long-Term Action Anticipation task part of the forecasting benchmark of the Ego4D dataset. The following sections discuss how to download and prepare the data, download the pre-trained models and train and test the different components of the baseline.

## Data and models
Download all necessary data and model checkpoints using the [Ego4D CLI tool](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) and link the necessary files to the project directory.

```
# set download dir for Ego4D
export EGO4D_DIR=/path/to/Ego4D/

# download annotation jsons, clips and models for the FHO tasks
python -m ego4d.cli.cli \
    --output_directory=${EGO4D_DIR} \
    --datasets annotations clips lta_models \
    --benchmarks FHO

# link data to the current project directory
mkdir -p data/long_term_anticipation/annotations/ data/long_term_anticipation/clips_hq/
ln -s ${EGO4D_DIR}/v1/annotations/* data/long_term_anticipation/annotations/
ln -s ${EGO4D_DIR}/v1/clips/* data/long_term_anticipation/clips_hq/

# link model files to current project directory
mkdir -p pretrained_models
ln -s ${EGO4D_DIR}/v1/lta_models/* pretrained_models/

```

The `data/long_term_anticipation/annotations` directory should contain the following files

 ```
fho_lta_train.json
fho_lta_val.json
fho_lta_test_unannotated.json
fho_lta_taxonomy.json
```

Where `fho_lta_train.json`, `fho_lta_val.json` and `fho_lta_test_unannotated.json` contain the training, validation and test annotations, respectively, and `fho_lta_taxonomy.json.json` contains the verb/noun class id to text mapping.

### Downsampling video clips
To allow dataloaders to load clips efficiently, we will downsample video clips to 320p using ffmpeg. The script can be found at `tools/long_term_anticipation/resize_clips.sh` and can be run in parallel as a SLURM array job. Remember to adjust the paths and SLURM parameters before running.

```
sbatch tools/long_term_anticipation/resize_clips.sh
```
This will create and populate `data/long_term_anticipation/clips/` with downsampled clips

