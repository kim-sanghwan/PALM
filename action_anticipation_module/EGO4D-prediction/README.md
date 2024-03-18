## How to run code 

1. Install conda environment based on environment.yml

2. Download LLM of your interest and update get_generator(text_generation_model, device) function in utils/prompt_utils.py
We leveraged llama-2-7b. Please refer to official Llama2 gihub https://github.com/meta-llama/llama to download the model weight.

3. We assume you already have the results of action recognition and image captioning models as json files. You should also have Ego4D video clips and annotations. Change corresponding directories in the codebase and simply run ```bash llama7B_run.sh ```.



