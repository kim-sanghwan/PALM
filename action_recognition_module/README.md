To generate the recognized actions, please follow below instructions

1. Generate and save video features at [EgoVLP](EgoVLP/).
   - Refer to official github page (https://github.com/showlab/EgoVLP/tree/main) to download pretrained model (EgoVLP_PT_BEST: egovlp.pth) and set environment
   - You should also download Ego4D video clips and annotations following official Ego4D github for long-term action anticipation https://github.com/EGO4D/forecasting/blob/main/LONG_TERM_ANTICIPATION.md and change the corresponding directory in this codebase.
   - Save train data features
   ```python run/test_lta.py --gpu 1 --config configs/eval/lta.json --save_feats /your/save/directory/ --split train```
