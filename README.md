# Dementia_Screening


An LLM-based cognitive impairment screening framework that elevates LLMs from feature extractors to intelligent decision makers. It employs a two-level ML-based indicator generation strategy with dual-pathway (acoustic + semantic) analysis and multi-path fusion to produce both predictions and clinically interpretable explanations from speech recordings.

## How to Run?

Please go through the entire procedure step by step. You will run the code from folder 01** to 05**. We keep an individual main function in each folder, to allow readers to check the output of each phase.

The acoustic feature extraction can be found in folder 04**.  The linguistic feature extraction can be found in folder 05**. Then, you will run "semantic_metric_extractor.py" to measure the linguistic features.

The final prediction can be made by the "main_all.py" file in the 05** folder. 

-----------------------

We used the DementiaBank dataset. According to the policy, we cannot republish the dataset. Please refer to https://talkbank.org/dementia/ to access the original dataset. This project only provides our prototype of the proposed framework.

