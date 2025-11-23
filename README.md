# Engineering Thesis Classifiers

This is our engineering thesis repository for the classifiers and analysis. Here we've collected the training code for the model architectures that we've tried, other experiments and model evaluation analysis.

Aside from the training and analysis, this project also contains the code for the demo application that we use to present our model. It uses Gradio API, that've enabled us to swiftly create the UI.

## Project structure

- `data` - The input directory with LIAR PLUS dataset taken from previous LIAR PLUS EDA stage.
    - `data/normalize_dataset.py` - The script for LIAR PLUS dataset normalization.
    - `data/normalized` - The normalized LIAR PLUS dataset, ready for training and classification.
- `notebooks` - The jupyter notebooks used for training the models.
    - `notebooks/utils.py` - The package for handling saving/loading the models' checkpoints and for sending the models to our SFTP server.
- `analysis` - The classification results and manual labeling analysis directory.
    - `analysis/base model analysis` - The analysis directory for the thesis's section 5 base model.
    - `analysis/input methods analysis` - The directory for the input representations analysis for the thesis's section 5.
        - `wyniki eksperymentu 1.ods` - The results of the textual input representations experiment for thesis's section 5.
    - `analysis/prepare analysis data` - The directory for evaluating final SM and SMA models and preparing the features for further analysis.
        - `analysis/prepare analysis data/Eval SM vs SMA.ipynb` - The jupyter notebook used for SM and SMA model evaluation on test dataset, and for gathering metrics like SM_pred, SMA_pred, SM_prob, SMA_prob, label_num, SM_highest_prob, SMA_highest_prob, and saving them to the file named `sm_vs_sma_dataset.csv`
        - `analysis/prepare analysis data/SM vs SMA comparison.ipynb` - The jupyter notebook used for further analisys of the results saved in `sm_vs_sma_dataset.csv` and for extraction of the 30 examples per each model for final analysis (described in section 7.3).
    - `analysis/final experiment analysis` - The directory for the final SM and SMA models analysis for the thesis's section 7.
        - `wyniki ostatecznego eksperymentu.ods` - The results of the models training stage from thesis's section 7.
        - `analysis/final experiment analysis/SM_correct_SMA_incorrect.ods` - The 30 examples from SM_correct_SMA_incorrect set, described in section 7.3.
        - `analysis/final experiment analysis/SM_incorrect_SMA_correct.ods` - The 30 examples from SM_incorrect_SMA_correct set, described in section 7.3.
    - `analysis/models eval` - The directory for evaluation analysis of the final models in thesis's section 7.
- `demo` - The demo program scripts directory.
    - `demo/demo_gemma.py` - The article generator script for the demo program.
    - `demo/demo_pipelines.py` - An additional metadata columns generator.
    - `demo/ demo_utils.py` - The utility package and final PyTorch model definition.
    - `demo/demo.py` - The main demo program file.
- `results` - The resulting models directory. It's in gitignore but it existed in the local instance of the repository and is mentioned in the code.
- `env.yml` - The conda environment export file.
