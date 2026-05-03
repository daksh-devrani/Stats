#!bin/bash

source .venv/bin/activate

python data_cleaning.py

python phase_1_split.py

python phase_2_train.py

python phase_3_evaluate.py

deactivate

