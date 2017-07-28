#!/bin/bash

python3 simple_predict.py --days 4
python3 model.py load 4
python3 model_mryab_4d.py
python3 avg.py