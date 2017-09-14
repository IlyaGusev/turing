#!/bin/bash

cd code
python3 modelA.py
python3 modelB.py load
python3 modelC.py
python3 avg.py