#!/bin/bash
python generate_silver_labels.py
python train_model.py
python predict.py
