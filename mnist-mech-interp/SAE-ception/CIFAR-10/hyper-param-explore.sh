#!/bin/bash

python3 adapt_classifier.py --base_lr 0.00001 --base_decay 0.05
python3 adapt_classifier.py --base_lr 0.00005 --base_decay 0.05
python3 adapt_classifier.py --base_lr 0.0001 --base_decay 0.05
python3 adapt_classifier.py --base_lr 0.0005 --base_decay 0.05
python3 adapt_classifier.py --base_lr 0.001 --base_decay 0.05

python3 adapt_classifier.py --base_lr 0.00001 --base_decay 0.1
python3 adapt_classifier.py --base_lr 0.00001 --base_decay 0.2
python3 adapt_classifier.py --base_lr 0.00001 --base_decay 0.5

python3 adapt_classifier.py --base_lr 0.0001 --base_decay 0.1
python3 adapt_classifier.py --base_lr 0.0001 --base_decay 0.2
python3 adapt_classifier.py --base_lr 0.0001 --base_decay 0.5

python3 adapt_classifier.py --base_lr 0.001 --base_decay 0.1
python3 adapt_classifier.py --base_lr 0.001 --base_decay 0.2
python3 adapt_classifier.py --base_lr 0.001 --base_decay 0.5