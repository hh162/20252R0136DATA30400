# 20252R0136DATA30400
DATA304 Final Project

Description and instructions to reproduce results:
1. I used Visual Studio Code for this project.
2. Folder Structure:
final_project/
│
├── Amazon_products/
│   ├── classes.txt
│   ├── class_hierarchy.txt
│   ├── class_related_keywords.txt
│   │
│   ├── train/
│   │   └── train_corpus.txt
│   │
│   └── test/
│       └── test_corpus.txt
│
├── utils_taxonomy.py
├── generate_silver_labels.py
├── train_model.py
├── predict.py
└── run_all.sh

3. Code Run Order (run_all.sh):
   generate_silver_labels.py → train_model.py → predict.py
