# NNdisulfide
Neural network to Predict Artificial Disulfide Bonds
Given a PDF or CIF structure, predict which 2 residues could be mutated to create an artificial disulfide bonds

**Installation:**

    conda create -n nndisulfide python=3.11
    conda activate nndisulfide
    pip install gemmi torch pandas numpy tqdm scikit-learn

**Build**

      python NNdisulfide.py build \
      --data_dir /data/pdb-mmCIF \
      --out_csv disulfides.csv \
      --nproc 32        # adapt to your CPU budget

**Train**

      python NNdisulfide.py train \
      --dataset disulfides.csv \
      --model_file ss_model.pt \
      --epochs 30

**Predict**

      python NNdisulfide.py predict \
      --model_file ss_model.pt \
      --structure my_enzyme.pdb \
      --top_k 25 \
      --out my_enzyme_ss_predictions.csv
