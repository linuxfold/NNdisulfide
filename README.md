# NNdisulfide
Simple Neural Network to Predict Artificial Disulfide Bonds
Given a PDB or CIF structure, predict which 2 residues could be mutated to create an artificial disulfide bond. 

You can install and predict by supplying the included ss_model.pt, which was trained on the same mmcif files that are used as part of the AF3 database (but includes newer files up to 5/24/25). 


**1 | What the script does**

**build**	

Parse every .mmCIF in a directory, pull out all annotated disulfide bonds (inter- & intra-chain) and generate matching negative examples.	Uses gemmi for blazing-fast CIF access and multiprocessing to scale across cores.

**train**	

Learn to score cysteine-pair compatibility.	A lightweight PyTorch feed-forward network (3 numeric features → 16→8→1). Default training loop + early model checkpointing.

**predict**	

Scan a new structure, enumerate residue pairs close enough to fuse, and rank them by probability they could form a disulfide once both are cysteines.	CA-distance cutoff defaults to 8 Å; adjust with --cutoff. Results land in predictions.csv (chain, residue numbers, prob).

All three stages are wrapped as CLI sub-commands, so one file drives the whole pipeline.


**2 | Installation:**

    conda create -n ssbond python=3.11
    conda activate ssbond
    pip install gemmi torch pandas numpy tqdm scikit-learn

**3 | Build & Extract Data**

      python NNdisulfide.py build \
      --data_dir /data/pdb-mmCIF \
      --out_csv disulfides.csv \
      --nproc 32        # adapt to your CPU budget

**4 | Train**

      python NNdisulfide.py train \
      --dataset disulfides.csv \
      --model ss_model.pt

**5 | Predict**

      python NNdisulfide.py predict \
      --model ss_model.pt \
      --structure my_enzyme.cif \
      --top_k 25 \
      --out my_enzyme_ss_predictions.csv
