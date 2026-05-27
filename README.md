# NNdisulfide

A lightweight feed-forward neural network for predicting candidate artificial disulfide bonds from protein structure files.

Given a PDB or mmCIF structure, NNdisulfide scans residue pairs that are close enough in 3D space and ranks pairs that may be suitable for mutation to cysteine in order to form an engineered disulfide bond.

The included `ss_model.pt` checkpoint can be used directly for prediction, or a new model can be trained from a directory of PDB/mmCIF structures.

## What NNdisulfide does

NNdisulfide provides three CLI subcommands:

| Command   | Purpose                                                                                                                                                                |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build`   | Parse a directory of structures, extract annotated or geometry-derived disulfide-positive cysteine pairs, generate nearby negative examples, and write a training CSV. |
| `train`   | Train a small PyTorch feed-forward neural network on the generated residue-pair feature table.                                                                         |
| `predict` | Scan a new structure, enumerate nearby residue pairs, score each pair with the trained model, and write the top predictions to CSV.                                    |

## How the model handles arbitrary-size PDB/mmCIF files

The neural network is **not** given the whole protein structure as one fixed-size input.

Instead, NNdisulfide converts each structure into a variable number of residue-pair examples:

1. The input PDB/mmCIF file is parsed with `gemmi`.
2. Residues are enumerated from the first model in the structure.
3. All residue pairs are considered.
4. Pairs are filtered by CB–CB distance using `--cutoff`, which defaults to `8.0 Å`.
5. Each remaining candidate pair is converted into a fixed-length vector of 16 numeric features.
6. The neural network scores each 16-feature row independently.

So a small protein may produce hundreds of candidate rows, while a larger protein may produce thousands or more. The number of rows can vary, but each row has the same 16-feature shape expected by the model.

## Model input features

Each candidate residue pair is represented by the following 16 numeric features:

| Feature      | Description                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| `ca_dist`    | CA–CA distance between the two residues, transformed with `log1p`.                                       |
| `cb_dist`    | CB–CB distance between the two residues, transformed with `log1p`. Glycine uses CA as a fallback for CB. |
| `seq_sep`    | Absolute sequence-number separation, transformed with `log1p`.                                           |
| `same_chain` | `1` if both residues are on the same chain, otherwise `0`.                                               |
| `phi1`       | Backbone phi angle for residue 1, scaled by `/180`.                                                      |
| `psi1`       | Backbone psi angle for residue 1, scaled by `/180`.                                                      |
| `chi1_1`     | Chi1-like side-chain torsion for residue 1, scaled by `/180`.                                            |
| `ang1`       | Local CA angle for residue 1, scaled by `/180`.                                                          |
| `asa1`       | Approximate solvent-accessible surface area value for residue 1.                                         |
| `b1`         | Mean B-factor for residue 1.                                                                             |
| `phi2`       | Backbone phi angle for residue 2, scaled by `/180`.                                                      |
| `psi2`       | Backbone psi angle for residue 2, scaled by `/180`.                                                      |
| `chi1_2`     | Chi1-like side-chain torsion for residue 2, scaled by `/180`.                                            |
| `ang2`       | Local CA angle for residue 2, scaled by `/180`.                                                          |
| `asa2`       | Approximate solvent-accessible surface area value for residue 2.                                         |
| `b2`         | Mean B-factor for residue 2.                                                                             |

Missing angular values are replaced with `0.0` before scoring.

## Neural network architecture

The current model architecture is:

```text
16 input features → 32 hidden units → 16 hidden units → 1 output logit
```

In PyTorch terms:

```python
nn.Linear(16, 32)
nn.ReLU()
nn.Linear(32, 16)
nn.ReLU()
nn.Linear(16, 1)
```

The output logit is converted to a probability with `sigmoid` during prediction.

## Installation

```bash
git clone https://github.com/linuxfold/NNdisulfide
cd NNdisulfide

conda create -n ssbond python=3.11
conda activate ssbond

pip install gemmi torch pandas numpy tqdm scikit-learn
```

## Build a training dataset

```bash
python NNdisulfide.py build \
  --data_dir /data/pdb-mmCIF \
  --out_csv disulfides.csv \
  --nproc 32
```

The `build` command searches the input directory recursively for:

* `.cif`
* `.mmcif`
* `.cif.gz`
* `.mmcif.gz`
* `.pdb`

It writes one row per cysteine-pair training example. Positive examples come from annotated disulfide connections and close SG–SG geometry. Negative examples are generated from nearby non-positive cysteine pairs.

## Train a model

```bash
python NNdisulfide.py train \
  --dataset disulfides.csv \
  --model ss_model.pt \
  --epochs 1000
```

Training uses binary cross-entropy with logits and the Adam optimizer. The best validation checkpoint is saved to the path given by `--model`.

## Predict candidate artificial disulfides

```bash
python NNdisulfide.py predict \
  --model ss_model.pt \
  --structure my_enzyme.cif \
  --top_k 25 \
  --out my_enzyme_ss_predictions.csv
```

Optional cutoff example:

```bash
python NNdisulfide.py predict \
  --model ss_model.pt \
  --structure my_enzyme.pdb \
  --cutoff 8.0 \
  --top_k 25 \
  --out my_enzyme_ss_predictions.csv
```

The output CSV contains:

| Column   | Description                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------ |
| `chain1` | Chain ID for residue 1.                                                                                      |
| `res1`   | Residue number for residue 1.                                                                                |
| `chain2` | Chain ID for residue 2.                                                                                      |
| `res2`   | Residue number for residue 2.                                                                                |
| `prob`   | Predicted probability that the pair is compatible with a disulfide-like geometry after mutation to cysteine. |

## Notes and limitations

* NNdisulfide is a pairwise geometric classifier, not a full protein-design model.
* Predictions should be treated as ranked candidates for further structural inspection, modeling, and experimental validation.
* The model scores residue pairs independently and does not explicitly model global folding changes caused by mutation.
* Only the first model in a multi-model structure is used.
* Input quality matters: missing atoms, unusual numbering, alternate conformations, or incomplete residues can affect feature extraction.
* Glycine uses CA as a fallback when CB is unavailable.

## Dependencies

* Python 3.11
* gemmi
* torch
* pandas
* numpy
* tqdm
* scikit-learn

## License

This project is licensed under the MIT License.

Permission is granted to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of this software, subject to preservation of
this notice and the standard MIT License warranty disclaimer.
