# NNdisulfide

Simple feed-forward neural network to predict candidate artificial disulfide bonds.

Given a PDB or mmCIF structure, NNdisulfide scans nearby residue pairs and ranks pairs that may be suitable for mutation to cysteine to create an engineered disulfide bond.

You can use the included `ss_model.pt` directly for prediction. It was trained on the same mmCIF files used as part of the AF3 database, including newer files up to 5/24/25.

## Quick start

```bash
git clone https://github.com/linuxfold/NNdisulfide
cd NNdisulfide

conda create -n ssbond python=3.11
conda activate ssbond

pip install gemmi torch pandas numpy tqdm scikit-learn
```

Run prediction with the included model:

```bash
python NNdisulfide.py predict \
  --model ss_model.pt \
  --structure my_enzyme.cif \
  --top_k 25 \
  --out my_enzyme_ss_predictions.csv
```

For PDB input:

```bash
python NNdisulfide.py predict \
  --model ss_model.pt \
  --structure my_enzyme.pdb \
  --top_k 25 \
  --out my_enzyme_ss_predictions.csv
```

The output CSV contains:

| Column   | Description                                  |
| -------- | -------------------------------------------- |
| `chain1` | Chain ID for residue 1                       |
| `res1`   | Residue number for residue 1                 |
| `chain2` | Chain ID for residue 2                       |
| `res2`   | Residue number for residue 2                 |
| `prob`   | Model score for the candidate disulfide pair |

## Commands

NNdisulfide has three CLI subcommands:

| Command   | What it does                                                        |
| --------- | ------------------------------------------------------------------- |
| `predict` | Scan a PDB/mmCIF file and rank candidate artificial disulfide bonds |
| `build`   | Parse a directory of structures and build a training CSV            |
| `train`   | Train a new PyTorch model from the CSV                              |

## Predict

```bash
python NNdisulfide.py predict \
  --model ss_model.pt \
  --structure input.cif \
  --cutoff 8.0 \
  --top_k 25 \
  --out predictions.csv
```

Arguments:

| Argument      | Description                                                |
| ------------- | ---------------------------------------------------------- |
| `--model`     | Path to a trained `.pt` model file                         |
| `--structure` | Input `.pdb`, `.cif`, or `.mmcif` structure                |
| `--cutoff`    | CB–CB distance cutoff for candidate pairs, default `8.0 Å` |
| `--top_k`     | Number of top predictions to write                         |
| `--out`       | Output CSV file                                            |

## Build training data

```bash
python NNdisulfide.py build \
  --data_dir /data/pdb-mmCIF \
  --out_csv disulfides.csv \
  --nproc 32
```

The build command searches recursively for:

* `.cif`
* `.mmcif`
* `.cif.gz`
* `.mmcif.gz`
* `.pdb`

It extracts positive cysteine-pair examples from annotated disulfide bonds and close SG–SG geometry, then generates nearby negative examples.

## Train

```bash
python NNdisulfide.py train \
  --dataset disulfides.csv \
  --model ss_model.pt \
  --epochs 1000
```

Training uses PyTorch with Adam and binary cross-entropy with logits. The best validation checkpoint is saved to `--model`.

## Model details

NNdisulfide does not feed the whole protein into the neural network. Instead, it converts each structure into residue-pair candidates. Each candidate pair is represented by 16 numeric features, and the model scores each pair independently.

Current network architecture:

```text
16 input features → 32 hidden units → 16 hidden units → 1 output logit
```

The 16 features are:

```text
ca_dist, cb_dist, seq_sep, same_chain,
phi1, psi1, chi1_1, ang1, asa1, b1,
phi2, psi2, chi1_2, ang2, asa2, b2
```

Distances and sequence separation are transformed with `log1p`; backbone and side-chain angles are scaled by `/180`.

## Notes

* Predictions are ranked candidates for further inspection, modeling, or experimental validation.
* The model is a pairwise geometric classifier, not a full protein-design model.
* Only the first model in a multi-model structure is used.
* Missing atoms or unusual residue numbering may affect results.
* Glycine uses CA as a fallback for CB.

## License

This project is licensed under the MIT License.

Permission is granted to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of this software, subject to preservation of
this notice and the standard MIT License warranty disclaimer.
