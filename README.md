# Scalar Coherence Cosmology: CMB Spectra Generator

This repository contains the Jupyter notebook and extracted Python script used to generate the CMB temperature and polarization spectra (TT, EE, TE) using a redshift profile derived from classical scalar coherence. The method reproduces Planck 2018 results **without inflation or metric expansion**, relying on a redshift-distance relation defined by scalar gradient decay.

## How to Use

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```
   jupyter notebook notebook/tt_ee_te_polar.ipynb
   ```

3. Or run the Python script:
   ```
   python src/generate_spectra.py
   ```

## Output

- TT, EE, and TE angular spectra
- Comparison with Planck data overlays (if included)

## Citation

This code supports the model described in:

> *Reproducing the CMB and Matter Spectra Without Expansion or Inflation: A Classical Redshift from Scalar Coherence*, M. Shinn, 2025 (submitted to PRL)
