![logo](docs/logo_small.png)

---

## About Vitra

**Vitra** is a PyTorch-based force field designed to calculate the stability of a protein or complex in a differentiable way.

If you use Vitra in your research, please consider citing the related MadraX and FoldX publications.

---

## Documentation

### Energy Function Formulas (FoldX, MadraX, and Vitra)

Vitra reproduces the **FoldX** functional forms of protein energy terms, implemented in PyTorch tensors for differentiability—following the approach pioneered by **MadraX**. Each FoldX energy term is implemented identically in Vitra (via PyTorch), allowing gradients with respect to atomic positions to flow naturally. Vitra additionally introduces new differentiable penalties not present in FoldX.

#### Units & conventions
- Distances: Å  
- Angles: degrees  
- Charges: elementary charge (e)  
- Energies: kcal/mol (unless noted for coefficients)  
- ASA (solvent-accessible area): Å²

---

### Hydrogen Bond Energy (ΔG_hbond)
- **Basis:** FoldX orientation-dependent H-bond potential.  
- **Formula:**  
  \[
  E_{\text{hbond}} \propto S\,\exp[-a(d_{HA}-d_0)] \cdot f(\theta,\phi)
  \]
  where \(d_{HA}\) = H–acceptor distance (Å), \(\theta,\phi\) = bond angles (degrees), \(S,a,d_0\) = empirical parameters from FoldX.  
- **Units:** kcal/mol.

---

### Electrostatic Energy (ΔG_el)
- **Basis:** Coulomb law with Debye–Hückel screening.  
- **Formula:**  
  \[
  E_{el}^{ij} = \frac{332\,q_i q_j}{\epsilon_r\, d_{ij}} \exp\!\Big(-\frac{d_{ij}}{\kappa}\Big)
  \]
  where \(q_i,q_j\) = charges (e), \(d_{ij}\) = distance (Å), \(\epsilon_r\) = effective dielectric (dimensionless), \(\kappa\) = Debye length (Å).  
- **Units:** kcal/mol. (332 converts to kcal·Å/(mol·e²)).

---

### Disulfide Bond Energy (ΔG_SS)
- **Basis:** Favorable covalent S–S interaction (Cys–Cys) near 2.03 Å with geometry checks.  
- **Units:** kcal/mol.

---

### Polar Solvation Energy (ΔG_solvP)
- **Basis:** ASA-weighted polar transfer as in FoldX.  
- **Formula:**  
  \[
  \Delta G_{\text{solvP}} = \sum_{i\in\text{polar}} S_i^{P}\,\text{ASA}_i
  \]
  where \(S_i^{P}\) = polar coefficient (kcal/mol/Å²), \(\text{ASA}_i\) = area (Å²).  
- **Units:** kcal/mol (result).

---

### Hydrophobic Solvation Energy (ΔG_solvH)
- **Basis:** ASA-weighted apolar transfer as in FoldX.  
- **Formula:**  
  \[
  \Delta G_{\text{solvH}} = \sum_{i\in\text{nonpolar}} S_i^{H}\,\text{ASA}_i
  \]
  where \(S_i^{H}\) = hydrophobic coefficient (kcal/mol/Å²), \(\text{ASA}_i\) = area (Å²).  
- **Units:** kcal/mol (result).

---

### Van der Waals Energy (ΔG_vdw)
- **Basis:** Lennard–Jones–like pairwise term.  
- **Formula:**  
  \[
  \Delta G_{vdw}^{ij} = \epsilon_{ij}\Big[\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^{12} - 2\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^{6}\Big]
  \]
  where \(d_{ij}\) = distance (Å), \(R_{min,ij}\) = equilibrium separation (Å), \(\epsilon_{ij}\) = depth (kcal/mol).  
- **Units:** kcal/mol.

---

### Clash (Steric) Energy
- **Basis:** Exponential penalty for overlaps.  
- **Formula:**  
  \[
  E^{ij}_{\text{clash}} = w \,\exp\!\big[10\,(R_i+R_j - d_{ij} - t_g)\big], \quad d_{ij}<R_i+R_j+t_g
  \]
  where \(R_i,R_j\) = vdW radii (Å), \(d_{ij}\) = distance (Å), \(t_g\) = correction (Å), \(w\) = scale (kcal/mol).  
- **Units:** kcal/mol.

---

### Backbone Entropy (TΔS_mc)
- **Basis:** Empirical \(\phi,\psi,\omega\) preferences (KDE).  
- **Formula:**  
  \[
  E_{bb,i} = -w_1\ln\big[{\rm KDE}_\omega(\omega)\big] - w_2\ln\big[{\rm KDE}_{\phi\psi}(\phi,\psi)\big]
  \]
  where KDEs are probabilities (dimensionless), \(w_1,w_2\) (kcal/mol).  
- **Units:** kcal/mol.

---

### Side-Chain Entropy (TΔS_sc)
- **Basis:** Residue-specific entropy reduced by constraints.  
- **Formula:**  
  \[
  E_{sc,i} = -w_i \ln\big[{\rm KDE}_\chi(\chi_1,\chi_2,\dots)\big]
  \]
  where KDE\(_\chi\) is probability (dimensionless), \(w_i\) (kcal/mol).  
- **Units:** kcal/mol.

---

### Peptide Bond Geometry Violation (Vitra/MadraX-only)
- **Basis:** Gaussian penalty on peptide geometry.  
- **Formula:**  
  \[
  E_{\rm pep} = w \sum_{k\in\{\text{angles},\text{bond}\}} -\ln\big[G_k(X_k)\big]
  \]
  where \(X_k\) = angle (degrees) or bond length (Å), \(G_k\) = Gaussian probability (dimensionless), \(w\) (kcal/mol).  
- **Units:** kcal/mol.

---

### Side-Chain Conformation Violation (Vitra/MadraX-only)
- **Basis:** KDE penalty for rare rotamers.  
- **Formula:**  
  \[
  E_{\rm sc\_viol}(i) = -w_i \ln\big[{\rm KDE}_i(\boldsymbol{\chi})\big]
  \]
  where KDE\(_i\) is probability (dimensionless), \(w_i\) (kcal/mol).  
- **Units:** kcal/mol.

---

## Summary

- **Vitra** extends MadraX.

Sources: Definitions and formulas are taken from the MadraX documentation/paper and the original FoldX publications.

---

## Development

Vitra uses [Poetry](https://python-poetry.org/) for dependency and environment management.

### Setup
```sh
poetry install
```

### Running tests
```sh
poetry run pytest
```

### Linting and static analysis
We enforce code quality with **pylint**, **ruff**, **mypy**, and **deptry**:
```sh
poetry run pylint vitra
poetry run ruff check vitra
poetry run mypy vitra
poetry run deptry .
```

### Pre-commit hooks (optional)
```sh
poetry run pre-commit install
```
