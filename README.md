![logo](docs/_static/logo_small.png)

---

## About MadraX

MadraX is a Pytorch-based Force Field designed to calculate the stability of a protein or complex in a differentiable way. 

MadraX is capable of interacting in an end-to-end way with neural networks and it can record the gradient via the autograd function of Pytorch

If you use MadraX in your research, please consider citing:


## Installation

For installation, refer to the official documentation: https://madrax.readthedocs.io/install.html

We recommend using MadraX with Python 3.7 3.8, 3.9 or 3.10. 
Package installation should only take a few minutes with any of these methods (conda, pip, source).

### Installing MadraX with [Anaconda](https://www.anaconda.com/download/):

```sh
 conda install -c grogdrinker madrax 
```

### Installing MadraX with pip:

We suggest to install pytorch separately following the instructions from https://pytorch.org/get-started/locally/

```sh
python -m pip install "madrax @git+https://bitbucket.org/grogdrinker/madrax/"
```

### Installing MadraX from source:

If you want to install MadraX from this repository, you need to install the dependencies first.
First, install [PyTorch](https://pytorch.org/get-started/locally/). The library is currently compatible with PyTorch versions between 1.8 and 1.13. We will continue to update MadraX to be compatible with the latest version of PyTorch.
You can also install Pytorch with the following command:

```sh
conda install pytorch -c pytorch
```

Finally, you can clone the repository with the following command:

```sh
git clone https://bitbucket.org/grogdrinker/madrax/
```

## Documentation

Energy Function Formulas (FoldX and MadraX)

MadraX implements the same physical energy terms as the FoldX force field, using PyTorch tensor operations for full differentiability ￼. In each case the FoldX functional form is preserved (with empirical parameters), while the MadraX implementation expresses that formula in tensor form. Below we list the mathematical form of each energy component. Where available, we cite the MadraX documentation/paper for the explicit form, and note the original FoldX basis.
	•	Hydrogen Bond Energy (ΔG_hbond): FoldX uses an orientation-dependent H-bond potential based on the distance between the hydrogen and acceptor electron orbital and the bond geometry (donor–H–acceptor and neighbor atom angles) ￼.  MadraX adopts the same form in PyTorch. In practice, the energy is often modeled by an exponential or inverse-distance term modulated by angle factors. For example, FoldX historically defines (in effect) an energy proportional to
$$E_{hbond}\propto S\exp[-a(d_{HA}-d_0)]\cdot f(\theta,\phi),, $$
where $d_{HA}$ is the H–acceptor distance, $\theta,\phi$ are ideal bond angles, and $S,a,d_0$ are parameters fit from data ￼.  MadraX computes the identical expression using PyTorch operations (so that gradients wrt atom positions flow through).  (Exact parameter values and angle functions are as in the original FoldX papers ￼.)
	•	Electrostatic Energy (ΔG_el): FoldX applies a Coulomb law with Debye–Hückel screening. The pairwise electrostatic energy between atoms $i,j$ with charges $q_i,q_j$ is given by (in kcal/mol)
$$E_{el}^{ij} = \frac{332,q_i q_j}{\epsilon_r,d_{ij}},\exp!\Big(-\frac{d_{ij}}{\kappa}\Big),, $$
where $d_{ij}$ is the inter-atomic distance, $\epsilon_r$ the effective dielectric, and $\kappa$ the Debye length (a function of temperature and ionic strength). MadraX implements this same screened Coulomb formula using tensor math.  (In FoldX, the 332 factor converts units; MadraX uses the same convention.)
	•	Disulfide Bond Energy (ΔG_SS): FoldX treats an S–S disulfide as a strong covalent bond: a favorable energy is assigned when two cysteine sulfurs are at the correct bonding distance/geometry. In MadraX the disulfide term is calculated “as in FoldX” ￼ (i.e. based on the S–S distance and bond angles). In essence, one can think of a narrow attractive well around the ideal S–S bond length (∼2.03 Å).  No explicit functional form is quoted in the available sources, but MadraX follows FoldX’s form exactly, implemented differentiably.
	•	Polar Solvation Energy (ΔG_solvP): FoldX decomposes solvation into polar and apolar (hydrophobic) contributions, each proportional to solvent-exposed surface area. Typically the energy is computed as
$$\Delta G_{solvP} = \sum_{i\in\text{polar atoms}} S_{i}^{P},\text{ASA}i,$$
where ASAi is the solvent-accessible area of atom/residue $i$ and $S{i}^{P}$ is a type-dependent coefficient (empirically derived free energy of transfer) ￼.  MadraX uses the same weighted-ASA form in tensor operations.
	•	Hydrophobic Solvation Energy (ΔG_solvH): Similarly, FoldX’s apolar (hydrophobic) solvation term is
$$\Delta G_{solvH} = \sum_{i\in\text{nonpolar atoms}} S_{i}^{H},\text{ASA}_i,$$
with coefficients $S_i^H$ fitted to transfer energies. MadraX implements this identical form.  (Together, these solvation terms account for the free-energy cost of exposing polar vs. nonpolar groups to solvent ￼.)
	•	Van der Waals Energy (ΔG_vdw): FoldX’s van der Waals term models short-range repulsion/dispersion. In practice it uses an empirical Lennard-Jones-like form or tabulated energy for each atom pair, derived from vapor-to-water transfer data ￼. A typical form is
$$\Delta G_{vdw}^{ij} = \epsilon_{ij}\Big[\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^{12} - 2\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^6\Big],$$
where $d_{ij}$ is the inter-atomic distance and $R_{min,ij},\epsilon_{ij}$ are fitted parameters. MadraX implements the same pairwise functional form in PyTorch (summing over all nonbonded pairs).
	•	Clash (Steric) Energy: This penalizes atoms that are too close. MadraX derives this penalty by grouping atom pairs (excluding bonded and special cases) and computing a threshold distance ($R_i+R_j+t_g$). If two atoms $i,j$ are closer than threshold, the clash energy is assigned by an exponential “growing” penalty ￼:
$$E^{ij}{\rm clash} ;=; w ,\exp\big[,10\big(R_i+R_j - d{ij} - t_g\big)\big],\quad d_{ij}<R_i+R_j+t_g,$$
where $R_i,R_j$ are van der Waals radii, $d_{ij}$ the distance, $t_g$ the group-specific correction, and $w$ a scaling factor ￼. (FoldX uses a similar functional form for overlaps. In MadraX this is implemented with tensors and automatic gradients.)
	•	Backbone Entropy (TΔS_mc): FoldX assigns an entropy cost for fixing the backbone dihedrals (φ,ψ,ω) from an empirical distribution. MadraX computes this as the log-probability of the observed angles under KDE-fitted distributions ￼. Specifically, for residue $i$ with backbone angles $(\phi,\psi,\omega)$:
$$E_{bb,i} ;=; -w_1\ln\big[{\rm KDE}\omega(\omega)\big] ;-; w_2\ln\big[{\rm KDE}{\phi\psi}(\phi,\psi)\big],$$
where KDE$\omega$ and KDE${\phi\psi}$ are the learned probability densities for the omega and phi-psi angles of that residue type, and $w_1,w_2$ are scale weights ￼. This matches FoldX’s backbone entropy term (FoldX uses tables of φ/ψ preferences).
	•	Side-Chain Entropy (TΔS_sc): FoldX includes an intrinsic side-chain entropy term for each amino acid (based on number of chi angles). In practice FoldX’s ΔS_sc is a constant per residue type scaled by solvent exposure. MadraX similarly uses a residue-specific entropy coefficient multiplied by the side chain’s accessibility ￼. In addition, MadraX reduces side-chain entropy if strong interactions (e.g. H-bonds) constrain a side chain. The exact formula is effectively:
$$E_{sc,i} = -w_i ,\ln\big[,KDE_{\chi}(\chi_1,\chi_2,\dots)\big],$$
i.e. the negative log-probability of the current χ torsions (see next term below) ￼. (FoldX’s original ΔS_sc is simpler, but MadraX combines that with the “violation” term below.)
	•	Peptide Bond Geometry Violation: (MadraX-only term) FoldX has no explicit peptide-bond penalty. MadraX adds a new term to softly enforce ideal peptide geometry.  For each consecutive residue pair, MadraX considers the two backbone angles (Cα–N–Cp and Cp–Cα–N, where Cp is the previous carbonyl C) and the C–N bond length. Each of these three values $X_i$ has a Gaussian target distribution $G_i(X)$. The energy penalty is the sum of negative log-likelihoods:
$$E_{\rm pep} = w\sum_{i\in{\text{angles},\text{bond}}} -\ln\big[G_i(X_i)\big],,\tag{3}$$
where $w$ is a scaling factor and $G_i(X_i)$ is the Gaussian probability for that angle/length ￼. This implements Eq. (3) from the MadraX paper, effectively penalizing deviations from ideal peptide bond geometry.
	•	Side-Chain Conformation Violation: (MadraX-only term) MadraX also adds a term penalizing rare side-chain rotamers. For each residue, it treats the vector of χ angles as a sample from a learned KDE distribution. The energy is the negative log-probability under this distribution:
$$E_{\rm sc_viol}(i) ;=; -w_i,\ln\big[{\rm KDE}_i(\boldsymbol{\chi})\big],\tag{4}$$
where ${\rm KDE}_i(\boldsymbol{\chi})$ is the kernel-estimated density for that residue’s side-chain angles, and $w_i$ a weight (Eq. 4 in the paper ￼). This penalizes conformations far from common rotamers. FoldX has no direct analog of this term.

In summary, MadraX implements each FoldX term with identical formulas (using PyTorch tensors). For example, the clash energy and backbone entropy terms above come directly from the MadraX documentation ￼ ￼, and the new peptide-bond and side-chain-violation penalties are given by Eqs. (3) and (4) in the MadraX preprint ￼ ￼. All such equations can be found in the MadraX reference and the original FoldX literature if needed, and MadraX’s code encodes them for GPU-accelerated, differentiable computation.

Sources: Definitions and formulas for these energy terms are given in the MadraX publication and documentation ￼ ￼ ￼ ￼, which closely follow the corresponding terms in FoldX ￼.Energy Function Formulas (FoldX and MadraX)

MadraX implements the same physical energy terms as the FoldX force field, using PyTorch tensor operations for full differentiability ￼. In each case the FoldX functional form is preserved (with empirical parameters), while the MadraX implementation expresses that formula in tensor form. Below we list the mathematical form of each energy component. Where available, we cite the MadraX documentation/paper for the explicit form, and note the original FoldX basis.
	•	Hydrogen Bond Energy (ΔG_hbond): FoldX uses an orientation-dependent H-bond potential based on the distance between the hydrogen and acceptor electron orbital and the bond geometry (donor–H–acceptor and neighbor atom angles) ￼.  MadraX adopts the same form in PyTorch. In practice, the energy is often modeled by an exponential or inverse-distance term modulated by angle factors. For example, FoldX historically defines (in effect) an energy proportional to
$$E_{hbond}\propto S\exp[-a(d_{HA}-d_0)]\cdot f(\theta,\phi),, $$
where $d_{HA}$ is the H–acceptor distance, $\theta,\phi$ are ideal bond angles, and $S,a,d_0$ are parameters fit from data ￼.  MadraX computes the identical expression using PyTorch operations (so that gradients wrt atom positions flow through).  (Exact parameter values and angle functions are as in the original FoldX papers ￼.)
	•	Electrostatic Energy (ΔG_el): FoldX applies a Coulomb law with Debye–Hückel screening. The pairwise electrostatic energy between atoms $i,j$ with charges $q_i,q_j$ is given by (in kcal/mol)
$$E_{el}^{ij} = \frac{332,q_i q_j}{\epsilon_r,d_{ij}},\exp!\Big(-\frac{d_{ij}}{\kappa}\Big),, $$
where $d_{ij}$ is the inter-atomic distance, $\epsilon_r$ the effective dielectric, and $\kappa$ the Debye length (a function of temperature and ionic strength). MadraX implements this same screened Coulomb formula using tensor math.  (In FoldX, the 332 factor converts units; MadraX uses the same convention.)
	•	Disulfide Bond Energy (ΔG_SS): FoldX treats an S–S disulfide as a strong covalent bond: a favorable energy is assigned when two cysteine sulfurs are at the correct bonding distance/geometry. In MadraX the disulfide term is calculated “as in FoldX” ￼ (i.e. based on the S–S distance and bond angles). In essence, one can think of a narrow attractive well around the ideal S–S bond length (∼2.03 Å).  No explicit functional form is quoted in the available sources, but MadraX follows FoldX’s form exactly, implemented differentiably.
	•	Polar Solvation Energy (ΔG_solvP): FoldX decomposes solvation into polar and apolar (hydrophobic) contributions, each proportional to solvent-exposed surface area. Typically the energy is computed as
$$\Delta G_{solvP} = \sum_{i\in\text{polar atoms}} S_{i}^{P},\text{ASA}i,$$
where ASAi is the solvent-accessible area of atom/residue $i$ and $S{i}^{P}$ is a type-dependent coefficient (empirically derived free energy of transfer) ￼.  MadraX uses the same weighted-ASA form in tensor operations.
	•	Hydrophobic Solvation Energy (ΔG_solvH): Similarly, FoldX’s apolar (hydrophobic) solvation term is
$$\Delta G_{solvH} = \sum_{i\in\text{nonpolar atoms}} S_{i}^{H},\text{ASA}_i,$$
with coefficients $S_i^H$ fitted to transfer energies. MadraX implements this identical form.  (Together, these solvation terms account for the free-energy cost of exposing polar vs. nonpolar groups to solvent ￼.)
	•	Van der Waals Energy (ΔG_vdw): FoldX’s van der Waals term models short-range repulsion/dispersion. In practice it uses an empirical Lennard-Jones-like form or tabulated energy for each atom pair, derived from vapor-to-water transfer data ￼. A typical form is
$$\Delta G_{vdw}^{ij} = \epsilon_{ij}\Big[\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^{12} - 2\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^6\Big],$$
where $d_{ij}$ is the inter-atomic distance and $R_{min,ij},\epsilon_{ij}$ are fitted parameters. MadraX implements the same pairwise functional form in PyTorch (summing over all nonbonded pairs).
	•	Clash (Steric) Energy: This penalizes atoms that are too close. MadraX derives this penalty by grouping atom pairs (excluding bonded and special cases) and computing a threshold distance ($R_i+R_j+t_g$). If two atoms $i,j$ are closer than threshold, the clash energy is assigned by an exponential “growing” penalty ￼:
$$E^{ij}{\rm clash} ;=; w ,\exp\big[,10\big(R_i+R_j - d{ij} - t_g\big)\big],\quad d_{ij}<R_i+R_j+t_g,$$
where $R_i,R_j$ are van der Waals radii, $d_{ij}$ the distance, $t_g$ the group-specific correction, and $w$ a scaling factor ￼. (FoldX uses a similar functional form for overlaps. In MadraX this is implemented with tensors and automatic gradients.)
	•	Backbone Entropy (TΔS_mc): FoldX assigns an entropy cost for fixing the backbone dihedrals (φ,ψ,ω) from an empirical distribution. MadraX computes this as the log-probability of the observed angles under KDE-fitted distributions ￼. Specifically, for residue $i$ with backbone angles $(\phi,\psi,\omega)$:
$$E_{bb,i} ;=; -w_1\ln\big[{\rm KDE}\omega(\omega)\big] ;-; w_2\ln\big[{\rm KDE}{\phi\psi}(\phi,\psi)\big],$$
where KDE$\omega$ and KDE${\phi\psi}$ are the learned probability densities for the omega and phi-psi angles of that residue type, and $w_1,w_2$ are scale weights ￼. This matches FoldX’s backbone entropy term (FoldX uses tables of φ/ψ preferences).
	•	Side-Chain Entropy (TΔS_sc): FoldX includes an intrinsic side-chain entropy term for each amino acid (based on number of chi angles). In practice FoldX’s ΔS_sc is a constant per residue type scaled by solvent exposure. MadraX similarly uses a residue-specific entropy coefficient multiplied by the side chain’s accessibility ￼. In addition, MadraX reduces side-chain entropy if strong interactions (e.g. H-bonds) constrain a side chain. The exact formula is effectively:
$$E_{sc,i} = -w_i ,\ln\big[,KDE_{\chi}(\chi_1,\chi_2,\dots)\big],$$
i.e. the negative log-probability of the current χ torsions (see next term below) ￼. (FoldX’s original ΔS_sc is simpler, but MadraX combines that with the “violation” term below.)
	•	Peptide Bond Geometry Violation: (MadraX-only term) FoldX has no explicit peptide-bond penalty. MadraX adds a new term to softly enforce ideal peptide geometry.  For each consecutive residue pair, MadraX considers the two backbone angles (Cα–N–Cp and Cp–Cα–N, where Cp is the previous carbonyl C) and the C–N bond length. Each of these three values $X_i$ has a Gaussian target distribution $G_i(X)$. The energy penalty is the sum of negative log-likelihoods:
$$E_{\rm pep} = w\sum_{i\in{\text{angles},\text{bond}}} -\ln\big[G_i(X_i)\big],,\tag{3}$$
where $w$ is a scaling factor and $G_i(X_i)$ is the Gaussian probability for that angle/length ￼. This implements Eq. (3) from the MadraX paper, effectively penalizing deviations from ideal peptide bond geometry.
	•	Side-Chain Conformation Violation: (MadraX-only term) MadraX also adds a term penalizing rare side-chain rotamers. For each residue, it treats the vector of χ angles as a sample from a learned KDE distribution. The energy is the negative log-probability under this distribution:
$$E_{\rm sc_viol}(i) ;=; -w_i,\ln\big[{\rm KDE}_i(\boldsymbol{\chi})\big],\tag{4}$$
where ${\rm KDE}_i(\boldsymbol{\chi})$ is the kernel-estimated density for that residue’s side-chain angles, and $w_i$ a weight (Eq. 4 in the paper ￼). This penalizes conformations far from common rotamers. FoldX has no direct analog of this term.

In summary, MadraX implements each FoldX term with identical formulas (using PyTorch tensors). For example, the clash energy and backbone entropy terms above come directly from the MadraX documentation ￼ ￼, and the new peptide-bond and side-chain-violation penalties are given by Eqs. (3) and (4) in the MadraX preprint ￼ ￼. All such equations can be found in the MadraX reference and the original FoldX literature if needed, and MadraX’s code encodes them for GPU-accelerated, differentiable computation.

Sources: Definitions and formulas for these energy terms are given in the MadraX publication and documentation ￼ ￼ ￼ ￼, which closely follow the corresponding terms in FoldX ￼.