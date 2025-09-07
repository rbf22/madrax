#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time
import torch

from vitra.mutate import rotator
from vitra.sources import hashings

letters = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ASN': 'N', 'PRO': 'P', 'THR': 'T', 'PHE': 'F',
           'ALA': 'A', 'HIS': 'H', 'GLY': 'G', 'ILE': 'I', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'VAL': 'V', 'GLU': 'E',
           'TYR': 'Y', 'MET': 'M'}  # gli aminoacidi, che male qui non fanno


def optimize(model, coords, info_tensors, epochs=50, verbose=False, learning_rate=0.1, backbone_rotation=False):
    """
    function that minimizes the energy of protein(s) or complex(es)

    Parameters ---------- model : vitra.ForceField A vitra ForceField object. It can be generated using the
    vitra.ForceField function coords : torch.Tensor shape: (Batch, nAtoms, 3) coordinates of the proteins. It can be
    generated using the vitra.utils.parse_pdb function info_tensors : tuple a set of precalculated information
    tensors required by the forcefield. It can be created out of the box with the function
    vitra.data_structures.create_info_tensors starting from the atom names (that can be obtained, along with the
    coordinates, by the  vitra.utils.parse_pdb function) epochs : int number of optimization epochs default = 50
    learning_rate : float learning rate of the optimization default = 0.1 verbose : bool if you wanna see a lot of
    text everywhere in your terminal

    Returns
    -------
    yp : torch.Tensor
    shape: shape (Batch, nChains, nResi, nMutants, 10)
    The Gibbs energy of the input proteins. The dimensions are organized as follow:

    The batch dimension refers to the protein number (same order of the one defined by the vitra.utils.parse_pdb
    finction), chain refers to the chain index (sorted alphabetically), residue number refers to the residue
    position, nMutants refers to the mutants you might have implemented in the calculation (this dimension is 1 if no
    mutants have been added). The last dimension refers to the different types of energy:

    0: Disulfide bonds Energy
    1: Hydrogen Bonds Energy
    2: Electrostatics Energy
    3: Van der Waals Clashes
    4: Polar Solvation Energy
    5: Hydrophobic Solvation Energy
    6: Van der Waals Energy
    7: Backbone Entropy
    8: Side Chain Entropy
    9: Peptide Bond Violations
    10: Rotamer Violation

    coords_local: torch.tensor
                shape: (Batch, nAtoms)
                coordinates of the optimized proteins
    """

    dev = model.device
    rotator_obj = rotator.RotateStruct(dev=dev)

    atom_number, atom_description, coordsIndexingAtom, partnersIndexingAtom, \
        angle_indices, alternativeMask = info_tensors

    batch = atom_description[:, hashings.atom_description_hash["batch"]].max() + 1
    maxseq = atom_description[:, hashings.atom_description_hash["resnum"]].max() + 1
    maxchain = atom_description[:, hashings.atom_description_hash["chain"]].max() + 1

    naltern = alternativeMask.shape[-1]
    dummy_rotationSC = torch.zeros((batch, maxchain, maxseq, naltern, 8), device=dev, dtype=torch.float)

    dummy_translationBB = torch.zeros((batch, maxchain, maxseq, naltern, 3), device=dev, dtype=torch.float)
    dummy_rotationSC.requires_grad = True
    dummy_translationBB.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': dummy_translationBB, "lr": learning_rate},
        {'params': dummy_rotationSC, 'lr': learning_rate}
    ], amsgrad=True, eps=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)

    optimizer.zero_grad()

    startRotation = []

    for e in range(epochs):
        start_time = time.time()

        dummy_rotationSCAngles = torch.sin(dummy_rotationSC) * math.pi

        start = time.time()

        coords_local = rotator_obj(coords, info_tensors, dummy_rotationSCAngles, backbone_rotation=backbone_rotation)

        startRotation += [start - time.time()]

        yp = model(coords_local, info_tensors)
        loss = yp.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(float(loss.data.cpu()))
        if verbose:
            print(" \t optimizing epoch:", e, "loss:", round(yp.sum().data.cpu().tolist(), 4), "time ",
                  round((time.time() - start_time), 4))

    return yp, coords_local
