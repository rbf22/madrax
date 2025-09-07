#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  parse_foldx_params.py
#  
#  Copyright 2020 Gabriele Orlando <orlando.gabriele89@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import os

letters = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ASN': 'N', 'PRO': 'P', 'THR': 'T', 'PHE': 'F',
           'ALA': 'A', 'HIS': 'H', 'GLY': 'G', 'ILE': 'I', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'VAL': 'V', 'GLU': 'E',
           'TYR': 'Y', 'MET': 'M'}  # gli aminoacidi, che male qui non fanno
hashing_hybrid = {"NO_HYBRID": 0, "SP2_N_H1": 1, "SP2_N_H2": 2, "SP2_N_ORB1": 3, "SP2_N_ORB2": 4, "SP3_N_H3": 5,
                  "SP3_O_H1ORB2": 6, "SP2_O_ORB2": 7}

path = os.path.dirname(os.path.abspath(__file__))


def read_h_positions(fil=path + "/../parameters/hbond_coords_params.txt"):
    # ADD BACKBONE PARTNERS MANUALLY!!! #
    diz = {}
    for line in open(fil).readlines():
        data = line.split("\t")

        if data[0] not in letters and data[0] != "ooo":
            continue

        aa = data[0]
        if data[6] == "_" or data[6][0] != "H":
            continue

        if aa not in diz:
            diz[aa] = {}

        if data[1] not in diz[aa]:
            diz[aa][data[1]] = {}

        if data[6] + "_0" in diz[aa][data[1]]:
            # continue
            cont = 1
            nameH = data[6] + "_" + str(cont)
            while nameH in diz[aa][data[1]]:
                nameH = data[6] + "_" + str(cont)
                cont += 1
            data[6] = nameH
        else:
            data[6] = data[6] + "_0"
        diz[aa][data[1]][data[6]] = (float(data[7]), float(data[8]), float(data[9]))

    for aa in letters.keys():

        if aa != "PRO":
            diz[aa]["N"] = {}
            diz[aa]["N"]["HN_0"] = diz["ooo"]["N"]["HN_0"]

    del diz["ooo"]

    return diz


def read_hbond_params(fil=path + "/../parameters/hbond_params.txt"):
    level_hashing = {
        "LEVEL_N": 0,
        "LEVEL_A": 1,
        "LEVEL_O": -1,
        "LEVEL_B": 2,
        "LEVEL_G": 3,
        "LEVEL_D": 4,
        "LEVEL_E": 5,
        "LEVEL_Z": 6,
        "LEVEL_H": 7,
        "LEVEL_I": 8,
        "LEVEL_K": 9,
    }

    header = []
    fin = {}
    for line in open(fil, "r").readlines():
        if line[0] == "#":
            header += [line.strip().replace("#", "")]
        elif line.strip() == "":
            continue
        else:
            data = line.strip().replace('"', "").replace(' ', "").replace('\t', "").strip(",").split(",")

            assert len(data) == len(header)
            diz = {}
            aa_ind = -1
            at_ind = -1
            for k in range(len(data)):
                if header[k] == "aa":
                    aa_ind = k

                elif header[k] == "atom":
                    at_ind = k
                elif header[k] == "level":
                    diz[header[k]] = level_hashing[data[k]]
                elif header[k] != "hybridation":
                    try:
                        diz[header[k]] = float(data[k])
                    except ValueError:
                        diz[header[k]] = data[k]

                else:
                    diz[header[k]] = hashing_hybrid[data[k]]

            if aa_ind == -1:
                raise ValueError("Header does not contain 'aa' column")
            if at_ind == -1:
                raise ValueError("Header does not contain 'atom' column")

            if data[aa_ind] not in fin:
                fin[data[aa_ind]] = {}

            fin[data[aa_ind]][data[at_ind]] = diz

    return fin


def read_hbond_partners(fil=path + "/../parameters/hbond_coords_params.txt"):
    # this should be replaced with some other data structure
    diz = {}
    for line in open(fil).readlines():
        data = line.split("\t")

        if data[0] not in letters:
            continue
        else:
            aa = data[0]
        if data[6] == "_":
            continue

        if aa not in diz:
            diz[aa] = {}

        diz[aa][data[1]] = [[data[2], 0], [data[4], 0]]

        # backbone manually #

    for aa in letters.keys():
        if aa != "PRO":
            diz[aa]["N"] = [["C", -1], ["O", -1]]

        diz[aa]["C"] = [["CA", 0], ["N", 0]]
        diz[aa]["O"] = [["C", 0], ["CA", 0]]

    # virtual manually #
    diz["ARG"]["ARG"] = [["NE", 0], ["NH1", 0]]
    diz["ARG"]["CZ"] = [["NE", 0], ["NH1", 0]]
    diz["PHE"]["RC"] = [["CG", 0], ["CD1", 0]]
    diz["TYR"]["RC"] = [["CG", 0], ["CD1", 0]]

    return diz


def read_FO_positions(fil=path + "/../parameters/hbond_coords_params.txt"):
    diz = {}
    for line in open(fil).readlines():
        data = line.split("\t")

        if not (data[0] in letters or data[0] == "ooo"):
            continue

        aa = data[0]

        if len(data[6]) < 2 or data[6][:2] != "FO":
            continue

        if aa not in diz:
            diz[aa] = {}

        if data[1] not in diz[aa]:
            diz[aa][data[1]] = {}
        data[6] = data[6].replace("FO", "X")  # names are too long for pdb

        if data[6] + "_0" in diz[aa][data[1]]:
            cont = 1
            nameFO = data[6] + "_" + str(cont)
            while nameFO in diz[aa][data[1]]:
                nameFO = data[6] + "_" + str(cont)
                cont += 1
            data[6] = nameFO
        else:
            data[6] = data[6] + "_0"

        diz[aa][data[1]][data[6]] = (float(data[7]), float(data[8]), float(data[9]))

    for aa in letters.keys():
        if aa not in diz:
            diz[aa] = {}

        diz[aa]["O"] = {}
        diz[aa]["O"]["X1_0"] = diz["ooo"]["O"]["X1_0"]
        diz[aa]["O"]["X2_0"] = diz["ooo"]["O"]["X2_0"]

    del diz["ooo"]

    return diz


if __name__ == '__main__':

    print('probably we need some test code here')
