#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Electrostatics.py
#  
#  Copyright 2019 Gabriele Orlando <orlando.gabriele89@gmail.com>
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

import torch,math,os

from vitra.sources import hashings,math_utils

from vitra.sources.globalVariables import *


class BondLenConstrain(torch.nn.Module):

	def __init__(self, name = "AngleScorer", dev = "cpu"):
		self.name=name
		self.dev = dev

		self.float_type = torch.float

		super(BondLenConstrain, self).__init__()

		self.weight = torch.nn.Parameter(torch.tensor([0.5],device=self.dev))
		self.weight.requires_grad = True

		#self.weight = 0.001

		### generating distributions ###
		if not "violations.m" in os.listdir("/".join(os.path.realpath(__file__).split("/")[:-1])+"/../parameters/"):
			self.fit()
		self.mean,self.std = torch.load("/".join(os.path.realpath(__file__).split("/")[:-1])+"/../parameters/violations.m")
		self.mean = self.mean.to(self.dev)
		self.std = self.std.to(self.dev)
		self.Natoms = []
		self.Catoms = []
		self.CAatoms = []
		for res1 in hashings.resi_hash.keys():
			self.Natoms += [hashings.atom_hash[res1]["N"]]
			self.Catoms += [hashings.atom_hash[res1]["C"]]
			self.CAatoms += [hashings.atom_hash[res1]["CA"]]

		self.Natoms = list(set(self.Natoms))
		self.Catoms = list(set(self.Catoms))
		self.CAatoms = list(set(self.CAatoms))

		self.weight = torch.nn.Parameter(torch.tensor([-5.0],device=self.dev))
		self.weight.requires_grad=True


	def scoreDistro(self,inputs,seq):

		score=[]
		for i in range(self.std.shape[1]):
			var = self.std[seq,i] ** 2
			denom = (2 * math.pi * var) ** .5
			num = torch.exp(-(inputs[:,i] - self.mean[seq,i]) ** 2 / (2 * var))
			norm_factor = 1.0/denom


			score += [-((num/denom).clamp(min=EPS).log() -torch.log(norm_factor)).unsqueeze(-1)]
		return torch.cat(score,dim=-1)


	def forward(self, atom_description,coords,alternatives):
		resnum = atom_description[:, hashings.atom_description_hash["resnum"]].long()
		atname = atom_description[:, hashings.atom_description_hash["at_name"]].long()
		chain = atom_description[:, hashings.atom_description_hash["chain"]].long()
		resname = atom_description[:, hashings.atom_description_hash["resname"]].long().to(self.dev)

		# m1 = (coords == 100.9290).sum(-1).bool()
		# m2 = (coords == -99.9220).sum(-1).bool()

		batch_ind = atom_description[:, hashings.atom_description_hash["batch"]].long()

		maxres = resnum.max() + 1
		nbatch = batch_ind.max() + 1
		natoms = atom_description.shape[1]
		maxchain = chain.max() + 1
		nalt =alternatives.shape[-1]

		resiEnergy = torch.zeros((nbatch,maxchain,maxres,nalt),dtype=torch.float,device=self.dev)

		for alt in range(nalt):
			mask = alternatives[...,alt]
			batch_ind_resi = torch.arange(nbatch,device=self.dev).unsqueeze(-1).unsqueeze(-1).expand(nbatch, maxchain, maxres)
			chain_resi = torch.arange(maxchain,device=self.dev).unsqueeze(0).unsqueeze(-1).expand(nbatch, maxchain, maxres)

			Narray = torch.full((nbatch, maxchain, maxres, 3), PADDING_INDEX, device=self.dev, dtype=self.float_type)
			Carray = torch.full((nbatch, maxchain, maxres, 3), PADDING_INDEX, device=self.dev, dtype=self.float_type)
			CAarray = torch.full((nbatch, maxchain, maxres, 3), PADDING_INDEX, device=self.dev, dtype=self.float_type)

			seq = torch.full((nbatch, maxchain, maxres), PADDING_INDEX, device=self.dev, dtype=torch.long)

			a = atom_description[:, hashings.atom_description_hash["at_name"]]
			for i, natom in enumerate(self.Natoms):
				if i == 0:
					natom_mask = a.eq(natom)
				else:
					natom_mask += a.eq(natom) & mask

			for i, caatom in enumerate(self.CAatoms):
				if i == 0:
					caatom_mask = a.eq(caatom)
				else:
					caatom_mask += a.eq(caatom) & mask

			for i, catom in enumerate(self.Catoms):
				if i == 0:
					catom_mask = a.eq(catom)
				else:
					catom_mask += a.eq(catom) & mask
			del a

			Narray.index_put_((batch_ind[natom_mask], chain[natom_mask], resnum[natom_mask]), coords[natom_mask])
			Carray.index_put_((batch_ind[catom_mask], chain[catom_mask], resnum[catom_mask]), coords[catom_mask])
			CAarray.index_put_((batch_ind[caatom_mask], chain[caatom_mask], resnum[caatom_mask]), coords[caatom_mask])

			seq.index_put_((batch_ind[caatom_mask], chain[caatom_mask], resnum[caatom_mask]), resname[caatom_mask])

			index = torch.arange(1, maxres,device=self.dev).unsqueeze(0).unsqueeze(0).expand(nbatch, maxchain, maxres - 1)
			previous_index = torch.arange(maxres - 1,device=self.dev).unsqueeze(0).unsqueeze(0).expand(nbatch, maxchain, maxres - 1)

			pepBonds = torch.cat([Narray[batch_ind_resi[:, :, 1:], chain_resi[:, :, 1:], index].unsqueeze(-2),
								  Carray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2)],
								 dim=-2)

			pad_mask = (seq[:, :, 1:] != PADDING_INDEX) & (seq[:, :, :-1] != PADDING_INDEX)
			todo = ~(pepBonds[:, :, :, :, 0].eq(PADDING_INDEX).sum(-1).bool()) & pad_mask
			peptide_bond_lenth = torch.norm(pepBonds[:, :, :, 0][todo] - pepBonds[:, :, :, 1][todo], dim=-1)


			C_N_CA_AngleAtoms = torch.cat(
				[Carray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2),
				 Narray[batch_ind_resi[:, :, 1:], chain_resi[:, :, 1:], index].unsqueeze(-2),
				 CAarray[batch_ind_resi[:, :, 1:], chain_resi[:, :, 1:], index].unsqueeze(-2)], dim=-2)

			CA_C_N_AngleAtoms = torch.cat(
				[CAarray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2),
				 Carray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2),
				 Narray[batch_ind_resi[:, :, 1:], chain_resi[:, :, 1:], index].unsqueeze(-2)], dim=-2)

			todo = ~(C_N_CA_AngleAtoms[:, :, :, :, 0].eq(PADDING_INDEX).sum(-1).bool()) & (
				~(CA_C_N_AngleAtoms[:, :, :, :, 0].eq(PADDING_INDEX).sum(-1).bool())) & pad_mask

			C_N_CA_Angle, nanmask, monkeys = math_utils.angle2dVectors(
				C_N_CA_AngleAtoms[:, :, :, 0][todo] - C_N_CA_AngleAtoms[:, :, :, 1][todo],
				C_N_CA_AngleAtoms[:, :, :, 2][todo] - C_N_CA_AngleAtoms[:, :, :, 1][todo])

			# CA C N

			CA_C_N_Angle, nanmask, monkeys = math_utils.angle2dVectors(
				CA_C_N_AngleAtoms[:, :, :, 0][todo] - CA_C_N_AngleAtoms[:, :, :, 1][todo],
				CA_C_N_AngleAtoms[:, :, :, 2][todo] - CA_C_N_AngleAtoms[:, :, :, 1][todo])
			distro_input = torch.cat([peptide_bond_lenth.unsqueeze(-1), C_N_CA_Angle, CA_C_N_Angle], dim=1)
			scores = self.scoreDistro(distro_input,seq[...,1:][todo]).sum(-1)

			resiEnergy[batch_ind_resi[...,1:][todo],chain_resi[...,1:][todo],index[todo],alt] = scores * (1 - torch.tanh(-self.weight))

		return resiEnergy
	def fit(self, dataset_folder="dataset/pdb/"):

		data = []
		ERASE_MARSHALLED = False
		aa = ['C', 'D', 'S', 'Q', 'K', 'N', 'P', 'T', 'F', 'A', 'H', 'G', 'I', 'L', 'R', 'W', 'V', 'E', 'Y', 'M']
		diz = {}

		#pdb_dir = "dataset/hiresPDB/"
		pdb_dir = "dataset/mariannePdb/"
		from sources import utils,dataStructures,math_utils
		coo, atnames = utils.parsePDB(pdb_dir)
		device = "cuda"

		Natoms = []
		Catoms = []
		Oatoms = []
		CAatoms = []
		for res1 in hashings.resi_hash.keys():
			Natoms += [hashings.atom_hash[res1]["N"]]
			#Natoms += [hashings.atom_hash[res1]["tN"]]
			Catoms += [hashings.atom_hash[res1]["C"]]
			Oatoms += [hashings.atom_hash[res1]["O"]]
			CAatoms += [hashings.atom_hash[res1]["CA"]]

		Natoms = list(set(Natoms))
		Catoms = list(set(Catoms))
		Oatoms = list(set(Oatoms))
		CAatoms = list(set(CAatoms))


		if True:
			ang1 ={}
			ang2 ={}
			ang3 = {}
			lens ={}
			split_numb = 300
			for split in range(split_numb,len(atnames),split_numb):
				print(split)
				coordinates = coo[split-split_numb:split]
				info_tensors = dataStructures.create_info_tensors(atnames[split-split_numb:split], device=device, verbose=True)
				atom_number, atom_description, coordsIndexingAtom, partnersIndexingAtom, angle_indices, alternativeMask = info_tensors

				partnersFinal1 = torch.full((atom_description.shape[0],3),float(PADDING_INDEX),device=self.dev)
				partnersFinal2 = torch.full((atom_description.shape[0],3),float(PADDING_INDEX),device=self.dev)

				padPart1 = partnersIndexingAtom[..., 0] != PADDING_INDEX
				coordsP1 = coordinates[atom_description[:, 0].long()[padPart1], partnersIndexingAtom[..., 0][padPart1]]
				partnersFinal1[padPart1] = coordsP1

				padPart2 = partnersIndexingAtom[..., 1] != PADDING_INDEX
				coordsP2 = coordinates[atom_description[:, 0].long()[padPart2], partnersIndexingAtom[..., 1][padPart2]]
				partnersFinal2[padPart2] = coordsP2

				del coordsP1,coordsP2,padPart1,padPart2

				coords = coordinates[atom_description[:, 0].long(), coordsIndexingAtom]



				resnum = atom_description[:, hashings.atom_description_hash["resnum"]].long()
				atname = atom_description[:, hashings.atom_description_hash["at_name"]].long()
				chain  = atom_description[:, hashings.atom_description_hash["chain"]].long()
				resname  = atom_description[:, hashings.atom_description_hash["resname"]].long().to(self.dev)

				#m1 = (coords == 100.9290).sum(-1).bool()
				#m2 = (coords == -99.9220).sum(-1).bool()

				batch_ind  = atom_description[:, hashings.atom_description_hash["batch"]].long()

				maxres = resnum.max() + 1
				nbatch = batch_ind.max() +1
				natoms = atom_description.shape[1]
				maxchain = chain.max() + 1

				batch_ind_resi = torch.arange(nbatch).unsqueeze(-1).unsqueeze(-1).expand(nbatch, maxchain, maxres)
				chain_resi =  torch.arange(maxchain).unsqueeze(0).unsqueeze(-1).expand(nbatch, maxchain, maxres)

				Narray = torch.full((nbatch,maxchain,maxres,3),PADDING_INDEX,device=self.dev,dtype=self.float_type)
				Carray = torch.full((nbatch,maxchain, maxres,3), PADDING_INDEX, device=self.dev, dtype=self.float_type)
				CAarray = torch.full((nbatch,maxchain,maxres,3),PADDING_INDEX,device=self.dev,dtype=self.float_type)

				seq = torch.full((nbatch,maxchain,maxres),PADDING_INDEX,device=self.dev,dtype=torch.long)

				a = atom_description[:, hashings.atom_description_hash["at_name"]]
				for i, natom in enumerate(Natoms):
					if i == 0:
						natom_mask = a.eq(natom)
					else:
						natom_mask += a.eq(natom)

				for i, caatom in enumerate(CAatoms):
					if i == 0:
						caatom_mask = a.eq(caatom)
					else:
						caatom_mask += a.eq(caatom)

				for i, catom in enumerate(Catoms):
					if i == 0:
						catom_mask = a.eq(catom)
					else:
						catom_mask += a.eq(catom)
				del a


				Narray.index_put_((batch_ind[natom_mask], chain[natom_mask], resnum[natom_mask]),coords[natom_mask])
				Carray.index_put_((batch_ind[catom_mask], chain[catom_mask], resnum[catom_mask]),coords[catom_mask])
				CAarray.index_put_((batch_ind[caatom_mask], chain[caatom_mask], resnum[caatom_mask]),coords[caatom_mask])

				seq.index_put_((batch_ind[caatom_mask], chain[caatom_mask], resnum[caatom_mask]),resname[caatom_mask])

				index = torch.arange(1, maxres).unsqueeze(0).unsqueeze(0).expand(nbatch, maxchain, maxres - 1)
				previous_index = torch.arange(maxres - 1).unsqueeze(0).unsqueeze(0).expand(nbatch, maxchain, maxres - 1)


				pepBonds = torch.cat([Narray[batch_ind_resi[:, :, 1:] , chain_resi[:, :, 1: ], index].unsqueeze(-2),
									  Carray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2)],
									 dim=-2)

				pad_mask = (seq[:,:,1:] != PADDING_INDEX) & (seq[:,:,:-1] != PADDING_INDEX)
				todo = ~(pepBonds[:, :, :, :, 0].eq(PADDING_INDEX).sum(-1).bool()) & pad_mask
				peptide_bond_lenth = torch.norm(pepBonds[:, :, :, 0][todo] - pepBonds[:, :, :, 1][todo], dim=-1)
				if peptide_bond_lenth.max().ge(2.5):
					sad
				#qui ci sono distanze di 15
				for i,r in enumerate(seq[..., :-1][todo]):
					res = int(r)
					if not res in lens:
						lens[res] = []

					lens[res] += [peptide_bond_lenth[i]]


				C_N_CA_AngleAtoms = torch.cat(
					[Carray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2),
					 Narray[batch_ind_resi[:, :, 1:], chain_resi[:, :, 1:], index].unsqueeze(-2),
					 CAarray[batch_ind_resi[:, :, 1:], chain_resi[:, :, 1:], index].unsqueeze(-2)], dim=-2)

				CA_C_N_AngleAtoms = torch.cat(
					[CAarray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2),
					 Carray[batch_ind_resi[:, :, :-1], chain_resi[:, :, :-1], previous_index].unsqueeze(-2),
					 Narray[batch_ind_resi[:, :, 1:], chain_resi[:, :, 1:], index].unsqueeze(-2)], dim=-2)

				todo = ~(C_N_CA_AngleAtoms[:, :, :, :, 0].eq(PADDING_INDEX).sum(-1).bool()) & (~(CA_C_N_AngleAtoms[:, :, :, :, 0].eq(PADDING_INDEX).sum(-1).bool())) & pad_mask

				C_N_CA_Angle, nanmask, monkeys = math_utils.angle2dVectors(
					C_N_CA_AngleAtoms[:, :, :, 0][todo] - C_N_CA_AngleAtoms[:, :, :, 1][todo],
					C_N_CA_AngleAtoms[:, :, :, 2][todo] - C_N_CA_AngleAtoms[:, :, :, 1][todo])

				# CA C N


				CA_C_N_Angle, nanmask, monkeys = math_utils.angle2dVectors(
					CA_C_N_AngleAtoms[:, :, :, 0][todo] - CA_C_N_AngleAtoms[:, :, :, 1][todo],
					CA_C_N_AngleAtoms[:, :, :, 2][todo] - CA_C_N_AngleAtoms[:, :, :, 1][todo])

				for i,r in enumerate(seq[...,:-1][todo]):
					res = int(r)

					if not res in ang1:
						ang1[res] = []
						ang2[res] = []

					ang1[res] += [C_N_CA_Angle.squeeze(1)[i]]
					ang2[res] += [CA_C_N_Angle.squeeze(1)[i]]


		for res in ang1.keys():
			ang1[res] = torch.stack(ang1[res])
			ang2[res] = torch.stack(ang2[res])
		for res in lens.keys():
			lens[res] = torch.stack(lens[res])
		asd



		## peptide bond lenght##





		PLOT=False
		if PLOT:
			import matplotlib.pylab as plt
			for r in lens.keys():
				plt.title(hashings.resi_hash_inverse[r])
				plt.hist(lens[r].cpu().tolist(),bins=50)
				print(hashings.resi_hash_inverse[r],"mean=",lens[r].mean(),"mean=",lens[r].max(),len(lens[r]))
				plt.xlabel("N-C distance")
				plt.show()

		### peptide bond angles violation ###

		# C N CA

		if PLOT:
			for r in ang2.keys():
				plt.title(hashings.resi_hash_inverse[r])
				plt.hist(ang2[r].cpu().tolist(),bins=50)
				print(hashings.resi_hash_inverse[r],"mean=",ang2[r].mean(),"mean=",ang2[r].max(),len(ang2[r]))
				plt.xlabel("ang2")
				plt.show()


		if PLOT:
			for r in ang1.keys():
				plt.title(hashings.resi_hash_inverse[r])
				plt.hist(ang1[r].cpu().tolist(),bins=50)
				print(hashings.resi_hash_inverse[r],"mean=",ang1[r].mean(),"mean=",ang1[r].max(),len(ang1[r]))
				plt.xlabel("ang1")
				plt.show()

		means = torch.zeros(20, 3,device=self.dev,dtype=torch.float)
		stds = torch.zeros(20, 3, device=self.dev, dtype=torch.float)
		for res in lens.keys():
			means[res,0] = lens[res].mean()
			stds[res,0] =  lens[res].std()

			means[res,1] = ang1[res].mean()
			stds[res,1] =  ang1[res].std()

			means[res,2] = ang2[res].mean()
			stds[res,2] =  ang2[res].std()
		print(means)
		torch.save((means,stds),"marshalled/violations.m")


	def getWeights(self):
		
		return 
		
	def getNumParams(self):
		p=[]
		for i in self.parameters():
			p+= list(i.data.cpu().numpy().flat)
		print('Number of parameters=',len(p))
