import torch
from vitra.sources import fakeAtomsGeneration
from vitra.sources.globalVariables import *

atomsOfEachresi = {
    'LYS': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'ILE': ['C', 'N', 'O', 'CA', 'CB', 'CG1', 'CG2', 'CD1'],
    'THR': ['C', 'N', 'O', 'CA', 'CB', 'CG2', 'OG1'],
    'VAL': ['C', 'N', 'O', 'CA', 'CB', 'CG1', 'CG2'],
    'HIS': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD2', 'ND1', 'CE1', 'NE2'],
    'PHE': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', "RC"],
    'ASN': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'OD1', 'ND2'],
    'ARG': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASP': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'OD1', 'OD2'],
    'GLY': ['C', 'N', 'O', 'CA'],
    'GLU': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'LEU': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', ],
    'SER': ['C', 'N', 'O', 'CA', 'CB', 'OG'],
    'GLN': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'ALA': ['C', 'N', 'O', 'CA', 'CB'],
    'CYS': ['C', 'N', 'O', 'CA', 'CB', 'SG'],
    'MET': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'SD', 'CE'],
    'TYR': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', "RC"],
    'PRO': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD'],
    'TRP': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']}

partners_Residue_builder = {
    'LYS': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD": ["CA", "CB", "CG"], "CE": ["CB", "CG", "CD"],
            "NZ": ["CG", "CD", "CE"]},
    'ILE': {"CB": ["N", "C", "CA"], "CG1": ["N", "CA", "CB"], "CG2": ["N", "CA", "CB"], "CD1": ["CA", "CB", "CG1"]},
    'THR': {"CB": ["N", "C", "CA"], "OG1": ["N", "CA", "CB"], "CG2": ["N", "CA", "CB"]},
    'VAL': {"CB": ["N", "C", "CA"], "CG1": ["N", "CA", "CB"], "CG2": ["N", "CA", "CB"]},
    'HIS': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD2": ["CA", "CB", "CG"], "ND1": ["CB", "CD2", "CG"],
            "CE1": ["CG", "CD2", "ND1"], "NE2": ["CD2", "ND1", "CE1"]},
    'PHE': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD1": ["CA", "CB", "CG"], "CD2": ["CA", "CB", "CG"],
            "CE1": ["CB", "CG", "CD1"], "CE2": ["CB", "CG", "CD2"], "CZ": ["CD1", "CE1", "CE2"]},
    'ASN': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "OD1": ["CA", "CB", "CG"], "ND2": ["OD1", "CB", "CG"]},
    'ARG': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD": ["CA", "CB", "CG"], "NE": ["CB", "CG", "CD"],
            "CZ": ["CG", "CD", "NE"], "NH1": ["CD", "NE", "CZ"], "NH2": ["NH1", "NE", "CZ"]},
    'ASP': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "OD1": ["CA", "CB", "CG"], "OD2": ["OD1", "CB", "CG"]},
    'GLY': {},
    'GLU': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD": ["CA", "CB", "CG"], "OE1": ["CB", "CG", "CD"],
            "OE2": ["OE1", "CG", "CD"]},
    'LEU': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD1": ["CA", "CB", "CG"], "CD2": ["CD1", "CB", "CG"]},
    'SER': {"CB": ["N", "C", "CA"], "OG": ["N", "CA", "CB"]},
    'GLN': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD": ["CA", "CB", "CG"], "OE1": ["CB", "CG", "CD"],
            "NE2": ["CG", "OE1", "CD"]},
    'ALA': {"CB": ["N", "C", "CA"]},
    'CYS': {"CB": ["N", "C", "CA"], "SG": ["N", "CA", "CB"]},
    'MET': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "SD": ["CA", "CB", "CG"], "CE": ["CB", "CG", "SD"]},
    'TYR': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD1": ["CA", "CB", "CG"], "CD2": ["CA", "CB", "CG"],
            "CE1": ["CB", "CG", "CD1"], "CE2": ["CB", "CG", "CD2"], "CZ": ["CD1", "CE1", "CE2"],
            "OH": ["CE1", "CE2", "CZ"]},
    'PRO': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD": ["N", "CB", "CG"]},
    'TRP': {"CB": ["N", "C", "CA"], "CG": ["N", "CA", "CB"], "CD1": ["CA", "CB", "CG"], "CD2": ["CD1", "CB", "CG"],
            "NE1": ["CG", "CD2", "CD1"], "CE2": ["CD2", "CD1", "NE1"], "CE3": ["CD2", "CD1", "NE1"],
            "CZ2": ["NE1", "CD2", "CE2"], "CZ3": ["CZ2", "CD2", "CE3"], "CH2": ["CE2", "CZ3", "CZ2"]}
}

mutCoords = {'LEU': {'CB': [-0.7350809304657459, 1.2273079602053614, -0.4863102739675821],
                     'CG': [-0.651789587897512, -1.2946591219974053, -0.445164195629477],
                     'CD1': [-0.7712573949320182, 1.3158502362764573, -0.25766502249494566],
                     'CD2': [-0.7634272420355059, 1.1909673312347575, -0.4829474361906585]},
             'SER': {'CB': [-0.6421258869867366, 1.2958377334188453, -0.5006915368228511],
                     'OG': [-0.7413914864684564, 1.2270207932161132, -0.11953575172909497]},
             'ASP': {'CB': [-0.7660168144040225, 1.281724406010385, -0.37927711329733405],
                     'CG': [-0.8395130148066636, -1.19399117054604, -0.06995746067049673],
                     'OD1': [-1.0312567718396173, -0.6447257272024545, 0.10372994435889815],
                     'OD2': [-1.0363407961935769, -0.1654361770519688, -0.6978711116758792]}, 'GLY': {},
             'GLU': {'CB': [-0.8148236704315747, 1.1541593841209106, -0.5700010594133049],
                     'CG': [-0.6595220053523559, -1.3730758901298419, -0.23154456093204484],
                     'CD': [0.7508379473359894, 0.3373683578830564, -1.2844081325868943],
                     'OE1': [-1.1533086268652166, -0.3790090701551663, 0.021891427322019985],
                     'OE2': [-0.9971793612784874, -0.1384789469084774, -0.6755059417264059]},
             'TRP': {'CB': [-0.7831139426496087, 1.2179380333425818, -0.5368493966524149],
                     'CG': [-0.5455135558119978, -1.331133665858544, -0.3591257015385787],
                     'CD1': [0.016310146061668862, 0.9802478750634946, -0.9868510672437467],
                     'CD2': [-1.3657654312374061, 0.009958900465979007, -0.4095019207686951],
                     'NE1': [1.2894290957364538, 0.009278021149559862, -0.44987030619145796],
                     'CE2': [-0.8351395775919841, 0.004290028885965497, 1.0692612619593984],
                     'CE3': [-0.6154037430157433, -0.03861551508907257, 3.5005253866934276],
                     'CZ2': [-1.0174999432791727, -0.051416104809017905, -0.9254028526635805],
                     'CZ3': [-1.1749451384987586, -0.043911453582818075, 0.7348913665102393],
                     'CH2': [1.2009861737548537, -0.0025832620740607326, -0.6601411091827696]},
             'GLN': {'CB': [-0.8072673795094497, 1.1649810950339519, -0.5099564977787958],
                     'CG': [0.8274523133239093, -0.16088509365794657, -1.271495557695728],
                     'CD': [0.8421461890599025, 0.2003290645239893, -1.1927355841398635],
                     'OE1': [-0.9917677452804562, 0.7131494439235395, -0.14939030392247835],
                     'NE2': [-1.2188438828396326, -0.16320611846858457, -0.526007438898767]},
             'VAL': {'CB': [-0.7728228494186924, 1.2203972518639512, -0.5449888044207373],
                     'CG1': [0.8628266866321189, 0.25819979798965204, -1.233998756288952],
                     'CG2': [-0.7826310806834775, -1.2876681696849615, -0.15083936165174205]},
             'ASN': {'CB': [-0.7646418357740561, 1.2093541843771398, -0.48140399754536334],
                     'CG': [0.8256629417175534, -0.23938616039832017, -1.1711093608566303],
                     'OD1': [-0.5774607343816793, 1.0617566635442182, -0.33731374698660704],
                     'ND2': [-1.059267390328325, 0.0040381784832925205, -0.6656443704824853]},
             'LYS': {'CB': [-0.7380646285646306, 1.2065857569391252, -0.5541778168559555],
                     'CG': [-0.9779972596176155, -1.1384803460080133, -0.09128083734100961],
                     'CD': [0.7470433366815347, 0.129379043839264, -1.318382303471651],
                     'CE': [0.9144365903605765, -0.0332591533776454, -1.3019617250535795],
                     'NZ': [-1.2776993598177544, -0.7179338669549615, 0.008206666357165493]},
             'ALA': {'CB': [-0.7879403343985765, 1.1999333170260422, -0.4728432088773886]},
             'ILE': {'CB': [-0.8783380792459496, 1.0785701194312989, -0.6677496463443421],
                     'CG1': [0.8983529931386982, -0.4706468949270802, -1.1322996100651772],
                     'CG2': [-0.7526719183192936, 1.2194132957310586, -0.2683257115991906],
                     'CD1': [0.6150623648489464, 0.4905348084612614, -1.268516809289812]},
             'HIS': {'CB': [-0.8140485769311768, 1.193436219836514, -0.5458447791012728],
                     'CG': [-0.6647657650169375, -1.3138700328123747, -0.31241010663554114],
                     'ND1': [-1.1833980164662303, 0.054283810009490785, -0.6940458973650878],
                     'CD2': [-0.3283458489429044, -0.9750916003696201, -0.8871254562264601],
                     'CE1': [1.2650552525378762, -0.06191711744861739, -0.4263801284007007],
                     'NE2': [-0.813569931110347, -0.05669289879454906, 1.0561213530577016]},
             'ARG': {'CB': [-0.8299653133477412, 1.1324415281753366, -0.5694420102289125],
                     'CG': [0.7139423868484347, 0.04097536561222603, -1.3089391486214688],
                     'CD': [-0.6767606738671866, 1.32211151990453, -0.22420323446186466],
                     'NE': [0.8571970538616298, -0.04364014247656992, -1.1432754289372862],
                     'CZ': [0.5240459216987173, -0.07734355680372933, -1.2182559632175753],
                     'NH1': [-1.3351358350907048, 0.10157109984025187, -0.02474313893119756],
                     'NH2': [-1.0796197315309677, 0.016772664001532343, -0.6569555379661445]},
             'PHE': {'CB': [-0.7262863357842886, 1.278943875163726, -0.5715937445953165],
                     'CG': [-0.842270505229392, -1.222308390496352, -0.20213128973098504],
                     'CD1': [-1.3009931048811507, -0.5053443389670605, -0.058672595219793955],
                     'CD2': [0.5391784669554324, 0.43194763649004647, -1.1786086019339927],
                     'CE1': [0.6697297320462736, -0.06559555156140573, -1.1770850311851349],
                     'CE2': [0.6391605884306043, 0.04420868332400037, -1.2346729706822108],
                     'CZ': [1.1428196457505027, 0.027298131885064136, 0.6395521492471613]},
             'THR': {'CB': [-0.8095755224937321, 1.2136362582269835, -0.5312004879291918],
                     'OG1': [-0.8045880057842184, 1.1544719213472263, -0.04919280367370127],
                     'CG2': [-0.81091658202108, -1.261923090217897, -0.09243693737750414]},
             'PRO': {'CB': [-0.8660788760625349, 1.2257880546212925, -0.3932436185112925],
                     'CG': [-1.2671941349420763, -0.6337054809521327, 0.47356677883003],
                     'CD': [-0.7648207975330172, 0.5206941284373336, 1.2104568954392503]},
             'MET': {'CB': [-0.8144178547861365, 1.157470091501116, -0.5833566158497443],
                     'CG': [-0.8900932656400796, -1.170253551098543, -0.1830922294191836],
                     'SD': [1.1075323427278534, 0.18213054173972187, -1.4598129158908448],
                     'CE': [1.282912680827644, -0.38268434003360374, -1.1672027704188375]},
             'TYR': {'CB': [-0.7421095041886396, 1.1922842176027175, -0.5501381921924184],
                     'CG': [-0.7794171743094804, -1.3012190580753087, -0.3445314605394168],
                     'CD1': [-1.3212430003998898, 0.28860569960038596, -0.1121970764548001],
                     'CD2': [0.6331607810394178, -0.2723489644295496, -1.1725793959787447],
                     'CE1': [0.6030466704632003, 0.09800337929469971, -1.2405962352418456],
                     'CE2': [0.5772123831573972, -0.04205170434348718, -1.2537065555156028],
                     'CZ': [1.2068396487378295, -0.03754241802188323, 0.6558587212375758],
                     'OH': [-1.141964178217751, -0.08309797303348876, -0.7253838667501282]},
             'CYS': {'CB': [-0.6789102874290904, 1.2729563753915585, -0.5205202108388043],
                     'SG': [-1.0553886819132507, -1.4384824339843314, -0.1305622731760497]}}


def mutate(coords, atName, mutationList):
    """
    function that implement mutation in input proteins

    Parameters ---------- coords : torch.Tensor shape: (Batch, nAtoms, 3) coordinates of the proteins. It can be
    generated using the vitra.utils.parsePDB function atName : list shape: list of lists A list of the atom
    identifier. It encodes atom type, residue type, residue position and chain as an example GLN_39_N_B_0_0 refers to
    an atom N in a Glutamine, in position 39 of chain B. the last two zeros are used for the mutation engine This
    list can be generated using the vitra.utils.parsePDB function mutationList: list shape: list of lists a list of
    mutations. every entry corresponds to a protein (same order of atName and coordinates)\n every element of the
    list corresponds to a mutant. Every mutant can have 0, 1 or more mutations\n every mutation is encoded with
    resNumber_chain_MutattionToImplement\n the residue number and chain arethe same of the ones in the PDB file\n as
    an example, if we have 2 proteins, we want to generate two mutants of the first one only\n mutationList = [ [ [
    "10_A_GLY","10_B_GLY"], ["11_B_PHE"] ], [] ]\n in this case we will obtain, mutant 1 with a GLY at position 10 of
    both chain A and B and mutant 2 having a PHE in position 11 of chain B. No mutant are implemented for the second
    protein.

    Returns
    -------
    coordinates : torch.Tensor
        shape: (Batch, nAtoms, 3)
        updated coordinates of the atoms, which include the mutants now
    info_tensors : tuple
        updated set of precalculated information tensors required by the forcefield.
    """

    if len(mutationList) != len(coords):
        raise "mutation list have to have the same length of atNames"
    atNameNew = []
    coordsNew = []
    for p, prot in enumerate(mutationList):

        if len(prot) == 0:  # no mutations to implement:
            atNameNew += [atName[p]]
            coordsNew += [coords[p][coords[p, ..., 0] != PADDING_INDEX]]
            continue
        atNameProt = atName[p]
        maxmut = 0
        protdiz = {}
        for i in range(len(atNameProt)):
            r, resind, a, chainInd, mut, alt_conf = atNameProt[i].split("_")

            if mut != 0:

                if chainInd not in protdiz:
                    protdiz[chainInd] = {}
                if resind not in protdiz[chainInd]:
                    protdiz[chainInd][resind] = {}
                protdiz[chainInd][resind][a] = coords[p][i]

            else:
                maxmut = max(maxmut, mut)

        mutCoo = []
        mutAname = []

        for m, mutList in enumerate(prot):
            for mut in mutList:
                rnum, chain, to = mut.split("_")
                fromAtoms = protdiz[chain][rnum]

                bb_atoms = ["Nt", "N", "C", "O", "CA", "OXT"]
                keepWTAtoms = ["CB", "CG", "CD", "CE"]
                already_made_mut = []

                for i in list(fromAtoms.keys()):
                    if i in bb_atoms:
                        pass
                    elif i in keepWTAtoms and i in atomsOfEachresi[to]:
                        already_made_mut += [i]

                # changing atoms #
                for at in atomsOfEachresi[to]:

                    if at in already_made_mut or at in bb_atoms:
                        mutAname += [to + "_" + rnum + "_" + at + "_" + chain + "_" + str(m + 1) + "_0"]
                        mutCoo += [fromAtoms[at].unsqueeze(0)]
                    elif at == "RC":
                        newcoords = (fromAtoms["CD1"] + fromAtoms["CE2"]) / 2
                        mutCoo += [newcoords.unsqueeze(0)]
                        mutAname += [to + "_" + rnum + "_" + at + "_" + chain + "_" + str(m + 1) + "_0"]
                    else:
                        part1 = partners_Residue_builder[to][at][0]
                        part2 = partners_Residue_builder[to][at][1]
                        center = partners_Residue_builder[to][at][2]

                        r3 = fromAtoms[part2]
                        r2 = fromAtoms[part1]
                        r1 = fromAtoms[center]

                        atomPosition = torch.tensor(mutCoords[to][at]).type_as(r1)

                        newcoords = fakeAtomsGeneration.add_hydrogen(r1.unsqueeze(0), r2.unsqueeze(0), r3.unsqueeze(0),
                                                                     atomPosition.unsqueeze(0).unsqueeze(0))

                        # creating atom #
                        fromAtoms[at] = newcoords.squeeze()
                        mutAname += [to + "_" + rnum + "_" + at + "_" + chain + "_" + str(m + 1) + "_0"]
                        mutCoo += [fromAtoms[at].unsqueeze(0)]
                pass
        atNameNew += [atNameProt + mutAname]
        mutCoo = torch.cat(mutCoo, dim=0)
        coordsNew += [torch.cat([coords[p][coords[p, ..., 0] != PADDING_INDEX], mutCoo], dim=0)]
    return torch.nn.utils.rnn.pad_sequence(coordsNew, batch_first=True, padding_value=PADDING_INDEX), atNameNew
