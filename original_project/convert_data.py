import sys, os
sys.path.insert(0, '/Users/Mark/Documents/UW/Research/Development')

import pandas as pd
from Psience.Molecools import Molecule
from McUtils.Profilers import BlockProfiler

pp = pd.read_pickle('/Users/Mark/Desktop/test_contrastive_MOL_448017_89914_38659.pkl')


with BlockProfiler.profiler():
    mols = [ 
        Molecule.from_rdmol(mmm.GetConformer()).to_string('smi', remove_hydrogens=True, include_tag=True) 
        for mmm in pp['rdkit_mol_cistrans_stereo'][:50000]
    ]

# with open('/Users/Mark/Desktop/coley_contrastive_test_tags.smi', 'w+') as bbbbb:
#     bbbbb.write("\n".join(mols))