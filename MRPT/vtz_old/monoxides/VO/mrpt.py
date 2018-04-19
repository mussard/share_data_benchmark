import json
from pyscf import gto,scf,mcscf, fci, lo, ci, cc
from pyscf.scf import ROHF, UHF,ROKS
import numpy as np
import pandas as pd

# THIS IS WERE IT STARTS ====================================

df=json.load(open("../../../trail.json"))

spins={'ScO':1, 'TiO':2, 'VO':3, 'CrO':4, 'MnO':5, 'FeO':4, 'CuO':1}

nd={'Sc':(1,0), 'Ti':(2,0), 'V':(3,0), 'Cr':(5,0), 'Mn':(5,0), 'Fe':(5,1), 'Cu':(5,4)}

cas={'Sc':3, 'Ti':4, 'V':5, 'Cr':6, 'Mn':7, 'Fe':8, 'Cu':11}

re={'ScO':1.668, 'TiO':1.623, 'VO':1.591, 'CrO':1.621, 'MnO':1.648, 'FeO':1.616, 'CuO':1.725}

datacsv={}
for nm in ['basis','charge','method','molecule','pseudopotential',
           'totalenergy','totalenergy-stocherr','totalenergy-syserr']:
  datacsv[nm]=[]

basis='vtz'
element='V'

mol=gto.Mole()
mol.ecp={}
mol.basis={}
for el in [element,'O']:
  mol.ecp[el]=gto.basis.parse_ecp(df[el]['ecp'])
  mol.basis[el]=gto.basis.parse(df[el][basis])
mol.charge=0
mol.spin=spins[element+'O']
mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(element,re[element+'O']),verbose=4)

m=ROHF(mol)
m.level_shift=1000.0
dm=m.from_chk("../../../../HF/monoxides/"+element+basis+"0.chk")
hf=m.kernel(dm)
m.analyze()

from pyscf.shciscf import shci
mc = shci.SHCISCF(m, 9, 4+cas[element])
#mc.fcisolver.conv_tol = 1e-14
mc.fcisolver.mpiprefix="mpirun -np 28"
mc.fcisolver.num_thrds=12
mc.verbose = 4
cas=mc.kernel()[0]
 
from pyscf.icmpspt import icmpspt
pt=icmpspt.icmpspt(mc,rdmM=500, PTM=1000,\
                          pttype="MRLCC",\
                          third_order=True,\
                          fully_ic=True,\
                          do_dm4=True)

datacsv['basis'].append(basis)
datacsv['charge'].append(0)
datacsv['method'].append('MRPT')
datacsv['molecule'].append(element)
datacsv['pseudopotential'].append('trail')
datacsv['totalenergy'].append(cas+pt)
datacsv['totalenergy-stocherr'].append(0.0)
datacsv['totalenergy-syserr'].append(0.0)
pd.DataFrame(datacsv).to_csv(element+".csv",index=False)

