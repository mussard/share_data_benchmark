import json
from pyscf import gto,scf,mcscf, fci, lo, ci, cc
from pyscf.scf import ROHF, UHF,ROKS
import numpy as np
import pandas as pd

# THIS IS WERE IT STARTS ====================================

df=json.load(open("../../../trail.json"))

spins={'Sc':1, 'Ti':2, 'V':3, 'Cr':6, 'Mn':5, 'Fe':4, 'Cu':1}

nd={'Sc':(1,0), 'Ti':(2,0), 'V':(3,0), 'Cr':(5,0), 'Mn':(5,0), 'Fe':(5,1), 'Cu':(5,5)}

cas={'Sc':3, 'Ti':4, 'V':5, 'Cr':6, 'Mn':7, 'Fe':8, 'Cu':11}

datacsv={}
for nm in ['atom','charge','method','basis','pseudopotential',
           'totalenergy','totalenergy-stocherr','totalenergy-syserr']:
  datacsv[nm]=[]

basis='vqz'
el='Fe'
charge=0

mol=gto.Mole()
mol.ecp={}
mol.basis={}
mol.ecp[el]=gto.basis.parse_ecp(df[el]['ecp'])
mol.basis[el]=gto.basis.parse(df[el][basis])
mol.charge=charge
if el == 'Cr' or el == 'Cu':
  mol.spin=spins[el]-charge
else:
  mol.spin=spins[el]+charge
mol.build(atom="%s 0. 0. 0."%el,verbose=4)

m=ROHF(mol)
m.level_shift=1000.0
dm=m.from_chk("../../../../HF/atoms/"+el+basis+str(charge)+".chk")
hf=m.kernel(dm)
m.analyze()

from pyscf.shciscf import shci
mc = shci.SHCISCF(m, 6, cas[el]-charge)
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

datacsv['atom'].append(el)
datacsv['charge'].append(charge)
datacsv['method'].append('MRPT')
datacsv['basis'].append(basis)
datacsv['pseudopotential'].append('trail')
datacsv['totalenergy'].append(cas+pt)
datacsv['totalenergy-stocherr'].append(0.0)
datacsv['totalenergy-syserr'].append(0.0)
pd.DataFrame(datacsv).to_csv(el+".csv",index=False)

