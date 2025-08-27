import anmld_python.models.anm as ANM

import numpy as np
import prody

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

TEST_PDB = "./tests/1ilr.pdb"

def test_buildHessian():
    prody_structure = prody.parsePDB(TEST_PDB)
    calphas = prody_structure.select('calpha')  # type: ignore
    prody_anm = prody.ANM('')
    prody_anm.buildHessian(calphas)

    hessian = ANM.build_hessian(calphas.getCoords())

    assert np.allclose(hessian, prody_anm.getHessian())  # type: ignore
