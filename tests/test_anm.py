import anmld_python.anm as ANM

import numpy as np
import prody

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

TEST_PDB = "./tests/1ilr.pdb"

def test_build_hessian():
    prody_structure = prody.parsePDB(TEST_PDB)
    calphas = prody_structure.select('calpha')  # type: ignore
    prody_anm = prody.ANM('')
    prody_anm.buildHessian(calphas)

    # perform JIT
    coords_tmp = jnp.ones_like(calphas.getCoords())
    ANM.build_hessian(coords_tmp)

    hessian = ANM.build_hessian(calphas.getCoords())

    assert np.allclose(hessian, prody_anm.getHessian())  # type: ignore

def test_calc_modes():
    prody_structure = prody.parsePDB(TEST_PDB)
    calphas = prody_structure.select('calpha')  # type: ignore
    prody_anm = prody.ANM('')
    prody_anm.buildHessian(calphas)
    prody_anm.calcModes(3*len(calphas))

    hessian = ANM.build_hessian(calphas.getCoords())
    L, Q = ANM.calc_modes(hessian)

    # take the abs of the eigenvectors before comparing them as the phase of
    # the eigenvectors are arbitrary.
    assert np.allclose(np.abs(prody_anm.getEigvecs()), np.abs(Q))  # type: ignore
    assert np.allclose(prody_anm.getEigvals(), L)  # type: ignore
