# ANM-LD: Predicting Conformational Transition Pathways with Intrinsic Dynamics

## Installation

> Note: It is recommended to install in a fresh Python environment

### Using OpenMM for Langevin dynamics

```sh
git clone https://github.com/ahmedselim2017/anmld_python
pip install ".[openmm]"
```

To install with CUDA12 support:
```sh
pip install ".[openmm-cuda]"
```

To install with HIP6 support:
```sh
pip install ".[openmm-hip]"
```

### Using AMBER for Langevin dynamics

```sh
git clone https://github.com/ahmedselim2017/anmld_python
pip install "."
```

To use AMBER, the `ambertools_prefix` and `pmemd_prefix` fields of the `AMBER`
section must be modified depending on your system to load AMBER. The given
command will be run before each of the AmberTools and PMEMD calls.

## Configuration

Configuration of `anmld-python` can be performed with a `toml` file.
Unspecified settings will use their default values. The default configuration
file that includes all of the settings that can be changed can be found at
`settings.toml`. 

## Running

You can run ANM-LD with:

```sh
anmld-python settings.toml initial.pdb target.pdb
```

You can check `anmld-python --help` for more details on how to start a run.

The final predicted structures are saved as `STEP_[STEP_NUMBER]_ANMLD.pdb`. The
intermediate structures, which have been deformed by ANM but have not yet
undergone LD, are saved as `STEP_[STEP_NUMBER]_ANM.pdb`.


## Citing ANM-LD

If you use ANM-LD, please cite:

```bibtex
@article{ersoy_computational_2023,
	title = {Computational analysis of long-range allosteric communications in {CFTR}},
	volume = {12},
	issn = {2050-084X},
	url = {https://elifesciences.org/articles/88659},
	doi = {10.7554/eLife.88659.3},
	abstract = {Malfunction of the {CFTR} protein results in cystic fibrosis, one of the most common hereditary diseases. {CFTR} functions as an anion channel, the gating of which is controlled by long-range allosteric communications. Allostery also has direct bearings on {CF} treatment: the most effective {CFTR} drugs modulate its activity allosterically. Herein, we integrated Gaussian network model, transfer entropy, and anisotropic normal mode-Langevin dynamics and investigated the allosteric communications network of {CFTR}. The results are in remarkable agreement with experimental observations and mutational analysis and provide extensive novel insight. We identified residues that serve as pivotal allosteric sources and transducers, many of which correspond to disease-causing mutations. We find that in the {ATP}-free form, dynamic fluctuations of the residues that comprise the {ATP}-binding sites facilitate the initial binding of the nucleotide. Subsequent binding of {ATP} then brings to the fore and focuses on dynamic fluctuations that were present in a latent and diffuse form in the absence of {ATP}. We demonstrate that drugs that potentiate {CFTR}’s conductance do so not by directly acting on the gating residues, but rather by mimicking the allosteric signal sent by the {ATP}-binding sites. We have also uncovered a previously undiscovered allosteric ‘hotspot’ located proximal to the docking site of the phosphorylated regulatory (R) domain, thereby establishing a molecular foundation for its phosphorylation-dependent excitatory role. This study unveils the molecular underpinnings of allosteric connectivity within {CFTR} and highlights a novel allosteric ‘hotspot’ that could serve as a promising target for the development of novel therapeutic interventions.},
	pages = {RP88659},
	journaltitle = {{eLife}},
	author = {Ersoy, Ayca and Altintel, Bengi and Livnat Levanon, Nurit and Ben-Tal, Nir and Haliloglu, Turkan and Lewinson, Oded},
	urldate = {2025-09-23},
	date = {2023-12-18},
	langid = {english},
}

@article{ACAR2020651,
title = {Distinct Allosteric Networks Underlie Mechanistic Speciation of ABC Transporters},
journal = {Structure},
volume = {28},
number = {6},
pages = {651-663.e5},
year = {2020},
issn = {0969-2126},
doi = {https://doi.org/10.1016/j.str.2020.03.014},
url = {https://www.sciencedirect.com/science/article/pii/S0969212620300964},
author = {Burçin Acar and Jessica Rose and Burcu {Aykac Fas} and Nir Ben-Tal and Oded Lewinson and Turkan Haliloglu},
keywords = {ABC transporter, membrane protein, molecular dynamics, ANM, ANM-LD, allostery, ATP hydrolysis, transport},
abstract = {Summary
ABC transporters couple the energy of ATP hydrolysis to the transmembrane transport of biomolecules. Here, we investigated the allosteric networks of three representative ABC transporters using a hybrid molecular simulations approach validated by experiments. Each of the three transporters uses a different allosteric network: in the constitutive B12 importer BtuCD, ATP binding is the main driver of allostery and docking/undocking of the substrate-binding protein (SBP) is the driven event. The allosteric signal originates at the cytoplasmic side of the membrane before propagating to the extracellular side. In the substrate-controlled maltose transporter, the SBP is the main driver of allostery, ATP binding is the driven event, and the allosteric signal propagates from the extracellular to the cytoplasmic side of the membrane. In the lipid flippase PglK, a cyclic crosstalk between ATP and substrate binding underlies allostery. These results demonstrate speciation of biological functions may arise from variations in allosteric connectivity.}
}
```

## Further Reading

```bibtex
@article{yang_single-molecule_2018,
	title = {Single-molecule probing of the conformational homogeneity of the {ABC} transporter {BtuCD}},
	volume = {14},
	issn = {1552-4450, 1552-4469},
	url = {https://www.nature.com/articles/s41589-018-0088-2},
	doi = {10.1038/s41589-018-0088-2},
	pages = {715--722},
	number = {7},
	journaltitle = {Nature Chemical Biology},
	shortjournal = {Nat Chem Biol},
	author = {Yang, Min and Livnat Levanon, Nurit and Acar, Burçin and Aykac Fas, Burcu and Masrati, Gal and Rose, Jessica and Ben-Tal, Nir and Haliloglu, Turkan and Zhao, Yongfang and Lewinson, Oded},
	urldate = {2025-09-23},
	date = {2018-07},
	langid = {english},
}
```
