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

In the `settings.toml` file, the `ambertools_prefix` and `pmemd_prefix` fields
of the `AMBER` section must be modified depending on your system to load AMBER.
The given command will be run before each of the AmberTools and PMEMD calls.

## Configuration

The default configuration can be found at the `settings.toml` file. While
running, the unspecified options will default to default values.

## Running

You can run ANMLD with:

```sh
anmld-python settings.toml initial.pdb target.pdb
```

You can check `anmld-python --help` for more details.

## Cite

If you use ANMLD, please cite:

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
```
