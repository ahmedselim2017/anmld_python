# Python implementation of the ANM-LD method


## TODO

- [x] Use `cwd` for ambmask instead of absolute paths (ambmask25 doesn't work
  with absolute path inputs ?)
- [x] Quote paths in config files and commands
- [ ] Add a flag to overwrite out dir
- [ ] Log steps information to CSV
- [ ] mmCIF support
- [ ] Early stopping
- [x] openMM support
    - [ ] first step
    - [x] min.in
        - [x] forcefield
        - [x] imin = 1
        - [x] maxcyc = {AS.min_step}
        - [-] ncyc = 50
        - [-] ntmin = 1
        - [-] drms = 0.01
        - [x] cut = 1000.0
        - [x] ntb = 0
        - [x] saltcon = 0.1
        - [x] igb = 1
    - [x] sim.in
        - [x] imin = 0,
        - [x] irest = 0, # Do not restart the simulation; instead, run as a new simulation.
        - [x] ntx = 1, # Coordinates, but no velocities, will be read;
        - [x] ntt = 3, # Use Langevin dynamics with the collision frequency γ given by gamma
        - [x] gamma_ln = 5.0, # The collision frequency γ, in ps−1 , when ntt = 3.
        - [-] ig = -1, # seed
        - [x] tempi = 310.0,
        - [x] temp0 = 310.0,
        - [x] nstlim = 100, # Number of MD-steps to be performed.
        - [x] dt = 0.002,
        - [x] ntc = 2, # = 2 bonds involving hydrogen are constrained
        - [-] ntf = 2, # = 2 bond interactions involving H-atoms omitted
        - [-] ntwr = 1, # Every ntwr steps during dynamics, the “restrt” file will be written, ensuring that recovery from a crash will not be so painful.
        - [-] ntpr = 1, # Every ntpr steps, energy information will be printed in human-readable form to files "mdout" and "mdinfo". "mdinfo"
        - [-] ntwx = 1, # Every ntwx steps, the coordinates will be written to the mdcrd file.
        - [-] ntwe = 1, # Every ntwe steps, the energies and temperatures will be written to file "mden" in a compact form.
        - [x] igb = 1,
        - [x] saltcon = 0.1,
        - [x] ntb = 0,
        - [x] cut = 1000.0,
