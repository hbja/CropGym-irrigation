This repository contains an extended version of [_CropGym_](https://cropgym.ai); the code used in the paper "[_Adaptive fertilizer management for optimizing nitrogen use efficiency
with constrained reinforcement learning_](https://www.sciencedirect.com/science/article/pii/S016816992500660X)".

### How to install:

Requires python 3.9+

1. Clone the [PCSE](https://github.com/ajwdewit/pcse.git) repo and install
2. Clone this repo
3. Download [this](https://doi.org/10.5281/zenodo.15267400) file and put the `.csv` file under the folder `pcse_gym/utils/weather_utils/random_weather_csv/` (create the non-existing folders). It is the generated random weather used to train the RL agents. 
4. Install stable-baselines3, sb3contrib, scipy, lib_programname, rllte-core and tqdm with pip

### How to use:

Example to train a model using the NUE reward function with the LagrangianPPO agent and the E3B intrinsic reward:

`python train_winterwheat.py --reward NUE --environment 2 --agent LagPPO --seed 4 --nsteps 3000000 --random-weather --random-init --irs E3B`

### Citation

If you find our work useful, please consider citing our work.

```
@article{baja2025nue,
  title={Adaptive fertilizer management for optimizing nitrogen use efficiency with constrained reinforcement learning},
  author={Baja, Hilmy and Kallenberg, Michiel GJ and Berghuijs, Herman NC and Athanasiadis, Ioannis N},
  journal={Computers and Electronics in Agriculture},
  volume={237},
  pages={110554},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compag.2025.110554},
}
```

### Acknowledgements

This work was done as part of the EU Horizon project, [Smart Droplets](https://github.com/Smart-Droplets-Project).
