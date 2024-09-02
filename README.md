# Time-Varying RHGs

This repository contains the Python code that accompanies the master thesis:

> Erdin, Alexander “Stability of time-varying Receding Horizon Games”
> 2024.

## Prerequisites

- Python 3.9
- Casadi
- Conda (optional)

## Installation

1. Download and install [Python](https://www.python.org/downloads/)
2. Install Casadi by following the instructions from the official [Casadi](https://web.casadi.org/get/) documentation
3. Create a [Conda](https://docs.anaconda.com/miniconda/miniconda-install/) environment to make sure all the necessary packages are installed

    ```bash
        conda env create --name time-varying-rhgs --file=environments.yml
        conda activate time-varying-rhgs
    ```

   or install the packages manually.
4. Clone this repository or download the code as a ZIP archive and extract it to a folder of your choice.

## Running Jupyter Notebooks

Start a jupyter notebook server by running

```bash
    jupyter notebook 
```

This requires that you have installed all the packages from `environment.yml`.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite it:

```text
@article{erdin2024rhg,
  title={Stability of time-varying Receding Horizon Games},
  author={Erdin, Alexander},
  year={2024}
}
```
  
## Support and Contact

For any questions or issues related to this code, please contact the author:

- Alexander Erdin: aerdin(at)ethz(dot)ch

We appreciate any feedback, bug reports, or suggestions for improvements.