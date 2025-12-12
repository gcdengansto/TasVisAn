# Install Miniforge and common Python packages

This document provides step‑by‑step instructions for installing **Miniforge** (a lightweight conda distribution that defaults to `conda-forge`) and then creating an environment and installing common Python packages such as `numpy`, `scipy`, `matplotlib`, `pandas`, `plotly`, `pyside6`, and `jupyter` (Notebook). Commands are shown for **Windows (PowerShell)**, **macOS**, and **Linux**. Where platform differences exist the commands are grouped.

---

## 1. Overview and recommendations

- **Why Miniforge?** Miniforge is a minimal conda distribution that uses the `conda-forge` community channel by default. It is lightweight and widely used in research and reproducible workflows.
- **Use an environment** for projects to avoid dependency conflicts. We show how to create a dedicated environment (example name: `pyenv`).
- **Prefer `mamba`** for faster dependency solving. `mamba` is a drop-in replacement for `conda` and uses the same syntax.

---

## 2. Prerequisites

- Internet connection for downloads.
- On Windows: administrative rights are not required for user installs; prefer installing for ``Just Me`` unless you manage multiple accounts.
- On macOS (Apple Silicon) choose the `arm64` installer if you have an M1/M2; choose `x86_64` for Intel macs.

---

## 3. Download & install Miniforge

### 3.1 Windows (PowerShell)

1. Open PowerShell (recommended: **Run as user** — not necessary to run as Administrator for a per-user install).
2. Download the Miniforge installer (example uses `winget` if available) or download from the Miniforge GitHub releases page and run it.

**Using `winget` (Windows 10/11 with winget):**

```powershell
winget install --id=condaforge.Miniforge -e
```

**Manual (if winget not available):**

1. Visit the Miniforge releases page and download the `Miniforge3-Windows-x86_64.exe` (or `arm64` if applicable).
2. Run the `.exe` and follow the installer prompts. Recommended choices:
   - Install for `Just Me` (unless you have a reason to do system-wide).
   - Allow the installer to initialize `conda` by adding it to the PATH or use the installer default that adds a “conda init” step to your shell.

After install, close PowerShell and re-open a new PowerShell window (or run `conda init powershell` and restart shell).


### 3.2 macOS

1. Download the correct Miniforge installer from the Miniforge GitHub releases page. Choose `Miniforge3-MacOSX-arm64.sh` for Apple Silicon (M1/M2/M3) or `Miniforge3-MacOSX-x86_64.sh` for Intel.
2. Install from Terminal:

```bash
# Example for arm64 installer (adjust filename if different):
bash ~/Downloads/Miniforge3-MacOSX-arm64.sh

# follow prompts: accept license, and choose install location (default is ~/.miniforge)
```

3. Initialize shell (if installer did not already do it):

```bash
# for zsh (default on modern macOS)
~/.miniforge/bin/conda init zsh
# or for bash
~/.miniforge/bin/conda init bash
```

Then close and reopen your Terminal.


### 3.3 Linux

1. Download the appropriate `Miniforge3-Linux-*.sh` installer from the Miniforge releases page.
2. Install from a terminal:

```bash
# example (adjust filename):
bash ~/Downloads/Miniforge3-Linux-x86_64.sh

# follow the prompts; accept the license and choose install path (default ~/.miniforge)
```

3. Initialize your shell if needed:

```bash
~/.miniforge/bin/conda init bash
# or for zsh
~/.miniforge/bin/conda init zsh
```

Re-open the terminal once completed.

---

## 4. Verify installation

Open a new shell and run:

```bash
conda --version
# or
mamba --version   # if installed
```

If `conda` is not found, ensure the Miniforge `bin` directory is on your PATH, e.g. `~/.miniforge/bin` (macOS/Linux) or `C:\Users\<you>\Miniforge3\Scripts` on Windows (PowerShell should normally work after `conda init`).

---

## 5. Create and activate a conda environment

**Recommended:** create a fresh environment named `pyenv` with a specific Python version (example uses Python 3.11). Adjust the version as required.

```bash
# create environment (conda)
conda create -n pyenv python=3.11 -y

# or using mamba if you installed it
mamba create -n pyenv python=3.11 -y

# activate
conda activate pyenv
```

You should see the environment name in the prompt: `(pyenv)`.

---

## 6. Install packages

We will install packages from `conda-forge` where possible because `conda-forge` often has high-quality builds and handles compiled dependencies.

### 6.1 Add `conda-forge` channel (if not already default)

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

> Note: Miniforge defaults to `conda-forge`, but running these commands ensures the configuration in case of a different setup.

### 6.2 Fast install with `mamba` (recommended)

If you don't have `mamba`, install it first in the base environment or the environment where you prefer to solve packages:

```bash
# install mamba into base (so it can be used to create environments quickly)
conda activate base
conda install mamba -n base -c conda-forge -y
```

Then use `mamba` to install packages into `pyenv`:

```bash
conda activate pyenv
mamba install numpy scipy matplotlib pandas plotly pyside6 jupyter -c conda-forge -y
```

This will install all listed packages from `conda-forge`.

### 6.3 Using `conda` (if you prefer not to install mamba)

```bash
conda activate pyenv
conda install numpy scipy matplotlib pandas plotly pyside6 jupyter -c conda-forge -y
```

### 6.4 When to use `pip`

- Use `pip` only when a package is not available (or up-to-date) on `conda-forge`.
- Install packages with `pip` **after** installing conda packages in the active environment.

```bash
# example: install a package via pip within the active conda env
conda activate pyenv
pip install some_package_not_on_conda
```

**Caveat:** mixing `conda` and `pip` is acceptable but can sometimes create conflicts — prefer `conda-forge` when possible.

---

## 7. Install and run Jupyter Notebook

```bash
conda activate pyenv
# if jupyter was installed above you can run:
jupyter notebook

# or install if missing
conda install jupyter -c conda-forge -y
jupyter notebook
```

This opens a browser window with the Notebook interface. To stop the server, press `Ctrl+C` in the terminal.

If you prefer JupyterLab:

```bash
mamba install jupyterlab -c conda-forge -y
jupyter lab
```

---

## 8. Example `environment.yml` (reproducible environment)

Save the following as `environment.yml` and create the environment from it. Useful for sharing or reproducibility.

```yaml
name: pyenv
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - matplotlib
  - pandas
  - plotly
  - pyside6
  - jupyter
  - pip
  - pip:
    - some-package-only-on-pypi
```

Create from the file:

```bash
conda env create -f environment.yml
conda activate pyenv
```

---

## 9. Verifying installed packages

Within `pyenv` run:

```bash
python -c "import sys, numpy, scipy, matplotlib, pandas, plotly, PySide6; print('python', sys.version); print('numpy', numpy.__version__); print('scipy', scipy.__version__)"
```

For PySide6 you can test import:

```bash
python -c "import PySide6; print('PySide6 imported, version:', PySide6.__version__)"
```

---

## 10. Common troubleshooting

- **`conda` command not found:** ensure you restarted the shell after installation or manually add the Miniforge `bin`/`Scripts` path to `PATH`.
- **Solver very slow:** install `mamba` and use it instead of `conda`.
- **Conflicting packages:** try creating a fresh environment and avoid mixing many package channels besides `conda-forge`.
- **GUI apps not showing on macOS (PySide6):** for macOS, ensure you have the correct `python.app` integration; launching from Terminal with the `python` from the conda env normally works. In some cases, `pythonw` or `python -m PySide6 ...` may help.

---

## 11. Uninstalling Miniforge

If you need to remove Miniforge:

- Remove the installation directory (for default installs): `~/.miniforge` on macOS/Linux, or `C:\Users\<you>\Miniforge3` on Windows.
- Remove shell initialization lines that `conda init` added to your shell rc files (e.g., `~/.bashrc`, `~/.zshrc`, or PowerShell profile).

---

## 12. Quick command summary

```bash
# create env and install packages (quick):
conda create -n pyenv python=3.11 -y
conda activate pyenv
conda install -c conda-forge numpy scipy matplotlib pandas plotly pyside6 jupyter -y
# or fast with mamba:
mamba install -c conda-forge numpy scipy matplotlib pandas plotly pyside6 jupyter -y
```

---

## 13. Next steps and variants

- **GPU support:** if you need GPU-enabled packages (e.g., `tensorflow`/`pytorch` with CUDA), follow the respective project documentation — these are often installed from specialized channels or via `pip` and have platform/driver dependencies.
- **Lightweight alternative:** if you want a minimal install without conda, `miniconda` or `venv` + `pip` are alternatives; however, compiled scientific packages (NumPy/Scipy) may require building from source on some platforms.

---

If you want, I can also:
- Produce a minimal one‑page cheat sheet (short list of commands only).
- Produce a Windows PowerShell script or macOS/Linux shell script to automate installation steps.
- Produce an `environment.yml` tuned to your required versions.

