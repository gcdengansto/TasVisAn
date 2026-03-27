# TasVisAn

**TasVisAn** is a Python library for data reduction, visualization, and analysis of neutron triple-axis spectrometers (TASs), particularly for the thermal-neutron triple-axis spectrometer Taipan and cold-neutron triple-axis spectrometer Sika at Australian Centre for Neutron Scattering, ANSTO. It can also be used for other triple-axis spectrometers in other facilities, for example, the thermal-neutron triple-axis spectrometer BT7 and the cold-neutron triple-axis spectrometer BT4 at NIST. More triple-axis spectrometers will be supported soon.

> Install name: `tasvisan`  
> Import name: `tasvisan`

---

## 🚀 Installation

```bash
pip install tasvisan
```

## 📦 Usage
```bash
import tasvisan

# Example usage
# (replace with real functionality)
result = tasvisan.do_something()
print(result)
```

Run the GUI, Go to terminal:
```bash
#run python in terminal
>python
#in the prompt of python
>>import tasvisan
>>tasvisan.gui.TASDataBrowser.main()
>>
```
On linux like Ubuntu and Linuxmint, if you have warning like this:
```bash
QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-user'libGL error: glx: failed to create dri3 screenlibGL error: failed to load driver: virtio_gpu python
```
add the following lines in your .bashrc
```bash
# Ensure always render the GUI using software
export LIBGL_ALWAYS_SOFTWARE=1

# Ensure XDG_RUNTIME_DIR exists for third-party apps
if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR=/tmp/runtime-$USER
    if [ ! -d "$XDG_RUNTIME_DIR" ]; then
        mkdir -p "$XDG_RUNTIME_DIR"
        chmod 0700 "$XDG_RUNTIME_DIR"
    fi
fi
```
then run
```bash
source ~/.bashrc
```
Then, you can rerun the GUI interface in python again.


## 🎯 Features
Modular design for scientific workflows
Lightweight and easy to integrate into existing pipelines
Designed for extensibility in research environments
Compatible with NumPy-based data processing
Friendly GUI for Data Browsing


## 📁 Project Structure
```bash
tasvisan/
├── src/tasvisan/     # Main package (import tasvisan)
├── pyproject.toml    # Build configuration
├── README.md
└── LICENSE
```

## 🔬 Scope

This package is intended for:

> Triple-axis Spectrometer Data Reduction, Normalization, Visualization, and Analysis

> Quick Data Combination and Contour Mapping

> Resolution Convolution Fitting to Inelastic Neutron Scattering Data

> Experimental Planning and Command Validation and Simulation

> Data Reduction and 3D Visualization of Multiplexing and Multi-analyzer Triple-axis Spectrometers

## 🛠️ Development

Repository:

https://github.com/gcdengansto/TasVisAn


Install in editable mode:
```bash
pip install -e .
```

## 📖 Documentation

Documentation is under development.
Usage examples and tutorials can be found at https://github.com/gcdengansto/TasVisAn.

## 🤝 Contributing

Please contact the author for fixing bugs and adding additional functions


## 📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 👤 Author

Guochu Deng

Email: gc.deng.ansto@gmail.com

## ⚠️ Disclaimer

This software is provided for scientific research purposes.
No guarantees are made regarding correctness or fitness for a particular application.
