# TasVisAn

TasVisAn is a Python library for data reduction of neutron triple-axis spectrometers, especially for data collected on Sika and Taipan at Australian Centre for Neutron Scattering, Australian Nuclear Science and Technology. 

    * Structure factor calculation, including
        * Structure factors
        * Magnetic form factor import and calculation
    * Least-Squares fitting (custom interface for scipy.optimize.leastsq using lmfit features), including
        * Built-in physical models
    * Basic data operations
        * Data Browsing
        * Data Reduction
        * Data Combination
        * Data Normalization (monitor)
        * Calculated peak integrated intensity, position, and width
        * Data Visualization
        * Contour Mapping
        * Resolution Convolution Fitting
    * Loading from common TAS file types, including
        * SPICE TXT Files
        * HDF5
        * Extendable data import function
    * Mulit-analyser / multiplexing data import and manipulation:
        * Mulit-analyser / multiplexing Configuration
        * Mulit-analyser / multiplexing Data Reduction
        * Mulit-analyser / multiplexing Data Combination and Normalization
        * Mulit-analyser / multiplexing Data Visulization in 3D





## Installation

Use the package manager [pip](https://) to install TasVisAn.
Go to the folder of TasVisAn
```bash
pip install -e . 
```

## Usage

```python
import TasVisAn


```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
