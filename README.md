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


TasVisAn is a Python library for data reduction of triple-axis spectrometers in general. Especially, it provides the customized classes to conduct data reduction, visualization and convolution fitting for the data collected from the following triple-axis spectrometers:
  * the thermal-neutron triple-axis spectrometer Taipan, ACNS, ANSTO
  * the cold-neutron triple-axis spectrometer Sika, ACNS, ANSTO
  * the thermal-neutron triple-axis spectrometer BT7, NCNR, NIST
  * the cold-neutron triple-axis spectrometer BT4, NCNR, NIST
More triple-axis spectrometers, including multiple-analyser triple-axis spectrometers, will be included in the future. 


## Installation

Use the package manager pip to install TasVisAn.
Go to the folder of TasVisAn
```bash
pip install -e . 


## Usage

```python
import TasVisAn

```

## Documentation
-------------
Please refer to the following article for detailed information about this package InsPy:
[TasVisAn and InsPy: Python packages for triple-axis spectrometer data visualization, analysis, instrument resolution calculation and convolution](https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576725008180)
by Guochu Deng* and Garry J. McIntyre, JOURNAL OF APPLIED CRYSTALLOGRAPHY Volume 58, Page 1-14,  2025
The DOI of this article is as follows:

[https://doi.org/10.1107/S1600576725008180](https://doi.org/10.1107/S1600576725008180)

Please find [the tutorial](https://github.com/gcdengansto/TasVisAn/blob/main/examples/TasVisAn_Demo.ipynb)  in [the examples folder](https://github.com/gcdengansto/TasVisAn/examples).

The video clips demonstrating how to run the GUIs for data fitting in this package can be found here:
[TASDataBrowser_demo] (https://doi.org/10.1107/S1600576725008180/te5154sup2.mp4)

## Contributing
---------------
Feature requests and bug reports can be made using the GitHub issues interface. 



## Copyright & Licensing
---------------------
Copyright (c) 2020-2025, Guochu Deng, Released under terms in [MIT](https://choosealicense.com/licenses/mit/) LICENSE.

## Disclaimer
----------
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


