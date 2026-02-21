from pathlib import Path
import pandas as pd
import numpy as np
import linecache
import h5py
import re
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
import logging

from ..base import TasData
from ..utils.toolfunc import (
    AnglesToQhkl, strisfloat, strisint, ni_s2_residual, 
    ni_s2_residual_30p5meV, gaussian_residual, lorentzian_residual, 
    fit_peak, angle2, SqwDemo, PrefDemo, SelFormFactor, PrefPhononDemo,
    descend_obj, h5dump
)
from .validator import TaipanCommandValidator
import inspy as npy
from inspy import TripleAxisSpectr
from inspy.insfit import FitConv, UltraFastFitConv
from lmfit import Parameters, fit_report, minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Taipan(TasData):
    """
    Taipan neutron scattering instrument data handler.
    
    Extends TasData to provide Taipan-specific data loading, processing,
    and analysis capabilities.
    """
    
    # Constants
    PG002_D_SPACING = 3.355  # Angstroms
    DEFAULT_NORM_COUNT = 1000000
    VARIANCE_THRESHOLD = 0.0006
    
    def __init__(self, expnum: Union[int, str], title: str, sample: str, user: str):
        """
        Initialize Taipan instrument handler.
        
        Args:
            expnum: Experiment number
            title: Experiment title
            sample: Sample name
            user: User name
        """
        super().__init__("Taipan", expnum, title, sample, user)
        self.expnum = str(expnum)  # Ensure string type for file operations

    def _taipan_data_header_info(self, path: Path) -> Dict:
        """
        Parse neutron data file and extract header information.
        
        Args:
            path: Path to the data file
            
        Returns:
            Dictionary containing all extracted parameters
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        params = {}
        last_header_line = ""
        last_header_line_num = 0
        data_start_line = 0
        line_count = 0
        
        # Parameters that should be parsed as arrays
        array_params = {'latticeconstants', 'ubmatrix'}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    
                    if line.startswith('#'):
                        last_header_line = line
                        last_header_line_num = line_count
                        
                        # Skip column header lines
                        if '# Pt.' in line:
                            continue
                        
                        # Process parameter line
                        clean_line = line[1:].strip()
                        
                        if not clean_line or '=' not in clean_line:
                            continue
                        
                        # Split by first '=' only
                        key, value = clean_line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle array parameters
                        if key in array_params:
                            try:
                                value = [float(item) for item in value.split()]
                            except ValueError as e:
                                logger.warning(f"Failed to parse array parameter '{key}': {e}")
                                continue
                        
                        params[key] = value
                    else:
                        data_start_line = line_count
                        break
        
        except Exception as e:
            raise ValueError(f"Error parsing header from {path}: {e}")
        
        # Add file structure metadata
        params['_data_start_line'] = data_start_line
        params['_header_line'] = last_header_line
        params['_header_line_num'] = last_header_line_num
        
        # Rename parameters to standard names (with safe get)
        params['expno'] = params.pop('experiment_number', self.expnum)
        params['scanno'] = params.pop('scan', None)
        params['lattice'] = params.pop('latticeconstants', None)
        params['scanax1'] = params.get('def_x', '')
        
        if not params['scanax1']:
            logger.warning("Scan axis is empty in header")
        
        return params

    def _make_unique_column_names(self, column_names: List[str]) -> List[str]:
        """
        Make column names unique by appending numbers to duplicates.
        
        Args:
            column_names: List of potentially duplicate column names
            
        Returns:
            List of unique column names
        """
        seen = {}
        unique_names = []
        
        for name in column_names:
            if name not in seen:
                seen[name] = 0
                unique_names.append(name)
            else:
                seen[name] += 1
                unique_names.append(f"{name}_dup{seen[name]}")
        
        return unique_names

    def taipan_data_to_pd(self, path='', scanno=None):
        """
        Load a single neutron data file into a pandas DataFrame with metadata in attrs.
        Args:
            path (str): Path to the data file
        Returns:
            pandas.DataFrame: DataFrame containing the data with metadata in attrs
        """
        ppath = Path(path)
        
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if scanno is None:
            raise ValueError("Scan number cannot be None")
        
        fileinitial = "TAIPAN_exp" + str(self.expnum) + "_scan"

        filefullname = fileinitial + str(scanno) + ".dat"
        
        #for fileindex, filename in enumerate (scanlist):
        datastart = 0
        datalines = 0       
     
        fullfilepath  = ppath / Path(filefullname) 
                
        # Parse header information
        params = self._taipan_data_header_info(fullfilepath)
        #print(params)
        # Get file structure information
        data_start_line = params.pop('_data_start_line', 0)
        header_line     = params.pop('_header_line', '')
        header_line_num = params.pop('_header_line_num', 0)
        # Process the header line to extract column names
        if header_line.startswith('#'):
            # Remove the leading # and split by whitespace
            header_line = header_line[1:].strip()
        # Create column names from the header line
        if header_line:
            # Clean and split the header line
            taipan_column_names = [col.strip() for col in header_line.split() if col.strip()]
        # Try to read the data part using pandas
        try:
            # No header line found, use pandas to read with default column names
            df = pd.read_csv(fullfilepath, sep=r'\s+', header=None, comment='#')

            seen={}
            unique_taipan_column_names =[]

            for name in taipan_column_names:
                if name not in seen:
                    seen[name] = 1
                    unique_taipan_column_names.append(name)
                else:
                    seen[name] += 1
                    unique_taipan_column_names.append(f"{name}_dup")
            #print(seen)
            df.columns = unique_taipan_column_names

            if 'qh' in df.columns and 'h' in df.columns:
                df.drop(columns ='qh', inplace=True)
                unique_taipan_column_names.remove('qh')
            if 'qk' in df.columns and 'k' in df.columns:
                df.drop(columns ='qk', inplace=True)
                unique_taipan_column_names.remove('qk')
            if 'ql' in df.columns and 'l' in df.columns:
                df.drop(columns ='ql', inplace=True)
                unique_taipan_column_names.remove('ql')
            if 'en' in df.columns and 'e' in df.columns:
                df.drop(columns ='en', inplace=True)
                unique_taipan_column_names.remove('en')
            
            taipan_motorname_list = ['h',  'k',   'l','e', 'T1_Sensor1', 'T1_Sensor2','T1_Sensor3',]
            stdtas_motorname_list = ['qh', 'qk', 'ql','en', 'tempVTI', 'tempSAMP','tempSAMPh']

            # Create a mapping from old to new
            replace_map = dict(zip(taipan_motorname_list, stdtas_motorname_list))

            # change into standard column names
            #print(df.columns)
            #print(unique_taipan_column_names)
            std_column_names = [replace_map.get(item, item) for item in unique_taipan_column_names]
            df.columns = std_column_names

        except Exception as e:
            print(f"Warning: Error parsing data section. {e}")
            print("A empty DataFrame is returned!")
            # Create an empty DataFrame as last resort
            df = pd.DataFrame()
    
        # Store all metadata in the DataFrame's attrs
        df.attrs = params
        
        return df  

    def taipan_hdf_to_pd(self, path: Union[str, Path], scanno: Union[int, str]) -> pd.DataFrame:
        """
        Load HDF5 neutron data file into pandas DataFrame.
        
        Args:
            path: Directory path containing HDF files
            scanno: Scan number (can be string or int)
            
        Returns:
            DataFrame with data and metadata in attrs
        """
        ppath = Path(path)
        
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        # Format filename
        scanno_str = str(scanno)
        if len(scanno_str) == 5:
            filename = f"00{scanno_str}"
        elif len(scanno_str) == 6:
            filename = f"0{scanno_str}"
        elif len(scanno_str) >= 7:
            filename = scanno_str
        else:
            raise ValueError(f"Invalid scan number format: {scanno}")
        
        filepath = ppath / f"TPN{filename}.nx.hdf"
        
        if not filepath.exists():
            raise FileNotFoundError(f"HDF file not found: {filepath}")
        
        try:
            namelist, col_list = h5dump(filepath)
            
            df = pd.DataFrame()
            scan_axis = None
            title = 'no title'
            
            with h5py.File(str(filepath), 'r') as file:
                # Get title
                title_dataset = file.get("/entry1/experiment/title")
                if title_dataset is not None:
                    title = title_dataset[0]
                
                # Load data columns
                for index, name in enumerate(namelist):
                    if "data" in name:
                        data = np.array(file[name])
                        # Detect scan axis by checking variance
                        if len(data) > 1:
                            variance = np.abs(data[0] - data[-1]) / len(data)
                            if variance > self.VARIANCE_THRESHOLD:
                                scan_axis = col_list[index]
                    else:
                        new_col = pd.DataFrame(file[name])
                        new_col.columns = [col_list[index]]
                        df = pd.merge(df, new_col, how='outer', 
                                    left_index=True, right_index=True)
            
            # Reorder columns to put scan axis first
            if scan_axis and scan_axis in df.columns:
                cols = df.columns.tolist()
                cols.remove(scan_axis)
                cols.insert(0, scan_axis)
                df = df[cols]
            
            # Rename columns to standard names
            rename_map = {
                'bm1_counts': 'monitor',
                'bm2_counts': 'detector',
                'VS_left': 'vs_left',
                'VS_right': 'vs_right',
                'sensorValueA': 'tempVTI',
                'sensorValueB': 'tempSAMP',
                'sensorValueC': 'tempSAMPh',
                'sensorValueD': 'tempIdle',
                'setpoint1': 'tempSP1',
                'setpoint2': 'tempSP2'
            }
            
            df.rename(columns={k: v for k, v in rename_map.items() 
                             if k in df.columns}, inplace=True)
            
            # Store metadata
            df.attrs = {
                "scanno": scanno,
                "scanax": scan_axis,
                "title": title
            }
            
            logger.info(f"Loaded HDF scan {scanno}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading HDF file {filepath}: {e}")
            return pd.DataFrame()

    def taipan_scanlist_to_dflist(self, path: Union[str, Path], 
                                   scanlist: List[int]) -> List[pd.DataFrame]:
        """
        Convert list of scan numbers to list of DataFrames.
        
        Args:
            path: Directory path containing data files
            scanlist: List of scan numbers (integers only)
            
        Returns:
            List of DataFrames
        """
        if not all(isinstance(x, (int, np.integer)) for x in scanlist):
            raise ValueError("All scan numbers must be integers")
        
        dflist = []
        for scanno in scanlist:
            try:
                df = self.taipan_data_to_pd(path, scanno)
                if not df.empty:
                    dflist.append(df)
                else:
                    logger.warning(f"Empty DataFrame for scan {scanno}")
            except Exception as e:
                logger.error(f"Error loading scan {scanno}: {e}")
        
        logger.info(f"Loaded {len(dflist)} scans successfully")
        return dflist

    def taipan_combplot(self, path: Union[str, Path], scanlist: List, 
                        fit: bool = False, norm_mon_count: int = None,
                        overplot: bool = False, offset: int = 1000, 
                        initial: Optional[Dict] = None):
        """
        Combine and plot multiple scans.
        
        Args:
            path: Directory path containing data files
            scanlist: List of scan numbers (can include sublists for combining)
            fit: Whether to fit peaks
            norm_mon_count: Monitor count for normalization
            overplot: Whether to overlay plots
            offset: Vertical offset between plots
            initial: Initial fit parameters
        """
        if norm_mon_count is None:
            norm_mon_count = self.DEFAULT_NORM_COUNT
        
        dflist = []
        for scanno in scanlist:
            try:
                if isinstance(scanno, list):
                    # Combine sub-scans
                    subdflist = self.taipan_scanlist_to_dflist(path, scanno)
                    if subdflist:
                        comb_df = super().tas_datacombine(subdflist)
                        dflist.append(comb_df)
                elif isinstance(scanno, (int, np.integer)):
                    single_df = self.taipan_data_to_pd(path, scanno)
                    if not single_df.empty:
                        dflist.append(single_df)
            except Exception as e:
                logger.error(f"Error processing scan {scanno}: {e}")
        
        if dflist:
            super().tas_combplot(dflist, fit, norm_mon_count, 
                               overplot, offset, initial)
        else:
            logger.warning("No data to plot")

    def taipan_batch_reduction(self, path: Union[str, Path], 
                               scanlist: List, 
                               motorlist: List[str] = None) -> List[pd.DataFrame]:
        """
        Batch reduce multiple scans to specified columns.
        
        Args:
            path: Directory path containing data files
            scanlist: List of scan numbers or sublists
            motorlist: List of column names to keep
            
        Returns:
            List of reduced DataFrames
        """
        if motorlist is None:
            motorlist = ['qh', 'qk', 'ql', 'en', 'ei', 'ef', 'm1', 'm2',
                        's1', 's2', 'a1', 'a2', 'detector', 'monitor',
                        'tempVTI', 'tempSAMP']
        
        dflist = []
        
        for scanno in scanlist:
            try:
                if isinstance(scanno, (int, np.integer)):
                    df = self.taipan_data_to_pd(path, scanno)
                elif isinstance(scanno, list):
                    temp_dflist = self.taipan_scanlist_to_dflist(path, scanno)
                    df = super().tas_datacombine(temp_dflist)
                else:
                    logger.warning(f"Invalid scan identifier: {scanno}")
                    continue
                
                if df.empty:
                    continue
                
                # Filter to existing columns only
                existing_cols = [col for col in motorlist if col in df.columns]
                missing_cols = [col for col in motorlist if col not in df.columns]
                
                if missing_cols:
                    logger.debug(f"Scan {scanno} missing columns: {missing_cols}")
                
                if existing_cols:
                    dflist.append(df[existing_cols])
                    
            except Exception as e:
                logger.error(f"Error reducing scan {scanno}: {e}")
        
        return dflist

    def taipan_reduction_by_row(self, path: Union[str, Path], 
                               scanlist: List,
                               motorlist: List[str] = None, 
                               sortby: str = "tempSAMP") -> pd.DataFrame:
        """
        Reduce and combine multiple scans row-wise.
        
        Args:
            path: Directory path containing data files
            scanlist: List of scan numbers
            motorlist: List of column names to keep
            sortby: Column name to sort by
            
        Returns:
            Combined and sorted DataFrame
        """
        if motorlist is None:
            motorlist = ['qh', 'qk', 'ql', 'en', 'ei', 'ef', 'm1', 'm2',
                        's1', 's2', 'a1', 'a2', 'detector', 'monitor',
                        'tempVTI', 'tempSAMP']
        
        dflist = self.taipan_batch_reduction(path, scanlist, motorlist)
        df_combined = pd.DataFrame(columns=motorlist)
        
        for df in dflist:
            #if df is not None and not df.empty:
            if df is not None and not df.empty and not df.isna().all().all():
                df_combined = pd.concat([df_combined, df], ignore_index=True)
        
        if sortby in df_combined.columns:
            df_combined = df_combined.sort_values(
                by=sortby, ascending=True
            ).reset_index(drop=True)
        
        return df_combined

    def taipan_export_hklw(self, path: Union[str, Path], 
                          scanlist: List,
                          hklw_file: str = "") -> List[pd.DataFrame]:
        """
        Export scan data in HKLW format for external analysis.
        
        Args:
            path: Directory path containing data files
            scanlist: List of scan numbers (can include sublists)
            hklw_file: Output filename
            
        Returns:
            List of DataFrames with HKLW data
        """
        hklw_df = self.taipan_batch_reduction(
            path, scanlist, 
            motorlist=['qh', 'qk', 'ql', 'en', 'detector']
        )
        
        super().tas_export_hklw(hklw_df, hklw_file)
        return hklw_df

    def export_scanlist(self, path: Union[str, Path], 
                       logfile: str,
                       outputfile: Optional[str] = None) -> pd.DataFrame:
        """
        Export comprehensive scan list from log file.
        
        Args:
            path: Directory path
            logfile: Log file name
            outputfile: Output HTML file name
            
        Returns:
            DataFrame with scan information
        """
        ppath = Path(path)
        
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if outputfile is None:
            outputfile = f"TAIPAN_exp{self.expnum}_scanlist.html"
        
        scanlist = self.export_scanlog(path=path, logfile=logfile, 
                                       outputfile=outputfile)
        
        if scanlist is None or scanlist.empty:
            logger.error("No information from log file")
            return pd.DataFrame(columns=['scan_no', 'command', 'scantitle'])
        
        scanno_list = []
        command_list = []
        scantitle_list = []
        
        for hdfname in scanlist['scanhdf_name']:
            try:
                scan_num = int(hdfname[-7:])
                scanno_list.append(scan_num)
                
                filename = f"TAIPAN_exp{self.expnum}_scan{scan_num}.dat"
                datafile_path = ppath / "Datafiles" / filename
                
                if datafile_path.is_file():
                    with datafile_path.open('r', encoding='utf-8') as f:
                        for line in f:
                            if "# command =" in line:
                                command_list.append(line[11:].strip())
                            if "# scan_title =" in line:
                                scantitle_list.append(line[14:].strip())
                else:
                    logger.warning(f"Data file not found: {filename}")
                    command_list.append("no-file")
                    scantitle_list.append("no-file")
                    
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing HDF name '{hdfname}': {e}")
                continue
        
        scan_dict = {
            "scan_no": scanno_list,
            "command_dat": command_list,
            "scantitle": scantitle_list
        }
        scantitlelist = pd.DataFrame(scan_dict)
        
        fulllist = pd.merge(scanlist, scantitlelist, how='outer',
                          left_index=True, right_index=True)
        fulllist = fulllist[['scan_no', 'command', 'scantitle']]
        
        output_path = ppath / outputfile
        fulllist.to_html(output_path)
        
        return fulllist

    def export_scanlog(self, path: Union[str, Path], 
                      logfile: str,
                      outputfile: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Extract scan information from log file.
        
        Args:
            path: Directory path
            logfile: Log file name
            outputfile: Output HTML file name
            
        Returns:
            DataFrame with scan information or None on error
        """
        ppath = Path(path)
        
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if outputfile is None:
            outputfile = f"TAIPAN_exp{self.expnum}_scanlist.html"
        
        logfile_path = ppath / logfile
        
        if not logfile_path.exists():
            logger.error(f"Log file not found: {logfile_path}")
            return None
        
        try:
            hdf_filelist = []
            command_list = []
            
            # Count total lines
            with logfile_path.open('r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            cur_line = total_lines
            
            while cur_line > 0:
                cur_line -= 1
                line = linecache.getline(str(logfile_path), cur_line)
                
                if "nx.hdf updated" in line:
                    hdf_filelist.append(line[22:32])
                    
                    # Search backwards for scan command
                    scan_found = False
                    search_line = cur_line
                    
                    while search_line > 0 and not scan_found:
                        search_line -= 1
                        check_line = linecache.getline(str(logfile_path), search_line)
                        
                        if "Scanvar:" in check_line:
                            # Check previous lines for command
                            prev_lines = [
                                linecache.getline(str(logfile_path), search_line - i)
                                for i in range(1, 4)
                            ]
                            
                            # Look for runscan or mscan command
                            for prev_line in prev_lines:
                                if "runscan" in prev_line or "mscan" in prev_line:
                                    command_list.append(prev_line[22:].strip())
                                    scan_found = True
                                    break
                            
                            if not scan_found:
                                # Combine Scanvar lines
                                combined = check_line[22:76]
                                for prev_line in prev_lines:
                                    if "Scanvar:" in prev_line:
                                        combined += prev_line[22:76]
                                command_list.append(combined.strip())
                                scan_found = True
            
            # Clear linecache to free memory
            linecache.clearcache()
            
            if not hdf_filelist:
                logger.warning("No HDF files found in log")
                return None
            
            scan_dict = {
                "scanhdf_name": hdf_filelist[::-1],  # Reverse order
                "command": command_list[::-1]
            }
            scanlist = pd.DataFrame(scan_dict)
            
            output_path = ppath / outputfile
            scanlist.to_html(output_path)
            
            return scanlist
            
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
            return None

    def taipan_calibr_6scans(self, path: Union[str, Path], 
                            scanlist: List[int]) -> Tuple:
        """
        Calibrate analyzer using 6 nickel scans.
        
        Args:
            path: Directory path containing data files
            scanlist: List of 6 scan numbers
            
        Returns:
            Tuple of (params_df, fitted_data, calibr_result, fig, ax)
        """
        if len(scanlist) != 6:
            logger.warning(f"Expected 6 scans, got {len(scanlist)}")
        
        niscan_dflist = self.taipan_batch_reduction(path, scanlist)
        
        params_df = pd.DataFrame(columns=[
            'A', 'A_err', 'w', 'w_err', 'x0', 'x0_err', 'bg', 'bg_err'
        ])
        fitted_data = pd.DataFrame()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for idx, df in enumerate(niscan_dflist):
            if df.empty:
                logger.warning(f"Scan {idx} is empty, skipping")
                continue
            
            try:
                col_x = df.attrs.get('scanax1', df.columns[0])
                col_y = "detector"
                
                dataX = df[col_x].dropna().to_numpy()
                dataY = df[col_y].dropna().to_numpy()
                
                # Ensure matching lengths
                min_len = min(len(dataX), len(dataY))
                dataX = dataX[:min_len]
                dataY = dataY[:min_len]
                
                cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G')
                
                ax.plot(dataX, dataY, 'o', label=f'Scan {idx}')
                ax.plot(cur_fitdat['X'], cur_fitdat['Y_fit'], '-')
                
                params_df = pd.concat([params_df, cur_fitpar], 
                                     axis=0, ignore_index=True)
                fitted_data = pd.merge(fitted_data, cur_fitdat, how='outer',
                                     left_index=True, right_index=True,
                                     suffixes=("", f"_{idx}"))
                
            except Exception as e:
                logger.error(f"Error fitting scan {idx}: {e}")
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sort by peak position (descending)
        params_df = params_df.sort_values(by='x0', ascending=False)
        ni_peakpos = params_df['x0'].to_numpy()
        
        calibr_result = self.calibr_fit_offset(ni_peakpos)
        
        return params_df, fitted_data, calibr_result, fig, ax

    def calibr_fit_offset(self, peaks: np.ndarray) -> Dict:
        """
        Fit analyzer offset from nickel peak positions.
        
        Args:
            peaks: Array of 6 peak positions
            
        Returns:
            Dictionary with calibration results
            
        Raises:
            ValueError: If peaks array is invalid
        """
        if peaks is None or len(peaks) != 6:
            raise ValueError("Exactly 6 peaks required for calibration")
        
        dataX = np.array([1, 2, 3, 4, 5, 6])
        dataY = peaks.flatten()
        
        # Set up fit parameters
        fit_params = Parameters()
        fit_params.add('s2_offset', value=0.01)
        fit_params.add('wavelen', value=2.345)
        
        # Perform fit
        try:
            result = minimize(ni_s2_residual, fit_params, 
                            args=(dataX, dataY), 
                            method='Levenberg-Marquardt')
            
            # Calculate m1 from wavelength
            wavelen = result.params['wavelen'].value
            m1 = np.arcsin(wavelen / (2.0 * self.PG002_D_SPACING)) * 180 / np.pi
            s2_offset = result.params['s2_offset'].value
            
            calibr_result = {
                "m1": m1,
                "m2": 2 * m1,
                "s2_offset": s2_offset,
                "wavelen": wavelen,
                "fit_success": result.success,
                "reduced_chi_sq": result.redchi if hasattr(result, 'redchi') else None
            }
            
            logger.info("\nCalibration Results:")
            logger.info(f"  m1: {m1:.8f}")
            logger.info(f"  m2: {2*m1:.8f}")
            logger.info(f"  s2 offset: {s2_offset:.8f}")
            logger.info(f"  wavelength: {wavelen:.8f}")
            
            return calibr_result
            
        except Exception as e:
            logger.error(f"Calibration fit failed: {e}")
            raise

    def export_scantitle(self, path: Union[str, Path], 
                        datafrom_to: Tuple[int, int],
                        outputfile: Optional[str] = None) -> pd.DataFrame:
        """
        Export scan titles for a range of scan numbers.
        
        Args:
            path: Directory path containing data files
            datafrom_to: Tuple of (first_scan, last_scan)
            outputfile: Output HTML file name
            
        Returns:
            DataFrame with scan information
        """
        ppath = Path(path)
        
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if datafrom_to is None or len(datafrom_to) != 2:
            raise ValueError("datafrom_to must be tuple of (first, last)")
        
        if outputfile is None:
            outputfile = f"TAIPAN_exp{self.expnum}_scanlist.html"
        
        first_scan = int(datafrom_to[0])
        last_scan = int(datafrom_to[1])
        
        scan_numbers = range(first_scan, last_scan + 1)
        
        scanno_list = []
        command_list = []
        scantitle_list = []
        
        for scan_num in scan_numbers:
            filename = f"TAIPAN_exp{self.expnum}_scan{scan_num}.dat"
            filepath = ppath / filename
            
            scanno_list.append(scan_num)
            
            if not filepath.exists():
                logger.warning(f"File not found: {filename}")
                command_list.append("file-not-found")
                scantitle_list.append("file-not-found")
                continue
            
            try:
                with filepath.open('r', encoding='utf-8') as f:
                    found_command = False
                    found_title = False
                    
                    for line in f:
                        if not line.startswith('#'):
                            break  # Stop at data section
                        
                        if "# command =" in line and not found_command:
                            command_list.append(line[11:].strip())
                            found_command = True
                        
                        if "# scan_title =" in line and not found_title:
                            scantitle_list.append(line[14:].strip())
                            found_title = True
                        
                        if found_command and found_title:
                            break
                    
                    if not found_command:
                        command_list.append("not-found")
                    if not found_title:
                        scantitle_list.append("not-found")
                        
            except IOError as e:
                logger.error(f"Error reading file {filename}: {e}")
                command_list.append("error")
                scantitle_list.append("error")
        
        scan_dict = {
            "scanno": scanno_list,
            "command": command_list,
            "scantitle": scantitle_list
        }
        
        scanlist = pd.DataFrame(scan_dict)
        
        output_path = ppath / outputfile
        scanlist.to_html(output_path)
        
        logger.info(f"Exported {len(scanlist)} scan titles to {output_path}")
        
        return scanlist
        

    def taipan_expconfig(self, ef: float = 14.87, aa: float = 4, bb: float = 5, 
                        cc: float = 6, ang_a: float = 90, ang_b: float = 90, 
                        ang_c: float = 90, uu: np.ndarray = np.array([1, 0, 0]), 
                        vv: np.ndarray = np.array([0, 0, 1])):
        """
        Configure Taipan experiment parameters.
        
        Args:
            ef: Final energy in meV
            aa, bb, cc: Lattice parameters in Angstroms
            ang_a, ang_b, ang_c: Unit cell angles in degrees
            uu: First orientation vector
            vv: Second orientation vector
            
        Returns:
            Configured experiment object
        """
        taipan_exp = super().tas_expconfig(ef)
        
        # Instrument configuration
        taipan_exp.method = 1  # 1 for Popovici, 0 for Cooper-Nathans
        taipan_exp.moncor = 1
        taipan_exp.efixed = ef
        taipan_exp.infin = -1  # const-Ef mode
        
        # Motor directions (fixed duplicate assignment bug)
        taipan_exp.mono.dir = 1  # Removed duplicate assignment
        taipan_exp.ana.dir = -1
        
        # Sample configuration
        taipan_exp.sample.a = aa
        taipan_exp.sample.b = bb
        taipan_exp.sample.c = cc
        taipan_exp.sample.alpha = ang_a
        taipan_exp.sample.beta = ang_b
        taipan_exp.sample.gamma = ang_c
        taipan_exp.sample.u = uu
        taipan_exp.sample.v = vv
        
        # Instrument geometry
        taipan_exp.arms = [202, 196, 179, 65, 106]
        taipan_exp.orient1 = uu
        taipan_exp.orient2 = vv
        
        self.exp = taipan_exp
        
        return taipan_exp

    def taipan_conv_init(self, hklw: pd.DataFrame = None, exp = None, 
                        initial: Dict = None, fixlist: List[int] = None, 
                        magion: str = "Mn2", sqw = SqwDemo, pref = PrefDemo, 
                        smoothfit: bool = True):
        """
        Initialize convolution fitting.
        
        Args:
            hklw: DataFrame with columns [H, K, L, W, Iobs]
            exp: Experiment configuration
            initial: Initial parameters dictionary
            fixlist: List of fixed parameter flags (0=free, 1=fixed)
            magion: Magnetic ion type for form factor
            sqw: Scattering function
            pref: Prefactor function
            smoothfit: Whether to use smooth fitting
            
        Returns:
            Initial simulation result
            
        Raises:
            ValueError: If required parameters are missing
        """
        if hklw is None:
            raise ValueError("hklw DataFrame cannot be None")
        
        if exp is None:
            exp = self.taipan_expconfig(ef=14.87)
        
        if initial is None:
            raise ValueError("initial parameters cannot be None")
        
        if fixlist is None:
            fixlist = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Extract data columns
        try:
            H, K, L, W, Iobs = hklw.to_numpy().T
        except ValueError as e:
            raise ValueError(f"hklw must have exactly 5 columns [H,K,L,W,Iobs]: {e}")
        
        dIobs = np.sqrt(np.maximum(Iobs, 0))  # Ensure non-negative before sqrt
        
        # Get form factor
        ffactor = SelFormFactor(magion)
        if ffactor is None:
            logger.warning(f"Magnetic ion '{magion}' not found, using Mn2")
            ffactor = SelFormFactor("Mn2")
            if ffactor is None:
                raise ValueError("Failed to load default form factor")
        
        # Extract form factor parameters
        form_factor_params = [
            ffactor["AA"], ffactor["aa"], ffactor["BB"], ffactor["bb"],
            ffactor["CC"], ffactor["cc"], ffactor["DD"]
        ]
        
        # Combine parameters
        initial_new = list(initial.values()) + form_factor_params
        fixlist_new = fixlist + [0, 0, 0, 0, 0, 0, 0]
        
        # Run initial simulation
        sim_initial = exp.ResConv(
            sqw=sqw, pref=pref, nargout=2, hkle=[H, K, L, W],
            METHOD='fix', ACCURACY=[5, 5], p=initial_new
        )
        
        return sim_initial

    def taipan_convfit(self, hklw: pd.DataFrame = None, exp = None, 
                      initial: Union[Dict, List] = None, 
                      fixlist: List[int] = None, magion: str = "Mn2", 
                      sqw = SqwDemo, pref = PrefDemo):
        """
        Perform convolution fitting with resolution function.
        
        Args:
            hklw: DataFrame with columns [H, K, L, W, Iobs]
            exp: Experiment configuration
            initial: Initial parameters (dict or list)
            fixlist: List of fixed parameter flags
            magion: Magnetic ion type
            sqw: Scattering function
            pref: Prefactor function
            
        Returns:
            Tuple of (fitted_parameters, new_HKLW_array)
        """
        if hklw is None:
            raise ValueError("hklw DataFrame cannot be None")
        
        if exp is None:
            exp = self.taipan_expconfig(ef=14.87)
        
        if initial is None:
            raise ValueError("initial parameters cannot be None")
        
        if fixlist is None:
            fixlist = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Extract data
        try:
            H, K, L, W, Iobs = hklw.to_numpy().T
        except ValueError as e:
            raise ValueError(f"hklw must have 5 columns [H,K,L,W,Iobs]: {e}")
        
        dIobs = np.sqrt(np.maximum(Iobs, 0))
        
        # Get form factor
        ffactor = SelFormFactor(magion)
        if ffactor is None:
            logger.warning(f"Magnetic ion '{magion}' not found, using Mn2")
            ffactor = SelFormFactor("Mn2")
            if ffactor is None:
                raise ValueError("Failed to load default form factor")
        
        form_factor_params = [
            ffactor["AA"], ffactor["aa"], ffactor["BB"], ffactor["bb"],
            ffactor["CC"], ffactor["cc"], ffactor["DD"]
        ]
        
        # Handle dict or list initial parameters
        if isinstance(initial, dict):
            initial_list = list(initial.values())
        else:
            initial_list = list(initial)
        
        initial_new = initial_list + form_factor_params
        fixlist_new = fixlist + [0, 0, 0, 0, 0, 0, 0]
        
        logger.info(f"Starting fit with {len(initial_new)} parameters")
        
        # Perform ultrafast fit
        fitter = UltraFastFitConv(exp, sqw, pref, [H, K, L, W], Iobs, dIobs)
        result = fitter.fit_ultrafast(
            param_initial=initial_new,
            param_fixed_mask=fixlist_new,
            maxfev=200,
            use_analytical_jacobian=True,
            early_stopping=True,
            verbose=True
        )
        
        final_params = result['params']
        param_errors = result['param_errors']
        chi2_reduced = result['chi2_reduced']
        model_fit = result['model']
        
        # Generate fine grid for plotting
        newH = np.linspace(H[0], H[-1], 101)
        newK = np.linspace(K[0], K[-1], 101)
        newL = np.linspace(L[0], L[-1], 101)
        newW = np.linspace(W[0], W[-1], 101)
        
        final = exp.ResConv(
            sqw=sqw, pref=pref, nargout=2, 
            hkle=[newH, newK, newL, newW],
            METHOD='fix', ACCURACY=None, p=final_params
        )
        
        # Format output
        par_output = "Fitted Parameters:\n"
        par_output += f"En1  : {final_params[0]:8.6f}  ± {param_errors[0]:8.6f}\n"
        par_output += f"En2  : {final_params[1]:8.6f}  ± {param_errors[1]:8.6f}\n"
        par_output += f"Int1 : {final_params[2]*final_params[5]:8.6f}  ± {final_params[2]*param_errors[5]:8.6f}\n"
        par_output += f"Int2 : {final_params[5]:8.6f}  ± {param_errors[5]:8.6f}\n"
        par_output += f"FWHM1: {final_params[3]:8.6f}  ± {param_errors[3]:8.6f}\n"
        par_output += f"FWHM2: {final_params[4]:8.6f}  ± {param_errors[4]:8.6f}\n"
        par_output += f"bg   : {final_params[6]:8.6f}  ± {param_errors[6]:8.6f}\n"
        par_output += f"temp : {final_params[7]:8.6f}  ± {param_errors[7]:8.6f}\n"
        par_output += f"χ²_red: {chi2_reduced:8.6f}\n"
        
        logger.info(par_output)
        
        newHKLW = np.column_stack([newH, newK, newL, newW, final])
        
        return final_params, newHKLW

    def taipan_phonon_convfit(self, hklw: pd.DataFrame = None, exp = None, 
                             initial: Dict = None, 
                             fixlist: List[int] = None, 
                             sqw = SqwDemo, pref = PrefPhononDemo):
        """
        Perform phonon convolution fitting.
        
        Args:
            hklw: DataFrame with columns [H, K, L, W, Iobs]
            exp: Experiment configuration
            initial: Initial parameters dictionary
            fixlist: List of fixed parameter flags
            sqw: Scattering function
            pref: Phonon prefactor function
            
        Returns:
            Tuple of (fitted_parameters, data_and_fit_array)
        """
        if hklw is None:
            raise ValueError("hklw DataFrame cannot be None")
        
        if exp is None:
            exp = self.taipan_expconfig(ef=14.87)
        
        if initial is None:
            raise ValueError("initial parameters cannot be None")
        
        if not isinstance(initial, dict):
            raise TypeError("initial must be a dictionary")
        
        if fixlist is None:
            fixlist = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Extract data
        try:
            H, K, L, W, Iobs = hklw.to_numpy().T
        except ValueError as e:
            raise ValueError(f"hklw must have 5 columns: {e}")
        
        dIobs = np.sqrt(np.maximum(Iobs, 0))
        
        initialnew = list(initial.values())
        
        # Perform fit
        fitter = FitConv(exp, sqw, pref, [H, K, L, W], Iobs, dIobs)
        
        [final_param, dpa, chisqN, sim, CN, PQ, nit, kvg, details] = \
            fitter.fitwithconv(
                exp, sqw, pref, [H, K, L, W], Iobs, dIobs,
                param=initialnew, paramfixed=fixlist
            )
        
        # Format output
        str_output = "Fitted Parameters:\n"
        parlist = list(initial.keys())
        
        for index, (iname, ipar, ierr) in enumerate(zip(parlist, final_param, dpa)):
            if index < 8:
                str_output += f"P{index}( {iname} ):\t {ipar:6.8f}\t +/- \t{ierr:6.8f}\n"
        
        str_output += f"Reduced χ²: {chisqN:6.8f}\n"
        logger.info(str_output)
        
        data_and_fit = np.vstack((H, K, L, W, Iobs, dIobs, sim))
        
        return final_param, data_and_fit

    def taipan_batch_validate_new(self, filename: Union[str, Path] = None, 
                                  bsim: bool = False, exp = None):
        """
        Read and validate commands from a batch file.
        
        Args:
            filename: Path to batch file
            bsim: Whether to simulate scans
            exp: Experiment configuration for simulation
            
        Returns:
            List of tuples: (line_number, command, is_valid, message)
        """
        if filename is None:
            raise ValueError("Filename cannot be None")
        
        filepath = Path(filename)
        if not filepath.exists():
            return [(0, "", False, f"File not found: {filename}")]
        
        results = []
        validator = TaipanCommandValidator()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines()]
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Handle mscan command
                if line.startswith('mscan'):
                    is_valid, message = validator.validate_command(line)
                    if is_valid:
                        logger.info(f"Line {i+1}: {line} - Validation Passed")
                        results.append((i + 1, line, is_valid, message))
                        
                        if bsim:
                            if exp is None:
                                logger.warning("No experiment provided for simulation")
                            else:
                                try:
                                    simres = self.taipan_scansim(line, exp=exp)
                                    if simres is not None:
                                        logger.info(f"Simulation result:\n{simres}")
                                except Exception as e:
                                    logger.error(f"Simulation failed: {e}")
                    else:
                        logger.warning(f"Line {i+1}: Invalid mscan command")
                        results.append((i + 1, line, False, message))
                    i += 1
                
                # Handle drive command
                elif line.startswith('drive'):
                    # Check if next line is runscan
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith('runscan'):
                        combined_cmd = f"{line}\n{lines[i + 1]}"
                        is_valid, message = validator.validate_command(combined_cmd)
                        
                        if is_valid:
                            logger.info(f"Lines {i+1}-{i+2}: Combined command validated")
                            results.append((i + 1, combined_cmd, is_valid, message))
                            
                            if bsim and exp is not None:
                                try:
                                    simres = self.taipan_scansim(combined_cmd, exp=exp)
                                    if simres is not None:
                                        logger.info(f"Simulation result:\n{simres}")
                                except Exception as e:
                                    logger.error(f"Simulation failed: {e}")
                            i += 2
                        else:
                            logger.warning(f"Lines {i+1}-{i+2}: Invalid combined command")
                            results.append((i + 1, combined_cmd, False, message))
                            i += 2
                    else:
                        # Single drive command
                        is_valid, message = validator.validate_command(line)
                        results.append((i + 1, line, is_valid, message))
                        logger.info(f"Line {i+1}: Single drive command")
                        i += 1
                
                else:
                    # Unknown command
                    results.append((i + 1, line, False, "Invalid scan command"))
                    logger.warning(f"Line {i+1}: Unknown command type")
                    i += 1
                    
        except Exception as e:
            return [(0, "", False, f"Error reading file: {str(e)}")]
        
        return results

    def taipan_batch_validate(self, batchfile: Union[str, Path] = None):
        """
        Validate batch file commands (legacy version).
        
        Args:
            batchfile: Path to batch file
            
        Returns:
            Tuple of (validation_result_string, valid_command_list)
        """
        if batchfile is None:
            raise ValueError("Batchfile cannot be None")
        
        filepath = Path(batchfile)
        if not filepath.exists():
            raise FileNotFoundError(f"Batch file not found: {batchfile}")
        
        valid_cmd_list = []
        validation_result = ""
        validator = TaipanCommandValidator()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            i = 0
            while i < len(lines):
                # Skip empty lines
                if not lines[i].strip():
                    i += 1
                    continue
                
                current_line = lines[i].strip()
                logger.info(f"Processing line {i + 1}")
                validation_result += f"\nLine {i + 1}:\n"
                
                if current_line.startswith('mscan'):
                    valid, message = validator.validate_command(current_line)
                    logger.info(f"Command: {current_line}")
                    
                    if valid:
                        logger.info("Validation passed!")
                        validation_result += f"Command: {current_line}\nValidation passed!\n\n"
                        valid_cmd_list.append(current_line)
                    else:
                        logger.warning("mscan validation failed")
                        validation_result += f"Command: {current_line}\nError: mscan error!\n\n"
                    i += 1
                
                elif current_line.startswith('drive'):
                    # Look for runscan on next line
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    
                    if next_line.startswith('runscan'):
                        combined_cmd = f"{current_line}\n{next_line}"
                        valid, message = validator.validate_command(combined_cmd)
                        logger.info(f"Combined command:\n{combined_cmd}")
                        
                        if valid:
                            logger.info("Validation passed!")
                            validation_result += f"Combined command:\n{combined_cmd}\nValidation passed!\n\n"
                            valid_cmd_list.extend([current_line, next_line])
                        else:
                            logger.warning("Combined drive-runscan validation failed")
                            validation_result += f"Combined command:\n{combined_cmd}\nERROR: combined drive-runscan error!\n\n"
                        i += 2
                    else:
                        # Single drive command
                        valid, message = validator.validate_command(current_line)
                        logger.info(f"Command: {current_line}")
                        
                        if valid:
                            logger.info("Validation passed!")
                            validation_result += f"Command: {current_line}\nValidation passed!\n\n"
                        else:
                            logger.warning("drive validation failed")
                            validation_result += f"Command: {current_line}\nERROR: drive command error!\n\n"
                        i += 1
                
                elif current_line.startswith('title'):
                    valid, message = validator.validate_command(current_line)
                    logger.info(f"Command: {current_line}")
                    
                    if valid:
                        logger.info("Validation passed!")
                        validation_result += f"Command: {current_line}\nValidation passed!\n\n"
                    else:
                        logger.warning("title validation failed")
                        validation_result += f"Command: {current_line}\nError: title command error!\n\n"
                    i += 1
                
                else:
                    logger.warning(f"Unknown command: {current_line}")
                    validation_result += f"Command: {current_line}\nERROR: unknown command!\n\n"
                    i += 1
                    
        except FileNotFoundError:
            logger.error("Batch file not found")
            raise
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise
        
        return validation_result, valid_cmd_list

    def taipan_scansim(self, cmdline: str = "", exp = None, 
                      runscanpos: List[float] = None):
        """
        Simulate scan from command line.
        
        Args:
            cmdline: Command line string (drive+runscan or mscan)
            exp: Experiment configuration
            runscanpos: Position for runscan [qh, qk, ql, en]
            
        Returns:
            DataFrame with simulated scan positions or None on error
        """
        if runscanpos is None:
            runscanpos = [2, 0, 0, 0]
        
        if not cmdline:
            logger.warning("No command provided")
            return None
        
        if exp is None:
            logger.warning("No experiment configuration provided")
            return None
        
        pd.set_option('display.expand_frame_repr', False)
        
        cmd_lines = cmdline.splitlines()
        cmd_items = cmd_lines[0].split()
        
        logger.info(f"Simulating: {cmdline}")
        
        if cmd_items[0] == 'drive':
            if len(cmd_lines) >= 2:
                return self.taipan_runscansim(cmd_lines[0], cmd_lines[1], exp)
            else:
                logger.error("drive command requires runscan on next line")
                return None
        
        elif cmd_items[0] == 'mscan':
            return self.taipan_mscansim(cmd_lines[0], exp)
        
        else:
            logger.error(f"Unknown scan command: {cmd_items[0]}")
            return None

    def taipan_mscansim(self, mscanline: str = "", exp = None):
        """
        Simulate mscan command.
        
        Args:
            mscanline: mscan command string
            exp: Experiment configuration
            
        Returns:
            DataFrame with scan positions or None on error
        """
        if not exp:
            logger.warning("No experiment configuration provided")
            return None
        
        if not mscanline:
            logger.warning("No mscan command provided")
            return None
        
        motorlist = ['m1', 'm2', 's1', 's2', 'a1', 'a2', 'qh', 'qk', 'ql', 'en']
        df_scanpos = pd.DataFrame()
        validator = TaipanCommandValidator()
        
        mscan_valid, message = validator.validate_command(mscanline)
        
        if not mscan_valid:
            logger.error(f"Invalid mscan command: {message}")
            return None
        
        cmd_items = mscanline.split()
        
        if cmd_items[0] != 'mscan':
            logger.error("Command must start with 'mscan'")
            return None
        
        # Validate command structure
        # Format: mscan motor1 start1 step1 motor2 start2 step2 ... N time/monitor count
        if len(cmd_items) < 7:
            logger.error("mscan requires at least: mscan motor start step N time/monitor count")
            return None
        
        # Check if number of motor parameters is divisible by 3
        num_motor_params = len(cmd_items) - 4
        if num_motor_params % 3 != 0:
            logger.error("Number of motor parameters must be divisible by 3")
            return None
        
        num_motors = num_motor_params // 3
        
        # Validate motor parameters
        for i in range(num_motors):
            motor = cmd_items[3*i + 1]
            start_str = cmd_items[3*i + 2]
            step_str = cmd_items[3*i + 3]
            
            if motor not in motorlist:
                logger.error(f"Invalid motor: {motor}")
                return None
            
            if not strisfloat(start_str) or not strisfloat(step_str):
                logger.error(f"Motor {motor}: start and step must be floats")
                return None
        
        # Validate step count
        if not strisint(cmd_items[-3]):
            logger.error("Number of steps must be an integer")
            return None
        
        numstep = int(cmd_items[-3])
        if numstep <= 0:
            logger.error("Number of steps must be positive")
            return None
        
        # Validate count mode
        if cmd_items[-2] not in ['time', 'monitor']:
            logger.error("Must count against 'time' or 'monitor'")
            return None
        
        # Generate scan positions
        for i in range(num_motors):
            motor = cmd_items[3*i + 1]
            start = float(cmd_items[3*i + 2])
            stepsize = float(cmd_items[3*i + 3])
            
            positions = [start + j*stepsize for j in range(numstep + 1)]
            df_scanpos[motor] = positions
        
        # Calculate angles if HKLE scan
        if all(col in df_scanpos.columns for col in ['qh', 'qk', 'ql', 'en']):
            hkle = df_scanpos[['qh', 'qk', 'ql', 'en']].to_numpy().T
            
            try:
                [M1, M2, S1, S2, A1, A2, Q] = exp.get_spec_angles(hkle)
                df_scanpos['m1'] = M1
                df_scanpos['m2'] = M2
                df_scanpos['s1'] = S1
                df_scanpos['s2'] = S2
                df_scanpos['a1'] = A1
                df_scanpos['a2'] = A2
            except Exception as e:
                logger.error(f"Failed to calculate angles: {e}")
        
        return df_scanpos

    def taipan_runscansim(self, driveline: str = None, scanline: str = None, 
                         exp = None):
        """
        Simulate combined drive+runscan command.
        
        Args:
            driveline: drive command string
            scanline: runscan command string
            exp: Experiment configuration
            
        Returns:
            DataFrame with scan positions or None on error
        """
        if not exp:
            logger.warning("No experiment configuration provided")
            return None
        
        if not driveline:
            logger.warning("No drive command provided")
            return None
        
        if not scanline:
            logger.warning("No runscan command provided")
            return None
        
        motorlist = ['m1', 'm2', 's1', 's2', 'a1', 'a2', 'qh', 'qk', 'ql', 'en']
        validator = TaipanCommandValidator()
        
        # Validate combined command
        runscan_valid, message = validator.validate_command(driveline + '\n' + scanline)
        
        if not runscan_valid:
            logger.error(f"Invalid drive+runscan: {message}")
            return None
        
        drive_items = driveline.split()
        scan_items = scanline.split()
        
        # Validate scan axis exists in drive command
        scan_axis = scan_items[1]
        
        try:
            motor_index = drive_items.index(scan_axis)
        except ValueError:
            logger.error(f"Scan axis '{scan_axis}' not found in drive command")
            return None
        
        # Build equivalent mscan command
        newcmd_parts = ["mscan"]
        
        # Process drive motors
        for index in range(1, len(drive_items), 2):
            if index >= len(drive_items):
                break
            
            motor = drive_items[index]
            position = drive_items[index + 1]
            
            if motor == scan_axis:
                # This is the scan axis
                try:
                    startpos = float(scan_items[2])
                    stoppos = float(scan_items[3])
                    stepno = int(scan_items[4])
                    
                    if stepno <= 0:
                        logger.error("Number of steps must be positive")
                        return None
                    
                    stepsize = (stoppos - startpos) / stepno
                    newcmd_parts.extend([motor, str(startpos), str(stepsize)])
                    
                except (ValueError, IndexError) as e:
                    logger.error(f"Invalid runscan parameters: {e}")
                    return None
            else:
                # Fixed motor
                newcmd_parts.extend([motor, position, "0.0"])
        
        # Add step count and counting mode
        try:
            newcmd_parts.extend([scan_items[4], scan_items[5], scan_items[6]])
        except IndexError:
            logger.error("Invalid runscan format")
            return None
        
        newcmd = " ".join(newcmd_parts)
        logger.info(f"Converted to mscan: {newcmd}")
        
        # Simulate using mscan
        sim_result = self.taipan_mscansim(newcmd, exp)
        return sim_result

    def taipan_bosecorrect(self, path: Union[str, Path] = '', 
                          scanlist: List = None, sample_T: float = 1.5):
        """
        Apply Bose correction to energy scans.
        
        Args:
            path: Directory path containing data files
            scanlist: List of scan numbers
            sample_T: Sample temperature in Kelvin
            
        Returns:
            Tuple of (corrected_dflist, scan_info_list)
        """
        ppath = Path(path)
        
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if sample_T <= 0:
            raise ValueError("Sample temperature must be positive (in Kelvin)")
        
        # Get data (this calls a method that should exist in part 1)
        dflist, scinf_list = self.taipan_reduction(path, scanlist)
        
        newdflist = []
        
        for index, df in enumerate(dflist):
            # Filter for positive energy transfers only
            newdf = df.loc[df['e'] > 0].copy()
            newdf.reset_index(drop=True, inplace=True)
            
            if scinf_list[index]['scanax'] == 'e':
                deltae = newdf['e'].to_numpy()
                counts = newdf['detector'].to_numpy()
                
                # Apply Bose correction
                # Formula: n/(n+1) where n = 1/(exp(E/kT) - 1)
                try:
                    bose_factor = deltae / (sample_T * self.BOLTZMANN_CONSTANT_MEV)
                    # Avoid division by zero and overflow
                    bose_factor = np.clip(bose_factor, 1e-10, 100)
                    exp_term = np.exp(-bose_factor)
                    
                    # Check for potential division by zero
                    denominator = 1 - exp_term
                    if np.any(np.abs(denominator) < 1e-10):
                        logger.warning("Near-zero denominator in Bose correction")
                        denominator = np.where(
                            np.abs(denominator) < 1e-10, 
                            1e-10, 
                            denominator
                        )
                    
                    correction = 1 + 1 / denominator
                    corrected_counts = counts / correction
                    
                    newdf['detector'] = corrected_counts
                    
                except Exception as e:
                    logger.error(f"Error applying Bose correction: {e}")
                    # Return uncorrected data on error
            else:
                logger.warning(f"Scan {index} is not an energy scan, skipping correction")
            
            newdflist.append(newdf)
        
        return newdflist, scinf_list

    def taipan_gen_escans(self, hkl_list: List[List[float]] = None, 
                         e_range: List[List[float]] = None, 
                         e_step: float = 0.1, sample_T: float = 1.5, 
                         beammon: int = 100000, 
                         titleTemplate: str = 'ABO3 HH0-00L'):
        """
        Generate batch energy scan commands for multiple Q positions.
        
        Args:
            hkl_list: List of [h, k, l] positions
            e_range: List of [e_min, e_max] ranges
            e_step: Energy step size in meV
            sample_T: Sample temperature in Kelvin
            beammon: Monitor count
            titleTemplate: Template for scan titles
            
        Returns:
            String containing batch scan commands
        """
        if hkl_list is None:
            hkl_list = [[0, 0, 1], [0, 0, 1.5]]
        
        if e_range is None:
            e_range = [[0, 5], [0, 6]]
        
        if len(hkl_list) != len(e_range):
            raise ValueError("hkl_list and e_range must have same length")
        
        if e_step <= 0:
            raise ValueError("e_step must be positive")
        
        batch_lines = []
        
        for index, hkl in enumerate(hkl_list):
            e_min, e_max = e_range[index]
            
            if e_max <= e_min:
                logger.warning(f"Invalid energy range for Q={hkl}: [{e_min}, {e_max}]")
                continue
            
            steps = 1 + round((e_max - e_min) / e_step)
            
            # Build command strings
            title_str = (
                f'title "{titleTemplate} Q({hkl[0]}, {hkl[1]}, {hkl[2]}) '
                f'Escan ({e_min} ~ {e_max}meV) at {sample_T} K"'
            )
            drive_str = f'drive qh {hkl[0]} qk {hkl[1]} ql {hkl[2]} en {e_min}'
            scan_str = f'runscan en {e_min} {e_max} {steps} monitor {beammon}'
            
            batch_lines.extend([title_str, drive_str, scan_str, ''])
        
        batch_str = '\n'.join(batch_lines)
        logger.info(f"Generated {len(hkl_list)} energy scan commands")
        
        return batch_str

    def taipan_gen_qscans(self, startQ: List[List[float]] = None, 
                         endQ: List[List[float]] = None, 
                         q_step: List[List[float]] = None, 
                         en: float = 2, sample_T: float = 1.5, 
                         beammon: int = 100000, 
                         titleTemplate: str = 'ABO3 HH0-00L'):
        """
        Generate batch Q scan commands from startQ to endQ.
        
        Args:
            startQ: List of [h, k, l] start positions
            endQ: List of [h, k, l] end positions
            q_step: List of [dh, dk, dl] step sizes
            en: Energy transfer in meV
            sample_T: Sample temperature in Kelvin
            beammon: Monitor count
            titleTemplate: Template for scan titles
            
        Returns:
            String containing batch scan commands
        """
        if startQ is None:
            startQ = [[0, 0, 1.1], [0.1, 0, 1.5]]
        
        if endQ is None:
            endQ = [[0, 0, 1.6], [0.9, 0, 1.5]]
        
        if q_step is None:
            q_step = [[0.00, 0.00, 0.01], [0.01, 0.00, 0.00]]
        
        if len(startQ) != len(endQ):
            raise ValueError("startQ and endQ must have same length")
        
        batch_lines = []
        qNameList = ['qh', 'qk', 'ql']
        
        for index in range(len(startQ)):
            # Calculate differences
            dH = abs(startQ[index][0] - endQ[index][0])
            dK = abs(startQ[index][1] - endQ[index][1])
            dL = abs(startQ[index][2] - endQ[index][2])
            
            deltaList = [dH, dK, dL]
            
            # Identify scan and fixed axes
            scanQList = [i for i, delta in enumerate(deltaList) if delta >= self.EPSILON]
            fixQList = [i for i, delta in enumerate(deltaList) if delta < self.EPSILON]
            
            if len(scanQList) == 0:
                logger.warning(f"Scan {index}: startQ and endQ are identical")
                continue
            
            # Calculate midpoint for title
            midQ = [
                (startQ[index][i] + endQ[index][i]) / 2 
                for i in range(3)
            ]
            
            if len(scanQList) == 1:
                # Single axis scan - use drive+runscan
                scan_idx = scanQList[0]
                
                if abs(q_step[index][scan_idx]) < self.EPSILON:
                    logger.error(f"Scan {index}: step size cannot be zero")
                    continue
                
                steps = 1 + round(
                    (endQ[index][scan_idx] - startQ[index][scan_idx]) / 
                    q_step[index][scan_idx]
                )
                
                # Build command strings
                title_str = (
                    f'title "{titleTemplate} Qscan {qNameList[scan_idx]}'
                    f'({startQ[index][scan_idx]}~{endQ[index][scan_idx]}) around '
                    f'Q({midQ[0]}, {midQ[1]}, {midQ[2]}) at E = {en}meV, {sample_T}K"'
                )
                drive_str = (
                    f'drive qh {startQ[index][0]} qk {startQ[index][1]} '
                    f'ql {startQ[index][2]} en {en}'
                )
                scan_str = (
                    f'runscan {qNameList[scan_idx]} {startQ[index][scan_idx]} '
                    f'{endQ[index][scan_idx]} {steps} monitor {beammon}'
                )
                
                batch_lines.extend([title_str, drive_str, scan_str, ''])
            
            elif len(scanQList) >= 2:
                # Multi-axis scan - use mscan
                if all(abs(q_step[index][i]) < self.EPSILON for i in range(3)):
                    logger.error(f"Scan {index}: at least one step must be non-zero")
                    continue
                
                # Calculate steps from first scan axis
                scan_idx = scanQList[0]
                if abs(q_step[index][scan_idx]) < self.EPSILON:
                    # Find first non-zero step
                    scan_idx = next(
                        (i for i in scanQList if abs(q_step[index][i]) >= self.EPSILON),
                        scanQList[0]
                    )
                
                steps = 1 + round(
                    (endQ[index][scan_idx] - startQ[index][scan_idx]) / 
                    q_step[index][scan_idx]
                )
                
                # Build title
                title_str = (
                    f'title "{titleTemplate} Qscan {qNameList[scan_idx]}'
                    f'({startQ[index][scan_idx]}~{endQ[index][scan_idx]}) around '
                    f'Q({midQ[0]}, {midQ[1]}, {midQ[2]}) at E = {en}meV, {sample_T}K"'
                )
                
                # Build mscan command
                scan_parts = ['mscan']
                scan_parts.extend([
                    'qh', str(startQ[index][0]), str(q_step[index][0]),
                    'qk', str(startQ[index][1]), str(q_step[index][1]),
                    'ql', str(startQ[index][2]), str(q_step[index][2]),
                    'en', str(en), '0',
                    str(steps), 'monitor', str(beammon)
                ])
                scan_str = ' '.join(scan_parts)
                
                batch_lines.extend([title_str, scan_str, ''])
        
        batch_str = '\n'.join(batch_lines)
        logger.info(f"Generated {len(startQ)} Q scan commands")
        
        return batch_str