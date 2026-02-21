from pathlib import Path
import pandas as pd
import numpy as np
import linecache
import h5py
import re
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Dict, Tuple, Any

from ..base import TasData
from ..utils.toolfunc import (AnglesToQhkl, strisfloat, strisint, ni_s2_residual, 
                ni_s2_residual_30p5meV, gaussian_residual, lorentzian_residual, 
                fit_peak, angle2, SqwDemo, PrefDemo, SelFormFactor, PrefPhononDemo)
from ..utils.toolfunc import descend_obj, h5dump
from .validator import TaipanCommandValidator

import inspy as npy
from inspy import TripleAxisSpectr
from inspy.insfit import FitConv, UltraFastFitConv
from lmfit import Parameters, fit_report, minimize



class Taipan(TasData):
    """Taipan class that extends TasData."""
    
    def __init__(self, expnum, title, sample, user):
        super().__init__("Taipan", expnum, title, sample, user)

    def _taipan_data_header_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a neutron data file and extract header information.
        Args:
            path: Path to the data file
        Returns:
            Dictionary containing all extracted parameters
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If the file cannot be read
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        params = {}
        last_header_line = ""
        last_header_line_num = 0
        data_start_line = 0
        line_count = 0
        array_params = ['latticeconstants', 'ubmatrix']
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    
                    if line.startswith('#'):
                        last_header_line = line
                        last_header_line_num = line_count
                        
                        if line.strip().startswith('# Pt.') or '# Pt.' in line:
                            continue
                        
                        clean_line = line[1:].strip()
                        
                        if not clean_line or '=' not in clean_line:
                            continue
                        
                        key, value = clean_line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key in array_params:
                            try:
                                value = [float(item) for item in value.split()]
                            except ValueError as e:
                                print(f"Warning: Could not parse array parameter '{key}': {e}")
                                continue
                        
                        params[key] = value
                    else:
                        data_start_line = line_count
                        break
        except UnicodeDecodeError:
            raise IOError(f"File encoding error: {path}")
        except Exception as e:
            raise IOError(f"Error reading file {path}: {e}")
        
        params['_data_start_line'] = data_start_line
        params['_header_line'] = last_header_line
        params['_header_line_num'] = last_header_line_num

        if 'experiment_number' in params:
            params['expno'] = params.pop('experiment_number')
        if 'scan' in params:
            params['scanno'] = params.pop('scan')
        if 'latticeconstants' in params:
            params['lattice'] = params.pop('latticeconstants')

        params['scanax1'] = params.get('def_x', "")
        if params['scanax1'] == "":
            print("Warning: Scan axis is empty! Please check the data file.")

        return params

    def taipan_data_to_pd(self, path: Union[str, Path] = '', scanno: Optional[int] = None) -> pd.DataFrame:
        """
        Load a single neutron data file into a pandas DataFrame with metadata in attrs.
        Args:
            path: Path to the data directory
            scanno: Scan number
        Returns:
            DataFrame containing the data with metadata in attrs
        Raises:
            ValueError: If path is invalid or scanno is None
            FileNotFoundError: If the data file doesn't exist
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        ppath = Path(path)
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if scanno is None:
            raise ValueError("Scan number cannot be None")
        
        try:
            scanno = int(scanno)
        except (ValueError, TypeError):
            raise ValueError(f"Scan number must be an integer, got: {scanno}")
        
        fileinitial = f"TAIPAN_exp{self.expnum}_scan"
        filefullname = f"{fileinitial}{scanno}.dat"
        fullfilepath = ppath / filefullname
        
        if not fullfilepath.exists():
            raise FileNotFoundError(f"Data file not found: {fullfilepath}")
        
        try:
            params = self._taipan_data_header_info(fullfilepath)
        except Exception as e:
            print(f"Error parsing header: {e}")
            raise
        
        data_start_line = params.pop('_data_start_line', 0)
        header_line = params.pop('_header_line', '')
        header_line_num = params.pop('_header_line_num', 0)
        
        if header_line.startswith('#'):
            header_line = header_line[1:].strip()
        
        taipan_column_names = []
        if header_line:
            taipan_column_names = [col.strip() for col in header_line.split() if col.strip()]
        
        try:
            df = pd.read_csv(fullfilepath, sep=r'\s+', header=None, comment='#')
            
            if len(taipan_column_names) != len(df.columns):
                print(f"Warning: Column count mismatch. Expected {len(df.columns)}, got {len(taipan_column_names)} names")
                if len(taipan_column_names) < len(df.columns):
                    taipan_column_names.extend([f'col_{i}' for i in range(len(taipan_column_names), len(df.columns))])
                else:
                    taipan_column_names = taipan_column_names[:len(df.columns)]
            
            seen = {}
            unique_taipan_column_names = []
            for name in taipan_column_names:
                if name not in seen:
                    seen[name] = 1
                    unique_taipan_column_names.append(name)
                else:
                    seen[name] += 1
                    unique_taipan_column_names.append(f"{name}_dup")
            
            df.columns = unique_taipan_column_names

            duplicate_map = {'qh': 'h', 'qk': 'k', 'ql': 'l', 'en': 'e'}
            for dup_col, orig_col in duplicate_map.items():
                if dup_col in df.columns and orig_col in df.columns:
                    df.drop(columns=dup_col, inplace=True)
                    if dup_col in unique_taipan_column_names:
                        unique_taipan_column_names.remove(dup_col)
            
            taipan_motorname_list = ['h', 'k', 'l', 'e', 'T1_Sensor1', 'T1_Sensor2', 'T1_Sensor3']
            stdtas_motorname_list = ['qh', 'qk', 'ql', 'en', 'tempVTI', 'tempSAMP', 'tempSAMPh']
            replace_map = dict(zip(taipan_motorname_list, stdtas_motorname_list))
            
            std_column_names = [replace_map.get(item, item) for item in unique_taipan_column_names]
            df.columns = std_column_names

        except pd.errors.EmptyDataError:
            print("Warning: Data file is empty")
            df = pd.DataFrame()
        except Exception as e:
            print(f"Warning: Error parsing data section: {e}")
            df = pd.DataFrame()
        
        df.attrs = params
        return df

    def taipan_hdf_to_pd(self, path: Union[str, Path] = '', scanno: Union[int, str] = 90000) -> pd.DataFrame:
        """
        Load HDF5 file into a pandas DataFrame.
        Args:
            path: Path to the data directory
            scanno: Scan number or filename
        Returns:
            DataFrame containing the data with metadata in attrs
        Raises:
            ValueError: If path is invalid
            FileNotFoundError: If HDF5 file doesn't exist
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        ppath = Path(path)
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        filename = str(scanno)
        
        if len(filename) == 5:
            filename = '00' + filename
        elif len(filename) == 6:
            filename = '0' + filename
        elif len(filename) < 5:
            raise ValueError(f"Invalid scan number: {scanno}")
        
        fileext = ".nx.hdf"
        fileinitial = 'TPN'
        fullfilename = fileinitial + filename + fileext
        fullpath = ppath / fullfilename
        
        if not fullpath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {fullpath}")
        
        try:
            namelist, col_list = h5dump(fullpath)
        except Exception as e:
            raise IOError(f"Error reading HDF5 structure: {e}")
        
        temptitle = 'no title'
        pd_hdf = pd.DataFrame([])
        tempax = None
        
        try:
            with h5py.File(str(fullpath), 'r') as file:
                title_dataset = file.get("/entry1/experiment/title")
                if title_dataset is not None and len(title_dataset) > 0:
                    temptitle = title_dataset[0]
                
                for index, name in enumerate(namelist):
                    try:
                        if name.find("data") != -1:
                            xx = np.array(file[name])
                            if len(xx) > 0 and np.abs(xx[0] - xx[-1]) / len(xx) > 0.0006:
                                tempax = col_list[index]
                        else:
                            new_col = pd.DataFrame(file[name])
                            new_col.columns = [col_list[index]]
                            pd_hdf = pd.merge(pd_hdf, new_col, how='outer', 
                                            left_index=True, right_index=True, suffixes=('', ''))
                    except Exception as e:
                        print(f"Warning: Could not process dataset '{name}': {e}")
                        continue
        
        except Exception as e:
            raise IOError(f"Error reading HDF5 file: {e}")
        
        neworder = pd_hdf.columns.to_list()
        
        if tempax and tempax in neworder:
            neworder.pop(neworder.index(tempax))
            neworder.insert(0, tempax)
            pd_hdf = pd_hdf.reindex(columns=neworder)
        
        newcolnames = neworder.copy()
        
        column_rename_map = {
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
        
        for old_name, new_name in column_rename_map.items():
            if old_name in newcolnames:
                newcolnames[newcolnames.index(old_name)] = new_name
        
        pd_hdf.columns = newcolnames
        
        scaninfo = {"scanno": scanno, "scanax": tempax, "title": temptitle}
        
        required_cols = ['qh', 'qk', 'ql', 'en', 'detector', 'monitor']
        available_cols = [col for col in required_cols if col in pd_hdf.columns]
        if available_cols:
            print(pd_hdf[available_cols])
        print(pd_hdf.columns)
        
        pd_hdf.attrs = scaninfo
        return pd_hdf

    def taipan_scanlist_to_dflist(self, path: Union[str, Path] = "", 
                                  scanlist: Optional[List[int]] = None) -> List[pd.DataFrame]:
        """
        Convert a list of scan numbers to a list of DataFrames.
        Args:
            path: Path to the data directory
            scanlist: List of scan numbers
        Returns:
            List of DataFrames
        Raises:
            ValueError: If scanlist is invalid
        """
        if scanlist is None:
            raise ValueError("Scanlist cannot be None")
        
        if not isinstance(scanlist, list):
            raise ValueError("Scanlist must be a list")
        
        if not all(isinstance(x, (int, np.integer)) for x in scanlist):
            raise ValueError("All elements in scanlist must be integers")
        
        dflist = []
        failed_scans = []
        
        for scanno in scanlist:
            try:
                df = self.taipan_data_to_pd(path, scanno)
                dflist.append(df)
            except Exception as e:
                print(f"Warning: Failed to load scan {scanno}: {e}")
                failed_scans.append(scanno)
                continue
        
        if failed_scans:
            print(f"Failed to load scans: {failed_scans}")
        
        return dflist

    def taipan_combplot(self, path: Union[str, Path] = '', scanlist: Optional[List] = None, 
                       fit: bool = False, norm_mon_count: int = 1000000, 
                       overplot: bool = False, offset: int = 1000, 
                       initial: Optional[Dict] = None) -> None:
        """
        Combine and plot multiple scans.
        Args:
            path: Path to the data directory
            scanlist: List of scan numbers (can include sublists)
            fit: Whether to fit the data
            norm_mon_count: Monitor count for normalization
            overplot: Whether to overlay plots
            offset: Vertical offset between plots
            initial: Initial parameters for fitting
        Raises:
            ValueError: If scanlist is invalid
        """
        if scanlist is None:
            raise ValueError("Scanlist cannot be None")
        
        if not isinstance(scanlist, list):
            raise ValueError("Scanlist must be a list")
        
        dflist = []
        
        for scanno in scanlist:
            try:
                if isinstance(scanno, list):
                    subscanlist = scanno
                    subdflist = self.taipan_scanlist_to_dflist(path, subscanlist)
                    comb_df = super().tas_datacombine(subdflist)
                    dflist.append(comb_df)
                elif isinstance(scanno, (int, np.integer)):
                    single_df = self.taipan_data_to_pd(path, scanno)
                    dflist.append(single_df)
                else:
                    print(f"Warning: Skipping invalid scan entry: {scanno}")
                    continue
            except Exception as e:
                print(f"Error processing scan {scanno}: {e}")
                continue
        
        if not dflist:
            raise ValueError("No valid data to plot")
        
        super().tas_combplot(dflist, fit, norm_mon_count, overplot, offset, initial)

    def taipan_batch_reduction(self, path: Union[str, Path] = '', 
                              scanlist: Optional[List] = None,
                              motorlist: List[str] = None) -> List[pd.DataFrame]:
        """
        Batch reduction of multiple scans.
        Args:
            path: Path to the data directory
            scanlist: List of scan numbers
            motorlist: List of motor/column names to extract
        Returns:
            List of reduced DataFrames
        Raises:
            ValueError: If inputs are invalid
        """
        if scanlist is None:
            raise ValueError("Scanlist cannot be None")
        
        if motorlist is None:
            motorlist = ['qh', 'qk', 'ql', 'en', 'ei', 'ef', 'm1', 'm2', 's1', 's2', 
                        'a1', 'a2', 'detector', 'monitor', 'tempVTI', 'tempSAMP']
        
        if not isinstance(motorlist, list):
            raise ValueError("Motorlist must be a list")
        
        dflist = []
        
        for scanno in scanlist:
            try:
                if isinstance(scanno, (int, np.integer)):
                    df = self.taipan_data_to_pd(path, scanno)
                elif isinstance(scanno, list):
                    temp_dflist = self.taipan_scanlist_to_dflist(path, scanno)
                    df = super().tas_datacombine(temp_dflist)
                else:
                    print(f"Warning: Invalid scan entry type: {type(scanno)}")
                    continue
                
                if df.empty:
                    print(f"Warning: Empty DataFrame for scan {scanno}")
                    continue
                
                col_notexist = [col for col in motorlist if col not in df.columns]
                if col_notexist:
                    print(f"Warning: Columns not in data for scan {scanno}: {col_notexist}")
                
                col_names_available = [x for x in motorlist if x in df.columns]
                
                if col_names_available:
                    dflist.append(df[col_names_available])
                else:
                    print(f"Warning: No requested columns found for scan {scanno}")
                    
            except Exception as e:
                print(f"Error processing scan {scanno}: {e}")
                continue
        
        return dflist

    def taipan_reduction_by_row(self, path: Union[str, Path] = '', 
                               scanlist: Optional[List] = None,
                               motorlist: Optional[List[str]] = None,
                               sortby: str = "tempSAMP") -> pd.DataFrame:
        """
        Reduce multiple scans and combine into a single DataFrame sorted by a column.
        Args:
            path: Path to the data directory
            scanlist: List of scan numbers
            motorlist: List of motor/column names to extract
            sortby: Column name to sort by
        Returns:
            Combined and sorted DataFrame
        """
        if motorlist is None:
            motorlist = ['qh', 'qk', 'ql', 'en', 'ei', 'ef', 'm1', 'm2', 's1', 's2',
                        'a1', 'a2', 'detector', 'monitor', 'tempVTI', 'tempSAMP']
        
        dflist = self.taipan_batch_reduction(path, scanlist, motorlist)
        df_extend = pd.DataFrame(columns=motorlist)
        
        for df in dflist:
            if df is not None and not df.empty and not df.isna().all().all():
                df_extend = pd.concat([df_extend, df], ignore_index=True)
        
        if sortby in df_extend.columns:
            df_extend = df_extend.sort_values(by=sortby, ascending=True).reset_index(drop=True)
        else:
            print(f"Warning: Sort column '{sortby}' not found in data")
        
        return df_extend

    def taipan_export_hklw(self, path: Union[str, Path] = '', 
                          scanlist: Optional[List] = None,
                          hklw_file: str = "") -> List[pd.DataFrame]:
        """
        Export scans to HKLW format.
        Args:
            path: Path to the data directory
            scanlist: List of scan numbers
            hklw_file: Output filename
        Returns:
            List of DataFrames with HKLW data
        """
        if scanlist is None:
            raise ValueError("Scanlist cannot be None")
        
        hklw_df = self.taipan_batch_reduction(path, scanlist, 
                                             motorlist=['qh', 'qk', 'ql', 'en', 'detector'])
        
        super().tas_export_hklw(hklw_df, hklw_file)
        return hklw_df

    def export_scantitle(self, path: Union[str, Path] = '', 
                        datafrom_to: Optional[Tuple[int, int]] = None,
                        outputfile: Optional[str] = None) -> pd.DataFrame:
        """
        Export scan titles from a range of scans.
        Args:
            path: Path to the data directory
            datafrom_to: Tuple of (first_scan, last_scan)
            outputfile: Output HTML filename
        Returns:
            DataFrame containing scan information
        Raises:
            ValueError: If inputs are invalid
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        ppath = Path(path)
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if datafrom_to is None:
            raise ValueError("No data from and to number is given")
        
        if len(datafrom_to) != 2:
            raise ValueError("datafrom_to must be a tuple of two integers")
        
        if outputfile is None:
            outputfile = f"TAIPAN_exp{self.expnum}_scanlist.html"
        
        try:
            firstno = int(datafrom_to[0])
            lastno = int(datafrom_to[1])
        except (ValueError, TypeError):
            raise ValueError("datafrom_to must contain valid integers")
        
        if firstno > lastno:
            raise ValueError("First scan number must be less than or equal to last scan number")
        
        fileext = ".dat"
        filenolist = np.arange(firstno, lastno + 1)
        
        scanno_list = []
        command_list = []
        scantitle_list = []
        fileinitial = f"TAIPAN_exp{self.expnum}_scan"
        
        for fileno in filenolist:
            fullfilename = f"{fileinitial}{int(fileno)}{fileext}"
            fullpath = ppath / fullfilename
            
            scanno_list.append(int(fileno))
            
            if not fullpath.exists():
                command_list.append("file-not-found")
                scantitle_list.append("file-not-found")
                print(f"Warning: File not found: {fullpath}")
                continue
            
            try:
                with fullpath.open('r', encoding='utf-8') as f:
                    totallines = list(f)
                    command_found = False
                    title_found = False
                    
                    for line in totallines:
                        if "# command =" in line:
                            command_list.append(line[11:-1])
                            command_found = True
                        if "# scan_title =" in line:
                            scantitle_list.append(line[14:-1])
                            title_found = True
                        if command_found and title_found:
                            break
                    
                    if not command_found:
                        command_list.append("no-command")
                    if not title_found:
                        scantitle_list.append("no-title")
                        
            except IOError as e:
                print(f"Warning: Couldn't open file {fullpath}: {e}")
                command_list.append("error")
                scantitle_list.append("error")
        
        scan_dict = {"scanno": scanno_list, "command": command_list, "scantitle": scantitle_list}
        scanlist = pd.DataFrame(scan_dict)
        
        try:
            scanlist.to_html(ppath / outputfile)
        except Exception as e:
            print(f"Warning: Could not write HTML file: {e}")
        
        return scanlist

    def export_scanlog(self, path: Union[str, Path] = '', 
                      logfile: Optional[str] = None,
                      outputfile: Optional[str] = None) -> pd.DataFrame:
        """
        Export scan information from a log file.
        Args:
            path: Path to the data directory
            logfile: Log filename
            outputfile: Output HTML filename
        Returns:
            DataFrame containing scan information
        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If log file doesn't exist
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        ppath = Path(path)
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if logfile is None:
            raise ValueError("No logfile name is given")
        
        if outputfile is None:
            outputfile = f"TAIPAN_exp{self.expnum}_scanlist.html"
        
        fullfilepath = ppath / logfile
        
        if not fullfilepath.exists():
            raise FileNotFoundError(f"Log file not found: {fullfilepath}")
        
        hdf_filelist = []
        command_list = []
        
        try:
            with fullfilepath.open('r', encoding='utf-8') as f:
                total_line_no = sum(1 for _ in f)
            
            cur_line = total_line_no
            
            while cur_line > 0:
                scancmdfound = 0
                cur_line = cur_line - 1
                
                try:
                    line = linecache.getline(str(fullfilepath), cur_line)
                except Exception as e:
                    print(f"Warning: Error reading line {cur_line}: {e}")
                    continue
                
                if "nx.hdf updated" in line and len(line) > 32:
                    hdf_filelist.append(line[22:32])
                    
                    while cur_line > 0 and scancmdfound == 0:
                        cur_line = cur_line - 1
                        line = linecache.getline(str(fullfilepath), cur_line)
                        
                        if "Scanvar:" in line:
                            lines_to_check = []
                            for i in range(1, 4):
                                if cur_line - i > 0:
                                    lines_to_check.append(linecache.getline(str(fullfilepath), cur_line - i))
                            
                            cur_scan = None
                            for check_line in lines_to_check:
                                if len(check_line) > 32:
                                    if "runscan" in check_line[22:32] or "mscan" in check_line[22:32]:
                                        cur_scan = check_line[22:-1]
                                        break
                            
                            if cur_scan is None:
                                cur_scan = line[22:76] if len(line) > 76 else line[22:]
                                for check_line in lines_to_check:
                                    if "Scanvar:" in check_line and len(check_line) > 76:
                                        cur_scan = cur_scan + check_line[22:76]
                            
                            command_list.append(cur_scan)
                            scancmdfound = 1
            
            linecache.clearcache()
            
            if len(hdf_filelist) != len(command_list):
                print(f"Warning: Mismatch between HDF files ({len(hdf_filelist)}) and commands ({len(command_list)})")
            
            scan_dict = {"scanhdf_name": hdf_filelist, "command": command_list}
            scanlist = pd.DataFrame(scan_dict)
            scanlist = scanlist[::-1].reset_index(drop=True)
            
            try:
                outputpath = ppath / outputfile
                scanlist.to_html(outputpath)
            except Exception as e:
                print(f"Warning: Could not write HTML file: {e}")
            
            return scanlist
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {fullfilepath}")
        except IOError as e:
            raise IOError(f"Could not read file: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")

    def export_scanlist(self, path: Union[str, Path] = '', 
                       logfile: Optional[str] = None,
                       outputfile: Optional[str] = None) -> pd.DataFrame:
        """
        Export comprehensive scan list combining log file and data files.
        Args:
            path: Path to the data directory
            logfile: Log filename
            outputfile: Output HTML filename
        Returns:
            DataFrame containing combined scan information
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        ppath = Path(path)
        if not ppath.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        if outputfile is None:
            outputfile = f"TAIPAN_exp{self.expnum}_scanlist.html"
        
        try:
            scanlist = self.export_scanlog(path=path, logfile=logfile, outputfile=outputfile)
        except Exception as e:
            print(f"Error reading log file: {e}")
            return pd.DataFrame(columns=['scan_no', 'command', 'scantitle'])
        
        if scanlist is None or scanlist.empty:
            print("ERROR: No information from the logfile.")
            return pd.DataFrame(columns=['scan_no', 'command', 'scantitle'])
        
        scanno_list = []
        command_list = []
        scantitle_list = []
        
        for hdfname in scanlist['scanhdf_name']:
            try:
                scan_number = int(hdfname[-7:])
                scanno_list.append(scan_number)
                
                filename = f"TAIPAN_exp{self.expnum}_scan{scan_number}.dat"
                datafilepath = ppath / "Datafiles" / filename
                
                if datafilepath.is_file():
                    try:
                        with datafilepath.open('r', encoding='utf-8') as f:
                            totallines = list(f)
                            cmd_found = False
                            title_found = False
                            
                            for line in totallines:
                                if "# command =" in line:
                                    command_list.append(line[11:-1])
                                    cmd_found = True
                                if "# scan_title =" in line:
                                    scantitle_list.append(line[14:-1])
                                    title_found = True
                                if cmd_found and title_found:
                                    break
                            
                            if not cmd_found:
                                command_list.append("no-command")
                            if not title_found:
                                scantitle_list.append("no-title")
                                
                    except IOError as e:
                        print(f"Warning: Couldn't open file {datafilepath}: {e}")
                        command_list.append("error")
                        scantitle_list.append("error")
                else:
                    command_list.append("no-file")
                    scantitle_list.append("no-file")
                    print(f"Warning: File does not exist: {datafilepath}")
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse HDF name {hdfname}: {e}")
                scanno_list.append(-1)
                command_list.append("error")
                scantitle_list.append("error")
        
        scan_dict = {"scan_no": scanno_list, "command_dat": command_list, "scantitle": scantitle_list}
        scantitlelist = pd.DataFrame(scan_dict)
        
        fulllist = pd.merge(scanlist, scantitlelist, how='outer', 
                           left_index=True, right_index=True, suffixes=('', ''))
        fulllist = fulllist[['scan_no', 'command', 'scantitle']]
        
        try:
            outputpath = ppath / outputfile
            fulllist.to_html(outputpath)
        except Exception as e:
            print(f"Warning: Could not write HTML file: {e}")
        
        return fulllist

    def taipan_calibr_6scans(self, path: Union[str, Path] = '', 
                            scanlist: Optional[List[int]] = None) -> Tuple:
        """
        Calibrate using 6 nickel scans.
        Args:
            path: Path to the data directory
            scanlist: List of 6 scan numbers
        Returns:
            Tuple of (params_df, fitted_data, calibr_result, fig, ax)
        Raises:
            ValueError: If scanlist is invalid
        """
        if scanlist is None:
            raise ValueError("Scanlist cannot be None")
        
        if len(scanlist) != 6:
            raise ValueError(f"Expected 6 scans for calibration, got {len(scanlist)}")
        
        niscan_dflist = self.taipan_batch_reduction(path, scanlist)
        
        if len(niscan_dflist) != 6:
            raise ValueError(f"Could not load all 6 scans. Only {len(niscan_dflist)} loaded successfully")
        
        fitted_data = pd.DataFrame([])
        params_df = pd.DataFrame(columns=['A', 'A_err', 'w', 'w_err', 'x0', 'x0_err', 'bg', 'bg_err'])
        
        fig, ax = plt.subplots(1, 1)
        
        for ii in range(len(scanlist)):
            if niscan_dflist[ii].empty:
                raise ValueError(f"Scan {scanlist[ii]} has no data")
            
            if 'scanax1' not in niscan_dflist[ii].attrs:
                raise ValueError(f"Scan {scanlist[ii]} missing scan axis information")
            
            col_x_title = niscan_dflist[ii].attrs['scanax1']
            col_y_title = "detector"
            
            if col_x_title not in niscan_dflist[ii].columns:
                raise ValueError(f"Scan {scanlist[ii]}: Column '{col_x_title}' not found")
            if col_y_title not in niscan_dflist[ii].columns:
                raise ValueError(f"Scan {scanlist[ii]}: Column '{col_y_title}' not found")
            
            dataX = niscan_dflist[ii][col_x_title].to_numpy()
            dataY = niscan_dflist[ii][col_y_title].to_numpy()
            
            dataX = dataX[np.logical_not(np.isnan(dataX))]
            dataY = dataY[np.logical_not(np.isnan(dataY))]
            
            if len(dataX) == 0 or len(dataY) == 0:
                raise ValueError(f"Scan {scanlist[ii]} has no valid data points")
            
            try:
                cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G')
            except Exception as e:
                raise RuntimeError(f"Failed to fit scan {scanlist[ii]}: {e}")
            
            plt.plot(dataX, dataY, 'o', cur_fitdat['X'], cur_fitdat['Y_fit'], '-')
            
            params_df = pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)
            fitted_data = pd.merge(fitted_data, cur_fitdat, how='outer', 
                                  left_index=True, right_index=True, 
                                  suffixes=("", "_" + str(ii)))
        
        params_df = params_df.sort_values(by=['x0'], ascending=False)
        ni_peakpos = params_df['x0'].to_numpy()
        
        try:
            calibr_result = self.calibr_fit_offset(ni_peakpos)
        except Exception as e:
            raise RuntimeError(f"Calibration fit failed: {e}")
        
        return params_df, fitted_data, calibr_result, fig, ax

    def calibr_fit_offset(self, peaks: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fit calibration offset from peak positions.
        Args:
            peaks: Array of peak positions
        Returns:
            Dictionary containing calibration results
        Raises:
            ValueError: If peaks is None or invalid
        """
        if peaks is None:
            raise ValueError("No peaks provided for fit")
        
        peaks = np.asarray(peaks).flatten()
        
        if len(peaks) != 6:
            raise ValueError(f"Expected 6 peaks for calibration, got {len(peaks)}")
        
        if np.any(np.isnan(peaks)) or np.any(np.isinf(peaks)):
            raise ValueError("Peaks contain invalid values (NaN or Inf)")
        
        d_pg002 = 3.355
        dataX = np.array([1, 2, 3, 4, 5, 6])
        dataY = peaks
        
        fit_params = Parameters()
        fit_params.add('s2_offset', value=0.01)
        fit_params.add('wavelen', value=2.345)
        
        try:
            out = minimize(ni_s2_residual, fit_params, args=(dataX, dataY), 
                          method='Levenberg-Marquardt')
        except Exception as e:
            raise RuntimeError(f"Minimization failed: {e}")
        
        if out.params['wavelen'].stderr is None or out.params['s2_offset'].stderr is None:
            print("Warning: Fit uncertainties could not be determined")
        
        m1 = np.arcsin(out.params['wavelen'].value / 2.0 / d_pg002) * 180 / np.pi
        
        calibr_result = {
            "m1": m1,
            "m2": 2 * m1,
            "s2_offset": out.params['s2_offset'].value,
            "wavelen": out.params['wavelen'].value,
        }
        
        print(f"\n\nCalibration results:")
        print(f"m1: {m1:.8f}")
        print(f"m2: {2*m1:.8f}")
        print(f"s2 offset: {out.params['s2_offset'].value:.8f}")
        
        return calibr_result

    def taipan_expconfig(self, ef: float = 14.87, aa: float = 4, bb: float = 5, 
                        cc: float = 6, ang_a: float = 90, ang_b: float = 90, 
                        ang_c: float = 90, uu: np.ndarray = None, 
                        vv: np.ndarray = None):
        """
        Configure Taipan experiment parameters.
        Args:
            ef: Final energy
            aa, bb, cc: Lattice constants
            ang_a, ang_b, ang_c: Lattice angles
            uu, vv: Orientation vectors
        Returns:
            Configured experiment object
        """
        if uu is None:
            uu = np.array([1, 0, 0])
        if vv is None:
            vv = np.array([0, 0, 1])
        
        uu = np.asarray(uu)
        vv = np.asarray(vv)
        
        if uu.shape != (3,) or vv.shape != (3,):
            raise ValueError("Orientation vectors uu and vv must have shape (3,)")
        
        if ef <= 0:
            raise ValueError(f"Final energy must be positive, got {ef}")
        
        if aa <= 0 or bb <= 0 or cc <= 0:
            raise ValueError("Lattice constants must be positive")
        
        if not (0 < ang_a < 180 and 0 < ang_b < 180 and 0 < ang_c < 180):
            raise ValueError("Lattice angles must be between 0 and 180 degrees")
        
        taipan_exp = super().tas_expconfig(ef)
        taipan_exp.method = 1  # 1 for Popovici, 0 for Cooper-Nathans
        taipan_exp.moncor = 1
        taipan_exp.efixed = ef
        taipan_exp.infin = -1  # const-Ef
        
        taipan_exp.mono.dir = -1
        taipan_exp.mono.dir = 1
        taipan_exp.ana.dir = -1
        
        taipan_exp.sample.a = aa
        taipan_exp.sample.b = bb
        taipan_exp.sample.c = cc
        taipan_exp.sample.alpha = ang_a
        taipan_exp.sample.beta = ang_b
        taipan_exp.sample.gamma = ang_c
        taipan_exp.sample.u = uu
        taipan_exp.sample.v = vv
        
        taipan_exp.arms = [202, 196, 179, 65, 106]
        taipan_exp.orient1 = uu
        taipan_exp.orient2 = vv
        
        self.exp = taipan_exp
        return taipan_exp

    def taipan_conv_init(self, hklw: Optional[pd.DataFrame] = None, 
                        exp=None, initial: Optional[Dict] = None,
                        fixlist: List[int] = None, magion: str = "Mn2",
                        sqw=SqwDemo, pref=PrefDemo, smoothfit: bool = True):
        """
        Initialize convolution fitting.
        Args:
            hklw: DataFrame with H, K, L, W, Iobs columns
            exp: Experiment configuration
            initial: Initial parameters dictionary
            fixlist: List indicating which parameters to fix
            magion: Magnetic ion type
            sqw: Scattering function
            pref: Preference function
            smoothfit: Whether to use smooth fitting
        Returns:
            Initial simulation result
        Raises:
            ValueError: If required inputs are missing
        """
        if hklw is None:
            raise ValueError("hklw cannot be None")
        
        if hklw.empty:
            raise ValueError("hklw DataFrame is empty")
        
        if exp is None:
            exp = self.taipan_expconfig(ef=14.87)
        
        if initial is None:
            raise ValueError("initial parameters cannot be None")
        
        if fixlist is None:
            fixlist = [0, 0, 0, 0, 0, 0, 0, 0]
        
        if not isinstance(initial, dict):
            raise ValueError("initial must be a dictionary")
        
        if len(fixlist) != len(initial):
            raise ValueError(f"fixlist length ({len(fixlist)}) must match initial length ({len(initial)})")
        
        required_cols = ['qh', 'qk', 'ql', 'en', 'detector']
        missing_cols = [col for col in required_cols if col not in hklw.columns]
        if missing_cols:
            # Try alternative column names
            alt_map = {'qh': 'h', 'qk': 'k', 'ql': 'l', 'en': 'e'}
            for orig, alt in alt_map.items():
                if orig in missing_cols and alt in hklw.columns:
                    hklw = hklw.rename(columns={alt: orig})
                    missing_cols.remove(orig)
        
        if missing_cols:
            raise ValueError(f"hklw missing required columns: {missing_cols}")
        
        try:
            [H, K, L, W, Iobs] = hklw[['qh', 'qk', 'ql', 'en', 'detector']].to_numpy().T
        except Exception as e:
            raise ValueError(f"Error extracting HKLW data: {e}")
        
        dIobs = np.sqrt(Iobs)
        
        ffactor = SelFormFactor(magion)
        if ffactor is None:
            ffactor = SelFormFactor("Mn2")
            print(f"Warning: Magnetic ion '{magion}' not found. Using Mn2 instead.")
        
        try:
            AA = ffactor["AA"]
            aa = ffactor["aa"]
            BB = ffactor["BB"]
            bb = ffactor["bb"]
            CC = ffactor["CC"]
            cc = ffactor["cc"]
            DD = ffactor["DD"]
        except KeyError as e:
            raise ValueError(f"Form factor missing required key: {e}")
        
        initial_new = list(initial.values()) + [AA, aa, BB, bb, CC, cc, DD]
        fixlist_new = fixlist + [0, 0, 0, 0, 0, 0, 0]
        
        try:
            sim_initial = exp.ResConv(sqw=SqwDemo, pref=PrefDemo, nargout=2,
                                     hkle=[H, K, L, W], METHOD='fix',
                                     ACCURACY=[5, 5], p=initial_new)
        except Exception as e:
            raise RuntimeError(f"ResConv initialization failed: {e}")
        
        return sim_initial

    def taipan_convfit(self, hklw: Optional[pd.DataFrame] = None, exp=None,
                      initial: Optional[List] = None, 
                      fixlist: List[int] = None,
                      magion: str = "Mn2", sqw=SqwDemo, pref=PrefDemo):
        """
        Perform convolution fitting on data.
        Args:
            hklw: DataFrame with H, K, L, W, Iobs columns
            exp: Experiment configuration
            initial: Initial parameters list
            fixlist: List indicating which parameters to fix
            magion: Magnetic ion type
            sqw: Scattering function
            pref: Preference function
        Returns:
            Tuple of (final_params, newHKLW)
        Raises:
            ValueError: If required inputs are missing
        """
        if hklw is None:
            raise ValueError("hklw cannot be None")
        
        if hklw.empty:
            raise ValueError("hklw DataFrame is empty")
        
        if exp is None:
            exp = self.taipan_expconfig(ef=14.87)
        
        if initial is None:
            raise ValueError("initial parameters cannot be None")
        
        if fixlist is None:
            fixlist = [0, 0, 0, 0, 0, 0, 0, 0]
        
        required_cols = ['qh', 'qk', 'ql', 'en', 'detector']
        missing_cols = [col for col in required_cols if col not in hklw.columns]
        if missing_cols:
            alt_map = {'qh': 'h', 'qk': 'k', 'ql': 'l', 'en': 'e'}
            for orig, alt in alt_map.items():
                if orig in missing_cols and alt in hklw.columns:
                    hklw = hklw.rename(columns={alt: orig})
                    missing_cols.remove(orig)
        
        if missing_cols:
            raise ValueError(f"hklw missing required columns: {missing_cols}")
        
        try:
            [H, K, L, W, Iobs] = hklw[['qh', 'qk', 'ql', 'en', 'detector']].to_numpy().T
        except Exception as e:
            raise ValueError(f"Error extracting HKLW data: {e}")
        
        dIobs = np.sqrt(Iobs)
        
        ffactor = SelFormFactor(magion)
        if ffactor is None:
            ffactor = SelFormFactor("Mn2")
            print(f"Warning: Magnetic ion '{magion}' not found. Using Mn2 instead.")
        
        try:
            AA = ffactor["AA"]
            aa = ffactor["aa"]
            BB = ffactor["BB"]
            bb = ffactor["bb"]
            CC = ffactor["CC"]
            cc = ffactor["cc"]
            DD = ffactor["DD"]
        except KeyError as e:
            raise ValueError(f"Form factor missing required key: {e}")
        
        initial_new = list(initial) + [AA, aa, BB, bb, CC, cc, DD]
        fixlist_new = fixlist + [0, 0, 0, 0, 0, 0, 0]
        
        print("Initial parameters:", initial_new)
        
        try:
            fitter = UltraFastFitConv(exp, SqwDemo, PrefDemo, [H, K, L, W], Iobs, dIobs)
            result = fitter.fit_ultrafast(param_initial=initial_new,
                                         param_fixed_mask=fixlist_new,
                                         maxfev=200,
                                         use_analytical_jacobian=True,
                                         early_stopping=True,
                                         verbose=True)
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {e}")
        
        final_params = result['params']
        param_errors = result['param_errors']
        chi2_reduced = result['chi2_reduced']
        model_fit = result['model']
        
        newH = np.linspace(H[0], H[-1], 101)
        newK = np.linspace(K[0], K[-1], 101)
        newL = np.linspace(L[0], L[-1], 101)
        newW = np.linspace(W[0], W[-1], 101)
        
        try:
            final = exp.ResConv(sqw=SqwDemo, pref=PrefDemo, nargout=2,
                              hkle=[newH, newK, newL, newW],
                              METHOD='fix', ACCURACY=None, p=final_params)
        except Exception as e:
            raise RuntimeError(f"Final ResConv calculation failed: {e}")
        
        par_output = "The fitted parameters:\n"
        par_output += f"En1  :\t{final_params[0]:8f}  \t{param_errors[0]:8f}\n"
        par_output += f"En2  :\t{final_params[1]:8f}  \t{param_errors[1]:8f}\n"
        par_output += f"Int1 :\t{final_params[2]*final_params[5]:8f}  \t{final_params[2]*param_errors[5]:8f}\n"
        par_output += f"Int2 :\t{final_params[5]:8f}  \t{param_errors[5]:8f}\n"
        par_output += f"FWHM1:\t{final_params[3]:8f}  \t{param_errors[3]:8f}\n"
        par_output += f"FWHM2:\t{final_params[4]:8f}  \t{param_errors[4]:8f}\n"
        par_output += f"bg   :\t{final_params[6]:8f}  \t{param_errors[6]:8f}\n"
        par_output += f"temp :\t{final_params[7]:8f}  \t{param_errors[7]:8f}\n"
        
        print(par_output)
        
        oldHKLW = np.column_stack([H, K, L, W, Iobs, dIobs])
        newHKLW = np.column_stack([newH, newK, newL, newW, final])
        
        return final_params, newHKLW

    def taipan_phonon_convfit(self, hklw: Optional[pd.DataFrame] = None, exp=None,
                             initial: Optional[Dict] = None,
                             fixlist: List[int] = None,
                             sqw=SqwDemo, pref=PrefPhononDemo):
        """
        Perform phonon convolution fitting.
        Args:
            hklw: DataFrame with H, K, L, W, Iobs columns
            exp: Experiment configuration
            initial: Initial parameters dictionary
            fixlist: List indicating which parameters to fix
            sqw: Scattering function
            pref: Preference function for phonons
        Returns:
            Tuple of (final_param, data_and_fit)
        Raises:
            ValueError: If required inputs are missing
        """
        if hklw is None:
            raise ValueError("No data provided to fit")
        
        if hklw.empty:
            raise ValueError("hklw DataFrame is empty")
        
        if exp is None:
            exp = self.taipan_expconfig(ef=14.87)
        
        if initial is None:
            raise ValueError("initial parameters cannot be None")
        
        if not isinstance(initial, dict):
            raise ValueError("initial must be a dictionary")
        
        if fixlist is None:
            fixlist = [0] * len(initial)
        
        if len(fixlist) != len(initial):
            raise ValueError(f"fixlist length ({len(fixlist)}) must match initial length ({len(initial)})")
        
        required_cols = ['qh', 'qk', 'ql', 'en', 'detector']
        missing_cols = [col for col in required_cols if col not in hklw.columns]
        if missing_cols:
            alt_map = {'qh': 'h', 'qk': 'k', 'ql': 'l', 'en': 'e'}
            for orig, alt in alt_map.items():
                if orig in missing_cols and alt in hklw.columns:
                    hklw = hklw.rename(columns={alt: orig})
                    missing_cols.remove(orig)
        
        if missing_cols:
            raise ValueError(f"hklw missing required columns: {missing_cols}")
        
        try:
            [H, K, L, W, Iobs] = hklw[['qh', 'qk', 'ql', 'en', 'detector']].to_numpy().T
        except Exception as e:
            raise ValueError(f"Error extracting HKLW data: {e}")
        
        dIobs = np.sqrt(Iobs)
        initialnew = list(initial.values())
        
        try:
            fitter = FitConv(exp, sqw, pref, [H, K, L, W], Iobs, dIobs)
            [final_param, dpa, chisqN, sim, CN, PQ, nit, kvg, details] = \
                fitter.fitwithconv(exp, sqw, pref, [H, K, L, W], Iobs, dIobs,
                                 param=initialnew, paramfixed=fixlist)
        except Exception as e:
            raise RuntimeError(f"Phonon fitting failed: {e}")
        
        str_output = "The fitted parameters:\n"
        parlist = list(initial.keys())
        
        for index, (iname, ipar, ierr) in enumerate(zip(parlist, final_param, dpa)):
            str_output += f"P{index}( {iname} ): {ipar:6.8f} +/- {ierr:6.8f}\n"
        
        print(str_output)
        
        data_and_fit = np.vstack((H, K, L, W, Iobs, dIobs, sim))
        return final_param, data_and_fit

    def taipan_batch_validate_new(self, filename: Optional[str] = None,
                                  bsim: bool = False, exp=None) -> List[Tuple]:
        """
        Validate commands from a batch file with optional simulation.
        Args:
            filename: Path to batch file
            bsim: Whether to simulate scans
            exp: Experiment configuration for simulation
        Returns:
            List of tuples: (line_number, command, is_valid, message)
        Raises:
            ValueError: If filename is None
            FileNotFoundError: If file doesn't exist
        """
        if filename is None:
            raise ValueError("Please provide a batch file")
        
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Batch file not found: {filename}")
        
        results = []
        validator = TaipanCommandValidator()
        
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines()]
        except Exception as e:
            raise IOError(f"Error reading file: {e}")
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if not line:
                i += 1
                continue
            
            if line.startswith('mscan'):
                is_valid, message = validator.validate_command(line)
                if is_valid:
                    print(f"Line {i+1}: {line}\nValidation Passed.\n")
                    results.append((i + 1, line, is_valid, message))
                    if bsim:
                        if exp is None:
                            print("Warning: No experiment provided for simulation.")
                        else:
                            try:
                                simres = self.taipan_scansim(line, exp=exp)
                                print(simres)
                            except Exception as e:
                                print(f"Simulation error: {e}")
                else:
                    print(f"Line {i+1}: Invalid mscan command\n")
                    results.append((i + 1, line, is_valid, message))
                i += 1
                
            elif line.startswith('drive'):
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('runscan'):
                    combined_cmd = f"{line}\n{lines[i + 1]}"
                    is_valid, message = validator.validate_command(combined_cmd)
                    if is_valid:
                        print(f"Line {i+1} and {i+2}: {combined_cmd}\nValidation Passed.\n")
                        results.append((i + 1, combined_cmd, is_valid, message))
                        if bsim:
                            if exp is None:
                                print("Warning: No experiment provided for simulation.")
                            else:
                                try:
                                    simres = self.taipan_scansim(combined_cmd, exp=exp)
                                    print(simres)
                                except Exception as e:
                                    print(f"Simulation error: {e}")
                        i += 2
                    else:
                        print(f"Line {i+2}: Validation Failure: Invalid runscan.\n")
                        results.append((i + 1, combined_cmd, is_valid, message))
                        i += 1
                else:
                    is_valid, message = validator.validate_command(line)
                    results.append((i + 1, line, is_valid, message))
                    print(f"Line {i+1}: {line}\nSingle drive command.\n")
                    i += 1
            else:
                results.append((i + 1, line, False, "Invalid scan command"))
                print(f"Line {i+1}: {line}\nValidation False. Invalid scan command.\n")
                i += 1
        
        return results

    def taipan_batch_validate(self, batchfile=None):
        """
        Validate commands from a batch file.
        
        Args:
            batchfile: Path to the batch file to validate
            
        Returns:
            tuple: (validation_result_string, valid_cmd_list)
            
        Raises:
            ValueError: If batchfile is None
            FileNotFoundError: If file doesn't exist
        """
        # Input validation
        if batchfile is None:
            raise ValueError("Batch file cannot be None. Please provide a valid file path.")
        
        # Convert to Path object for better path handling
        from pathlib import Path
        filepath = Path(batchfile)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Batch file not found: {batchfile}")
        
        if not filepath.is_file():
            raise ValueError(f"Path is not a file: {batchfile}")
        
        valid_cmd_list = []
        validation_result = ""
        validator = TaipanCommandValidator()
        
        try:
            # Read file with explicit encoding
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError as e:
            raise IOError(f"File encoding error. Please ensure file is UTF-8 encoded: {e}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {batchfile}")
        except Exception as e:
            raise IOError(f"Error reading file: {e}")
        
        # Check if file is empty
        if not lines:
            print("Warning: Batch file is empty")
            return validation_result, valid_cmd_list
        
        # Process lines
        i = 0
        line_count = len(lines)
        
        while i < line_count:
            # Skip empty or whitespace-only lines
            if not lines[i].strip():
                i += 1
                continue
            
            current_line = lines[i].strip()
            print(f"\nProcessing line {i + 1}:")
            validation_result += f"\nLine {i + 1}:\n"
            
            # Validate mscan command
            if current_line.startswith('mscan'):
                try:
                    valid, message = validator.validate_command(current_line)
                    print(f"Command: {current_line}")
                    
                    if valid:
                        print("Validation passed!")
                        validation_result += f"Command: {current_line}\nValidation passed!\n\n"
                        valid_cmd_list.append(current_line)
                    else:
                        print(f"Error: mscan validation failed - {message}")
                        validation_result += f"Command: {current_line}\nError: {message}\n\n"
                except Exception as e:
                    print(f"Error: Exception during mscan validation - {e}")
                    validation_result += f"Command: {current_line}\nError: Validation exception - {e}\n\n"
                
                i += 1
            
            # Validate drive command (with or without runscan)
            elif current_line.startswith('drive'):
                # Safely check for next line
                next_line = ""
                if i + 1 < line_count:
                    next_line = lines[i + 1].strip()
                
                # Check if this is a drive+runscan combination
                if next_line.startswith('runscan'):
                    combined_cmd = f"{current_line}\n{next_line}"
                    print(f"Combined command:\n{combined_cmd}")
                    
                    try:
                        valid, message = validator.validate_command(combined_cmd)
                        
                        if valid:
                            print("Validation passed!")
                            validation_result += f"Combined command:\n{combined_cmd}\nValidation passed!\n\n"
                            valid_cmd_list.append(current_line)
                            valid_cmd_list.append(next_line)
                        else:
                            print(f"Error: drive+runscan validation failed - {message}")
                            validation_result += f"Combined command:\n{combined_cmd}\nError: {message}\n\n"
                    except Exception as e:
                        print(f"Error: Exception during drive+runscan validation - {e}")
                        validation_result += f"Combined command:\n{combined_cmd}\nError: Validation exception - {e}\n\n"
                    
                    i += 2  # Skip both lines
                else:
                    # Process single drive command
                    print(f"Command: {current_line}")
                    
                    try:
                        valid, message = validator.validate_command(current_line)
                        
                        if valid:
                            print("Validation passed!")
                            validation_result += f"Command: {current_line}\nValidation passed!\n\n"
                            valid_cmd_list.append(current_line)
                        else:
                            print(f"Error: drive validation failed - {message}")
                            validation_result += f"Command: {current_line}\nError: {message}\n\n"
                    except Exception as e:
                        print(f"Error: Exception during drive validation - {e}")
                        validation_result += f"Command: {current_line}\nError: Validation exception - {e}\n\n"
                    
                    i += 1
            
            # Validate title command
            elif current_line.startswith('title'):
                print(f"Command: {current_line}")
                
                try:
                    valid, message = validator.validate_command(current_line)
                    
                    if valid:
                        print("Validation passed!")
                        validation_result += f"Command: {current_line}\nValidation passed!\n\n"
                        valid_cmd_list.append(current_line)
                    else:
                        print(f"Error: title validation failed - {message}")
                        validation_result += f"Command: {current_line}\nError: {message}\n\n"
                except Exception as e:
                    print(f"Error: Exception during title validation - {e}")
                    validation_result += f"Command: {current_line}\nError: Validation exception - {e}\n\n"
                
                i += 1
            
            # Handle unknown commands
            else:
                print(f"Command: {current_line}")
                print("ERROR: Unknown command type!")
                validation_result += f"Command: {current_line}\nERROR: Unknown command type!\n\n"
                i += 1
        
        # Summary statistics
        total_commands = len(valid_cmd_list)
        print(f"\n{'='*60}")
        print(f"Validation Summary:")
        print(f"Total valid commands: {total_commands}")
        print(f"{'='*60}\n")
        
        return validation_result, valid_cmd_list
        
        



    def taipan_scansim(self, cmdline="", exp=None, runscanpos=[2, 0, 0, 0]):
        """
        Simulate a scan command and return the scan positions.
        
        Args:
            cmdline: Command line string (drive+runscan or mscan)
            exp: Experiment configuration object
            runscanpos: Default runscan positions (not currently used)
            
        Returns:
            pd.DataFrame: DataFrame with simulated scan positions, or None on error
        """
        pd.set_option('display.expand_frame_repr', False)
        
        # Input validation
        if not cmdline:
            raise ValueError("Command line cannot be empty")
        
        if exp is None:
            raise ValueError("Experiment configuration is required")
        
        if not isinstance(cmdline, str):
            raise TypeError(f"cmdline must be a string, got {type(cmdline)}")
        
        # Parse command line
        try:
            cmd_lines = cmdline.splitlines()
        except Exception as e:
            raise ValueError(f"Error parsing command line: {e}")
        
        if not cmd_lines:
            raise ValueError("Command line is empty after parsing")
        
        cmd_items = cmd_lines[0].split()
        if not cmd_items:
            raise ValueError("No command found in command line")
        
        print(cmdline)
        
        command_type = cmd_items[0]
        
        # Handle drive command
        if command_type == 'drive':
            if len(cmd_lines) < 2:
                raise ValueError("Drive command found but no runscan follows")
            
            try:
                sim_result = self.taipan_runscansim(cmd_lines[0], cmd_lines[1], exp)
                return sim_result
            except Exception as e:
                raise RuntimeError(f"Error simulating drive+runscan: {e}")
        
        # Handle mscan command
        elif command_type == 'mscan':
            try:
                sim_result = self.taipan_mscansim(cmd_lines[0], exp)
                return sim_result
            except Exception as e:
                raise RuntimeError(f"Error simulating mscan: {e}")
        
        else:
            raise ValueError(f"Unknown command type: '{command_type}'. Expected 'drive' or 'mscan'")


    def taipan_mscansim(self, mscanline="", exp=None):
        """
        Simulate an mscan command and calculate scan positions.
        
        Args:
            mscanline: mscan command string
            exp: Experiment configuration object
            
        Returns:
            pd.DataFrame: DataFrame with scan positions and motor angles, or None on error
        """
        # Input validation
        if not exp:
            raise ValueError("Experiment configuration is required")
        
        if not mscanline:
            raise ValueError("mscan command line cannot be empty")
        
        if not isinstance(mscanline, str):
            raise TypeError(f"mscanline must be a string, got {type(mscanline)}")
        
        motorlist = ['m1', 'm2', 's1', 's2', 'a1', 'a2', 'qh', 'qk', 'ql', 'en']
        df_scanpos = pd.DataFrame([])
        
        # Validate command
        validator = TaipanCommandValidator()
        try:
            mscan_valid, message = validator.validate_command(mscanline)
        except Exception as e:
            raise RuntimeError(f"Error validating mscan command: {e}")
        
        if not mscan_valid:
            raise ValueError(f"Invalid mscan command: {message}")
        
        cmd_items = mscanline.split()
        
        if not cmd_items or cmd_items[0] != 'mscan':
            raise ValueError("Command must start with 'mscan'")
        
        # Check minimum parameters
        if len(cmd_items) < 7:
            raise ValueError(
                "Insufficient parameters. Format: mscan motor1 start1 step1 "
                "motor2 start2 step2 ... numsteps time/monitor count"
            )
        
        # Validate parameter count
        int_part, dec_part = divmod(len(cmd_items) - 4, 3)
        if dec_part > 0.000001:
            raise ValueError(
                f"Incorrect number of parameters. Expected format: "
                f"mscan [motor start step]... numsteps time/monitor count"
            )
        
        num_motors = int((len(cmd_items) - 4) / 3)
        
        # Validate motors and their parameters
        for ii in range(num_motors):
            motor_idx = 3 * ii + 1
            start_idx = 3 * ii + 2
            step_idx = 3 * ii + 3
            
            # Check array bounds
            if step_idx >= len(cmd_items):
                raise ValueError(f"Missing parameters for motor {ii + 1}")
            
            motor_name = cmd_items[motor_idx]
            start_val = cmd_items[start_idx]
            step_val = cmd_items[step_idx]
            
            # Validate motor name
            if motor_name not in motorlist:
                raise ValueError(
                    f"Motor '{motor_name}' is not valid. "
                    f"Valid motors: {', '.join(motorlist)}"
                )
            
            # Validate start position and step size
            if not strisfloat(start_val):
                raise ValueError(f"Motor {ii + 1} ({motor_name}): Invalid start position '{start_val}'")
            
            if not strisfloat(step_val):
                raise ValueError(f"Motor {ii + 1} ({motor_name}): Invalid step size '{step_val}'")
        
        # Validate number of steps
        if len(cmd_items) < 3:
            raise ValueError("Missing step count parameter")
        
        if not strisint(cmd_items[-3]):
            raise ValueError(f"Number of steps must be an integer, got '{cmd_items[-3]}'")
        
        numstep = int(cmd_items[-3])
        if numstep <= 0:
            raise ValueError(f"Number of steps must be positive, got {numstep}")
        
        # Validate count mode
        if len(cmd_items) < 2:
            raise ValueError("Missing count mode parameter")
        
        count_mode = cmd_items[-2]
        if count_mode not in ['time', 'monitor']:
            raise ValueError(f"Count mode must be 'time' or 'monitor', got '{count_mode}'")
        
        # Validate count value
        if len(cmd_items) < 1:
            raise ValueError("Missing count value parameter")
        
        if not strisfloat(cmd_items[-1]) and not strisint(cmd_items[-1]):
            raise ValueError(f"Count value must be numeric, got '{cmd_items[-1]}'")
        
        # Generate scan positions
        try:
            for ii in range(num_motors):
                motor_name = cmd_items[3 * ii + 1]
                start = float(cmd_items[3 * ii + 2])
                stepsize = float(cmd_items[3 * ii + 3])
                
                temppos = [start + jj * stepsize for jj in range(numstep + 1)]
                df_scanpos[motor_name] = temppos
        except Exception as e:
            raise RuntimeError(f"Error generating scan positions: {e}")
        
        # Calculate motor angles if HKLE scan
        required_cols = ['qh', 'qk', 'ql', 'en']
        if all(col in df_scanpos.columns for col in required_cols):
            try:
                hkle = df_scanpos[required_cols].to_numpy().T
                [M1, M2, S1, S2, A1, A2, Q] = exp.get_spec_angles(hkle)
                df_scanpos['m1'] = M1
                df_scanpos['m2'] = M2
                df_scanpos['s1'] = S1
                df_scanpos['s2'] = S2
                df_scanpos['a1'] = A1
                df_scanpos['a2'] = A2
            except Exception as e:
                print(f"Warning: Could not calculate motor angles: {e}")
                # Continue without motor angles rather than failing completely
        
        return df_scanpos


    def taipan_runscansim(self, driveline=None, scanline=None, exp=None):
        """
        Simulate a drive+runscan command by converting it to mscan format.
        
        Args:
            driveline: Drive command string
            scanline: Runscan command string  
            exp: Experiment configuration object
            
        Returns:
            pd.DataFrame: DataFrame with scan positions, or None on error
        """
        # Input validation
        if exp is None:
            raise ValueError("Experiment configuration is required")
        
        if not driveline:
            raise ValueError("Drive command line cannot be empty")
        
        if not scanline:
            raise ValueError("Runscan command line cannot be empty")
        
        if not isinstance(driveline, str):
            raise TypeError(f"driveline must be a string, got {type(driveline)}")
        
        if not isinstance(scanline, str):
            raise TypeError(f"scanline must be a string, got {type(scanline)}")
        
        motorlist = ['m1', 'm2', 's1', 's2', 'a1', 'a2', 'qh', 'qk', 'ql', 'en']
        
        # Validate combined command
        validator = TaipanCommandValidator()
        combined_cmd = driveline + '\n' + scanline
        
        try:
            runscan_valid, message = validator.validate_command(combined_cmd)
        except Exception as e:
            raise RuntimeError(f"Error validating drive+runscan command: {e}")
        
        if not runscan_valid:
            raise ValueError(f"Invalid drive+runscan command: {message}")
        
        # Parse command items
        drive_items = driveline.split()
        scan_items = scanline.split()
        
        if not drive_items:
            raise ValueError("Drive command is empty")
        
        if not scan_items:
            raise ValueError("Runscan command is empty")
        
        if drive_items[0] != 'drive':
            raise ValueError(f"Drive command must start with 'drive', got '{drive_items[0]}'")
        
        if scan_items[0] != 'runscan':
            raise ValueError(f"Scan command must start with 'runscan', got '{scan_items[0]}'")
        
        # Validate runscan has enough parameters
        if len(scan_items) < 7:
            raise ValueError(
                "Insufficient runscan parameters. Format: "
                "runscan motor start stop steps time/monitor count"
            )
        
        scan_motor = scan_items[1]
        
        # Check if scan motor is valid
        if scan_motor not in motorlist:
            raise ValueError(
                f"Scan motor '{scan_motor}' is not valid. "
                f"Valid motors: {', '.join(motorlist)}"
            )
        
        # Find scan motor in drive command
        try:
            motor_index = drive_items.index(scan_motor)
        except ValueError:
            raise ValueError(
                f"Scan motor '{scan_motor}' not found in drive command. "
                f"Available motors in drive: {drive_items[1::2]}"
            )
        
        # Validate scan parameters
        try:
            startpos = float(scan_items[2])
            stoppos = float(scan_items[3])
            stepno = float(scan_items[4])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid scan parameters: {e}")
        
        if stepno <= 0:
            raise ValueError(f"Step number must be positive, got {stepno}")
        
        if not isinstance(int(stepno), int) or stepno != int(stepno):
            print(f"Warning: Step number {stepno} is not an integer, converting to {int(stepno)}")
            stepno = int(stepno)
        
        # Build mscan command
        newcmd = "mscan"
        
        try:
            for index, item in enumerate(drive_items):
                # Process motor positions (odd indices after 'drive')
                if index % 2 == 1 and index < len(drive_items) - 1:
                    motor_name = drive_items[index]
                    motor_pos = drive_items[index + 1]
                    
                    # Validate motor name
                    if motor_name not in motorlist:
                        raise ValueError(f"Invalid motor '{motor_name}' in drive command")
                    
                    # Validate position is numeric
                    if not strisfloat(motor_pos):
                        raise ValueError(f"Invalid position '{motor_pos}' for motor '{motor_name}'")
                    
                    if index != motor_index:
                        # Fixed position motor
                        newcmd += f' {motor_name} {motor_pos} 0.0'
                    else:
                        # Scanned motor
                        stepsize = (stoppos - startpos) / stepno
                        newcmd += f' {motor_name} {startpos} {stepsize}'
            
            # Add step count and counting mode
            if len(scan_items) >= 7:
                newcmd += f' {int(stepno)} {scan_items[5]} {scan_items[6]}'
            else:
                raise ValueError("Missing count mode or count value in runscan command")
            
        except IndexError as e:
            raise ValueError(f"Error parsing drive command parameters: {e}")
        except Exception as e:
            raise RuntimeError(f"Error building mscan command: {e}")
        
        # Simulate the converted mscan command
        try:
            sim_result = self.taipan_mscansim(newcmd, exp)
            return sim_result
        except Exception as e:
            raise RuntimeError(f"Error simulating converted mscan command: {e}")





    def taipan_bosecorrect(self, path: str = '', scanlist=None, sample_T: float = 1.5):
        """
        Apply Bose factor correction to energy scans.

        Parameters
        ----------
        path : str
            Directory path containing scan data.
        scanlist : list
            List of scan identifiers.
        sample_T : float
            Sample temperature in Kelvin (must be > 0).

        Returns
        -------
        newdflist : list of pandas.DataFrame
            Corrected dataframes.
        scinf_list : list
            Corresponding scan information.
        """
        ppath = Path(path)

        if not ppath.is_dir():
            raise ValueError("Error: wrong path!") 

        if sample_T <= 0:
            raise ValueError('Error: sample temperature must be > 0 K.')

        dflist, scinf_list = self.taipan_reduction(path, scanlist)
        newdflist = []

        for df, scinf in zip(dflist, scinf_list):
            newdf = df.loc[df['e'] > 0].reset_index(drop=True)

            if scinf.get('scanax') == 'e':
                deltae = newdf['e'].to_numpy()
                counts = newdf['detector'].to_numpy()

                # Avoid division by zero or overflow
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    bose_factor = 1 + 1 / (1 - np.exp(-deltae / (sample_T * 0.08713)))
                    counts /= bose_factor

                newdf['detector'] = counts
            else:
                print("Warning: This is not an energy scan.")

            newdflist.append(newdf)

        return newdflist, scinf_list


    def taipan_gen_escans(self, hkl_list=[[0, 0, 1], [0, 0, 1.5]], e_range=[[0, 5], [0, 6]], 
                          e_step: float = 0.1, sample_T: float = 1.5, 
                          beammon: int = 100000, titleTemplate: str = 'ABO3 HH0-00L'):
        """
        Generate batch energy scans at a list of Q positions.
        """
        batch_str = ''

        if len(hkl_list) != len(e_range):
            raise ValueError(('Error: The number of Q positions and energy ranges must match!')

        for hkl, erange in zip(hkl_list, e_range):
            steps = 1 + round((erange[1] - erange[0]) / e_step)
            scantitle = (
                f'title "{titleTemplate} Q({hkl[0]}, {hkl[1]}, {hkl[2]}) '
                f'Escan ({erange[0]} ~ {erange[1]} meV) at {sample_T} K"\n'
            )
            drivestr = f'drive qh {hkl[0]} qk {hkl[1]} ql {hkl[2]} en {erange[0]}\n'
            scanstr = f'runscan en {erange[0]} {erange[1]} {steps} monitor {beammon}\n\n'
            batch_str += scantitle + drivestr + scanstr
            print(batch_str)

        return batch_str


    def taipan_gen_qscans(self, startQ=[[0, 0, 1.1], [0.1, 0, 1.5]], 
                          endQ=[[0, 0, 1.6], [0.9, 0, 1.5]], 
                          q_step=[[0.00, 0.00, 0.01], [0.01, 0.00, 0.00]], 
                          en: float = 2, sample_T: float = 1.5, 
                          beammon: int = 100000, titleTemplate: str = 'ABO3 HH0-00L'):
        """
        Generate batch Q scans from startQ to endQ with q_step size.
        """
        batch_str = ''
        qNameList = ['qh', 'qk', 'ql']

        if len(startQ) != len(endQ):
            raise ValueError(('Error: startQ and endQ lengths do not match!')

        for sQ, eQ, step in zip(startQ, endQ, q_step):
            deltaList = [abs(s - e) for s, e in zip(sQ, eQ)]
            scanQList = [i for i, d in enumerate(deltaList) if d >= 0.0001]
            midQ = [(s + e) / 2 for s, e in zip(sQ, eQ)]

            if not scanQList:
                print("Error: startQ and endQ are identical!")
                continue

            if len(scanQList) == 1:  # Single axis scan
                axis = scanQList[0]
                if step[axis] == 0:
                    print("Error: step cannot be zero!")
                    continue

                steps = 1 + round((eQ[axis] - sQ[axis]) / step[axis])
                scantitle = (
                    f'title "{titleTemplate} Qscan {qNameList[axis]}({sQ[axis]}~{eQ[axis]}) '
                    f'around Q({midQ[0]}, {midQ[1]}, {midQ[2]}) at E = {en} meV, {sample_T} K"\n'
                )
                drivestr = f'drive qh {sQ[0]} qk {sQ[1]} ql {sQ[2]} en {en}\n'
                scanstr = f'runscan {qNameList[axis]} {sQ[axis]} {eQ[axis]} {steps} monitor {beammon}\n\n'
                batch_str += scantitle + drivestr + scanstr
                print(batch_str)

            else:  # Multi-axis scan
                if all(st == 0 for st in step):
                    print("Error: At least one step must be non-zero!")
                    continue

                axis = scanQList[0]
                steps = 1 + round((eQ[axis] - sQ[axis]) / step[axis])
                scantitle = (
                    f'title "{titleTemplate} Qscan {qNameList[axis]}({sQ[axis]}~{eQ[axis]}) '
                    f'around Q({midQ[0]}, {midQ[1]}, {midQ[2]}) at E = {en} meV, {sample_T} K"\n'
                )
                scanstr = (
                    f'mscan qh {sQ[0]} {step[0]} '
                    f'qk {sQ[1]} {step[1]} '
                    f'ql {sQ[2]} {step[2]} '
                    f'en {en} 0 {steps} monitor {beammon}\n\n'
                )
                batch_str += scantitle + scanstr
                print(batch_str)

        return batch_str