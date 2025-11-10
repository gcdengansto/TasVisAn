from pathlib import Path
import pandas as pd
import numpy as np
import json

import linecache
import h5py
import re
import matplotlib.pyplot as plt


from ..base import TasData
from ..utils.toolfunc import (AnglesToQhkl, strisfloat, strisint, ni_s2_residual, ni_s2_residual_30p5meV, 
                gaussian_residual, lorentzian_residual, fit_peak, angle2, SqwDemo, PrefDemo, SelFormFactor, PrefPhononDemo)
from ..utils.toolfunc import descend_obj, h5dump
from .validator import TaipanCommandValidator

import inspy as npy
from inspy import TripleAxisSpectr
from inspy.insfit import FitConv, UltraFastFitConv
from lmfit import Parameters, fit_report, minimize



class BT4(TasData):
    def __init__(self, expnum, title, sample, user):
        super().__init__("BT4", expnum, title, sample, user)
        #self.specific_value = specific_value

    def print_exp(self):
        print('This is a BT4 experiment (ExpID: '+ self.expnum + ') entitled with ' + self.title + ' using ' + self.sample + ' by ' + self.user + ' et. al.')
        return

    def bt4_data_parser(self, file_path):
        """
        Parse NICE format data file (.bt4) and return DataFrame with metadata.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the NICE data file
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame containing the data with metadata stored in df.attrs
        """
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Initialize metadata dictionary
        metadata = {}
        header_lines = []
        data_start_idx = 0
        column_line = None
        
        # Parse header lines
        for idx, line in enumerate(lines):
            if line.startswith('##'):
                # Column names line
                column_line = line[2:].strip()
                data_start_idx = idx + 1
                break
            elif line.startswith('#'):
                header_lines.append(line)
            else:
                # Data has started without ## line
                data_start_idx = idx
                break
        
        # Extract metadata from header
        for line in header_lines:
            line = line[1:].strip()  # Remove '#' and whitespace
            
            if not line or line.startswith('?') or line.startswith('!') or line.startswith('*'):
                continue
            
            # Parse key-value pairs
            if ':' in line or ' ' in line:
                parts = line.split(None, 1)  # Split on first whitespace
                if len(parts) == 2:
                    key, value = parts
                    
                    # Store specific metadata fields
                    if key == 'filename':
                        metadata['filename'] = value
                    
                    elif key == 'experiment.title':
                        metadata['experiment.title'] = value
                    
                    elif key.startswith('sampleIndexToLattice'):
                        # Extract lattice parameters (A, B, C, Alpha, Beta, Gamma)
                        param_match = re.search(r'sampleIndexToLattice([A-Za-z]+)\.map', key)
                        if param_match:
                            param_name = param_match.group(1)
                            # Parse {1:value} format and extract the value
                            value_match = re.search(r'\{[^:]+:([^}]+)\}', value)
                            if value_match:
                                try:
                                    metadata[f'sampleIndexToLattice{param_name}'] = float(value_match.group(1))
                                except ValueError:
                                    metadata[f'sampleIndexToLattice{param_name}'] = value_match.group(1)
                            else:
                                metadata[f'sampleIndexToLattice{param_name}'] = value
                    
                    elif key == 'sampleIndexToU_reflections.map':
                        # Parse the reflections JSON and separate by id
                        try:
                            # Extract the array part from {1:[...]}
                            json_match = re.search(r'\{[^:]+:(\[.*\])\}', value)
                            if json_match:
                                reflections_list = json.loads(json_match.group(1))
                                # Store each reflection by its id
                                for refl in reflections_list:
                                    if 'id' in refl:
                                        metadata[f'reflection_id{refl["id"]}'] = refl
                        except (json.JSONDecodeError, ValueError):
                            # If parsing fails, store raw value
                            metadata['sampleIndexToU_reflections'] = value
                    
                    elif key.startswith('scatteringPlane.hkl'):
                        # Extract hkl1 and hkl2
                        metadata[key] = value
                    
                    elif key == 'trajectory.command':
                        metadata['trajectory.command'] = value
                    
                    elif key == 'trajectoryData.xAxis':
                        metadata['xAxis'] = value
                    
                    elif key == 'trajectoryData.yAxis':
                        metadata['yAxis'] = value
        lattice =[ metadata["sampleIndexToLatticeA"], metadata["sampleIndexToLatticeB"],metadata["sampleIndexToLatticeC"],
                metadata["sampleIndexToLatticeAlpha"], metadata["sampleIndexToLatticeBeta"], metadata["sampleIndexToLatticeGamma"]]
        params={"scanax1": metadata['xAxis'], "scanax2": None, 
                "title": metadata['experiment.title'], "filename":  metadata['filename'],
                "lattice": lattice, "orient_v1": metadata["scatteringPlane.hkl1"], "orient_v2": metadata["scatteringPlane.hkl2"] }
            
        # Parse column names
        if column_line:
            columns = column_line.split()
        else:
            # Try to infer from first data line
            first_data = lines[data_start_idx].strip().split()
            columns = [f'col_{i}' for i in range(len(first_data))]
        
        # Read data into DataFrame
        data_lines = lines[data_start_idx:]
        data_rows = []
        
        for line in data_lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Split by whitespace and convert to appropriate types
                values = line.split()
                data_rows.append(values)
        
        # Create DataFrame
        #print(columns)
        """
        ['pt', 'A3(deg)', 'monitor', 'roi', 'time(s)', 'start', 'stop', 'counts', 
        'rateMeter.detectorRate(1/s)', 'rateMeter.monitorRate(1/s)', 'reactorPower.coldSourcePressure(PSI)', 'reactorPower.coldSourceTemperature(K)',
        'reactorPower.reactorPowerPercent(%)', 'reactorPower.reactorPowerThermal(MW)', 'reactorPower.reactorState', 
        'CuFocus(cm)', 'PGFocus(cm)', 'A5(deg)', 'A6(deg)', 'chi(deg)', 'cmode', 'mon.set', 'roi.set', 'time.set(s)', 'coupledChi(deg)', 'coupledPhi(deg)',
        'ef(meV)', 'ei(meV)', 'et(meV)', 'eulerian.lcutchi(deg)', 'eulerian.lcutphi(deg)', 'eulerian.stratchi(deg)', 'eulerian.stratphi(deg)', 
        'filterCart(deg)', 'slitH(mm)', 'slitW(mm)', 'lowerTilt(deg)', 'lowerTranslation(mm)', 'A1(deg)', 'A2(deg)', 'phi(deg)', 'sample', 
        'sampleIndex', 'H', 'K', 'L', 'cutth(deg)', 'cut2th(deg)', 'scatteringPlaneTolerance', 'A4(deg)', 'tilt.cutl(deg)', 'tilt.cutu(deg)', 
        'tilt.stratl(deg)', 'tilt.stratu(deg)', 'trajectory.experimentPointID', 'pt#', 'upperTilt(deg)', 'upperTranslation(mm)']
        """
        newColNames=    ['Pt.', 's1', 'monitor', 'det_roi', 'time', 'start', 'stop', 'detector', 
        'detRate', 'monRate', 'coldSourceP', 'coldSourceT', 'reactorPPercent', 'reactorPower', 'reactorState', 
        'CuFocus', 'PGFocus', 'a1', 'a2', 'chi(deg)', 'cmode', 'mon.set', 'roi.set', 'time.set', 'coupledChi', 'coupledPhi',
        'ef', 'ei', 'en', 'lcutchi', 'lcutphi', 'stratchi', '.stratphi', 
        'filterCart', 'slitH', 'slitW', 'sgl', 'stl', 'm1', 'm2', 'phi', 'sample', 
        'sampleIdx', 'qh', 'qk', 'ql', 'cutth', 'cut2th', 'PlaneTol', 's2', 'cutl', 'cutu', 
        'stratl', 'stratu', 'expPtID', 'PtNo.', 'sgu', 'stu']
        df = pd.DataFrame(data_rows, columns=newColNames)
        params["scanax1"]= newColNames[1]  # the first column after Pt.
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Keep as string if conversion fails
                pass
        
        # Store metadata in DataFrame attributes
        df.attrs = params
        
        return df


    def bt4_data_to_pd(self, path, filename):
        # convert the bt4 raw data file into dataframe

        filefullPath = Path(path) / filename
        if filefullPath.is_file() and filefullPath.name.lower().endswith(".bt4"):
            df = self.bt4_data_parser(filefullPath)
            return df
        else:
            print(f"Error: {filefullPath} is not a bt4 file and an empty df is returned.")
            return pd.DataFrame([])          #if the filename is wrong, create an emtpy df.
        
        
    def bt4_data_filelist(self, folder_path):
        # find all the data files in the given path

        path = Path(folder_path)
        filename_list=[]

        if path.is_dir():
            # Only check if the file name ends with ".bt4"
            filename_list = [f.name for f in path.iterdir() if f.is_file() and f.name.lower().endswith(".bt4")]
            return filename_list        

        else:
            raise ValueError("Error: the given directory doesn't exist!")
            


    def bt4_filelist_to_dflist(self, path="", filename_list=None):

        dflist=[]
        for filename in filename_list:
            df  = self.bt4_data_to_pd(path, filename)
            dflist.append(df)

        return dflist
    

            
    def bt4_simpleplot(self, path="", filename_list=None, fit=False, initial=None):
        #quick plot without normalization and combination
        dflist = self.bt4_filelist_to_dflist(path, filename_list)
        parlist,fitlist,fig, ax=self.tas_simpleplot(dflist, fit, initial)

        return parlist, fitlist, fig, ax