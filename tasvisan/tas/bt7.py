from pathlib import Path
import pandas as pd
import numpy as np

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


class BT7(TasData):

    def __init__(self, expnum, title, sample, user):
        super().__init__("BT7", expnum, title, sample, user)
        #self.specific_value = specific_value

    def print_exp(self):
        print('This is a BT7 experiment (ExpID: '+ self.expnum + ') entitled with ' + self.title + ' using ' + self.sample + ' by ' + self.user + ' et. al.')
        return

    def bt7_data_parser(self, file_path, retopt="STD"):
        # Read all lines to extract header and column names
        #retopt: STD: return normal data hkl, monitor, detector, m1,m2,s1,s2, a1,a2 .. in pd form
        #        PSD: return hkl, monitor, detector, psd, 
        #        ANAConf:
        #        MONOConf:
        #        DOORDET:
        #        DDSD: 
        #        ALL:
    
        params={}
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Collect header lines (starting with '#') for metadata
        #header_lines = [line.strip() for line in lines if line.startswith('#')]
        
        # Find the column names (line starting with '#Columns')
        col_name_list  = None
        scan_descr     = None
        det_eff_list   = None
        det_eff_dict   = {}
        det_inuse      = None
        dd_list        = None
        sd_list        = None
        psd_list       = None
        doord_list     = None
        psd_eff_list   = None
        scan_axis1      = None
        scan_axis2      = None
        lattice        = None
        orient         = None
        
        for line in lines:
            if line.startswith('#Columns'):
                col_name_list = line.strip('#Columns').strip().split()
                #print(col_name_list)
            if line.startswith('#ScanDescr'):
                scan_descr = line.strip('#ScanDescr ').strip().split()
                #print(scan_descr)
                
            if line.startswith('#DetectorEfficiencies'):
                det_eff_list = line.strip('#DetectorEfficiencies').strip().split()
                for effstr in det_eff_list:
                    temp=effstr.split("=")
                    det_eff_dict.update({temp[0]: float(temp[1])})
                #print(det_eff_dict)
            if line.startswith('#AnalyzerDetectorDevicesOfInterest'):
                det_inuse = line.strip('#AnalyzerDetectorDevicesOfInterest').strip().split()
                #print(det_inuse)
    
            if line.startswith('#AnalyzerDDGroup'):
                dd_list = line.strip('#AnalyzerDDGroup').strip().split()
                #print(dd_list)
    
            if line.startswith('#AnalyzerDoorDetectorGroup'):
                doord_list = line.strip('#AnalyzerDoorDetectorGroup').strip().split()
                #print(doord_list)
    
            if line.startswith('#AnalyzerPSDGroup'):
                psd_list = line.strip('#AnalyzerPSDGroup').strip().split()
                #print(psd_list)
            if line.startswith('#AnalyzerSDGroup'):
                sd_list = line.strip('#AnalyzerSDGroup').strip().split()
                #print(sd_list)
            if line.startswith('#Scan    '):
                scan_axis1=line.strip('#Scan ').strip().split()[1]
                #print(scan_axis1)
            if line.startswith('#ScanRanges:'):
                scan_axes=line.strip('#ScanRanges:').strip().split()
                if len(scan_axes) >= 3:
                    scan_axis2=scan_axes[3]
                
            if line.startswith('#FixedE'): #FixedE      Ef 14.7
                fixed=line.strip('#FixedE').split()
            if line.startswith('#MonoVertiFocus'): #FixedE      Ef 14.7
                monoVF=line.strip('#MonoVertiFocus').strip()
            if line.startswith('#MonoHorizFocus'): #FixedE      Ef 14.7
                monoHF=line.strip('#MonoHorizFocus').strip()
            if line.startswith('#AnalyzerFocusMode'): #FixedE      Ef 14.7
                anaFocus=line.strip('#AnalyzerFocusMode').strip()
    
            if line.startswith('#Lattice'):
                lattice=[float(s) for s in line.strip('#Lattice').strip().split()]
                #print(lattice)
            if line.startswith('#Orient'):
                H1, K1, L1, H2, K2, L2 =[float(s) for s in line.strip('#Orient').strip().split()] 
                orient=[[H1, K1, L1], [ H2, K2, L2]]
                #print(orient)
                
        if det_eff_dict is not None:
            det_eff_dict=dict(sorted(det_eff_dict.items()))
            
        # Load the data into a DataFrame, skipping header lines
        df = pd.read_csv(file_path, comment='#', delimiter=r'\s+', names=col_name_list)
    
        #get the ana blade angles
        anaBlades = ['AnalyzerBlade01', 'AnalyzerBlade02', 'AnalyzerBlade03', 'AnalyzerBlade04', 'AnalyzerBlade05', 
                              'AnalyzerBlade06', 'AnalyzerBlade07', 'AnalyzerBlade08', 'AnalyzerBlade09', 'AnalyzerBlade10', 
                              'AnalyzerBlade11', 'AnalyzerBlade12', 'AnalyzerBlade13']
        df_Ana_bladeAng = None
        if set(anaBlades).issubset(df.columns):
            df_Ana_bladeAng  = df[anaBlades]
        
        #get the mono blade angles   
        monoBlades = ['MonoBlade01', 'MonoBlade02', 'MonoBlade03', 'MonoBlade04', 'MonoBlade05', 'MonoBlade06', 
                               'MonoBlade07', 'MonoBlade08', 'MonoBlade09', 'MonoBlade10']
        df_Mono_bladeAng = None
        if set(monoBlades).issubset(df.columns):
            df_Mono_bladeAng  = df[monoBlades]
        
        #get the collimators' values
        cols   = ['PostAnaColl', 'PostMonoColl', 'PreAnaColl', 'PreMonoColl']
        df_COL = None
        if set(cols).issubset(df.columns):
            df_COL = df[cols]
            
        #Door Detectors    
        df_DoorD = None
        if set(doord_list).issubset(df.columns):
            df_DoorD = df[doord_list]
            
        #Diffr Detectors
        df_DD = None
        if set(dd_list).issubset(df.columns):
            df_DD = df[dd_list]
            
        #Single Detectors
        df_SD = None
        if set(sd_list).issubset(df.columns):
            df_SD = df[sd_list]
            
        #PSD Detectors
        df_PSD = None
        if set(psd_list).issubset(df.columns):
            df_PSD = df[psd_list]
    
        #change the column names into standard
        origin_colnames = ['QX', 'QY', 'QZ', 'E', 'A4', 'Time', 'Temp', 'Monitor', 'Detector', 'DFMRot', 'A2', 'A3', 'A5', 'A6', 'DFM', 'A1', 'ApertHori', 'ApertVert', 'BkSltHght', 'BkSltWdth', 
         'DiffDet', 'Ef', 'Ei', 'FLIP', 'Filtran', 'FocusCu', 'FocusPG', 'H', 'K', 'L', 'Monitor2',  'MonoElev', 'MonoTrans', 'PSDet',  'RC', 'SC', 'SingleDet', 
         'SmplElev', 'SmplGFRot', 'SmplHght', 'SmplLTilt', 'SmplLTrn', 'SmplUTilt', 'SmplUTrn', 'SmplWdth', 'Velsel', 'TemperatureControlReading', 'TemperatureSetpoint', 'TemperatureHeaterPower', 
         'TemperatureSensor1', 'TemperatureSensor2', 'TemperatureSensor3', 'timestamp']
    
       
        new_colnames = ['qh', 'qk', 'ql', 'en', 's2', 'time', 'temp', 'monitor', 'detector', 'DFMRot', 'm2', 's1', 'a1', 'a2', 'DFM', 'm1', 'vs_hori', 'vs_vert', 'pa_height', 'pa_width', 
         'a2_dd', 'ef', 'ei', 'FLIP', 'Filtran', 'FocusCu', 'FocusPG', 'h', 'k', 'l', 'monitor2',  'mvtrans', 'mtrans', 'a2_psd',  'RC', 'SC', 'a2_sd', 
         'svtrans', 'SmplGFRot', 'ps_vslit', 'sgl', 'stl', 'sgu', 'stu', 'ps_hslit', 'Velsel', 'tempContr', 'tempSP', 'tempHPower', 
         'tempVTI', 'tempSAMP', 'tempSAMPh', 'timestamp'] #'tempVTI', 'tempSAMP','tempSAMPh'
        if df_PSD is not None:
            origin_colnames = origin_colnames+psd_list
            new_colnames = new_colnames+psd_list

        #in case, there are some columns in origin_colnames missing from the data file, then only get the existing ones
        existing = df.columns.intersection(origin_colnames)  
        missing = list(set(origin_colnames) - set(df.columns))
        #for the missing columns, we need to remove them from new_colnames as well.
        if missing != []: 
            print(f"Warning: these columns does not exist in the data file:{missing}")
        missingidxlist = list([])
        for col in missing:
            absentidx= origin_colnames.index(col)
            missingidxlist.append(absentidx)
        
        for idx in sorted(missingidxlist, reverse=True):  #pop from the last one to the first one, to make sure index was correct.
            new_colnames.pop(idx)


        df_red         = df[existing]    # STD df
        df_red.columns =  [new_colnames]
        
        for idx, name in  enumerate(existing):
            if  name == scan_axis1 :
                scan_axis1 = new_colnames[idx]
    
    
        params={"scanax1": scan_axis1,"scanax2": scan_axis2, 'det_inuse': det_inuse, 'sd_list': sd_list, 'dd_list': dd_list, 'doord_list': doord_list, 
                'psd_list': psd_list, "det_eff": det_eff_dict, "fixed": fixed, "monoVF": monoVF, "monoHF": monoHF,"anaFocus": anaFocus,  "lattice": lattice, "orient_v1": orient[0], "orient_v2": orient[1] }
        df_red.attrs = params

        
        if retopt=="STD" :
            return df_red
        elif retopt=="PSD" :
            if df_PSD is None:
                print("Error: no psd data is available for the current scan!")
            else:
                return df_red
        elif retopt=="ANAConf" :
            return df_Ana_bladeAng
        elif retopt=="MONOConf" :
            return df_Mono_bladeAng
        elif retopt=="DOORDET" :
            return df_DoorD
        elif retopt=="DD" :
            return df_DD
        elif retopt=="SD" :
            return df_SD
        elif retopt=="ALL" :
            for idx, name in  enumerate(origin_colnames):
                df = df.rename(columns={name: new_colnames[idx]}) 
            df.attrs = params
            return df 
        else:
            print("Error: The given turn option does not exist. thus return the standard reduced df.")
            return df_red


    
    def bt7_data_to_pd(self, path, filename):
        # convert the bt7 raw data file into dataframe

        filefullPath = Path(path) / filename
        if filefullPath.is_file() and filefullPath.name.lower().endswith(".bt7"):
            df = self.bt7_data_parser(filefullPath, retopt="STD")
            return df
        else:
            print(f"Error: {filefullPath} is not a bt7 file and an empty df is returned.")
            return pd.DataFrame([])          #if the filename is wrong, create an emtpy df.
        
        
    def bt7_data_filelist(self, folder_path):
        # find all the data files in the given path

        path = Path(folder_path)
        filename_list=[]

        if path.is_dir():
            # Only check if the file name ends with ".bt7"
            filename_list = [f.name for f in path.iterdir() if f.is_file() and f.name.lower().endswith(".bt7")]
            return filename_list        

        else:
            raise ValueError("Error: the given directory doesn't exist!")
            


    def bt7_filelist_to_dflist(self, path="", filename_list=None):

        dflist=[]
        for filename in filename_list:
            df  = self.bt7_data_to_pd(path, filename)
            dflist.append(df)

        return dflist
    

        
    def bt7_simpleplot(self, path="", filename_list=None, fit=False, initial=None):
        #quick plot without normalization and combination
        dflist = self.bt7_filelist_to_dflist(path, filename_list)
        parlist,fitlist,fig, ax=self.tas_simpleplot(dflist, fit, initial)

        return parlist, fitlist, fig, ax


