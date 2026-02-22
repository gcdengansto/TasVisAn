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


class Taipan(TasData):
    """Taipan class that extends TasData."""
    
    def __init__(self, expnum, title, sample, user):
        super().__init__("Taipan", expnum, title, sample, user)
        #self.specific_value = specific_value

    def _taipan_data_header_info(self, path):
        """
        Parse a neutron data file and extract header information.
        Args:
            path (str): Path to the data file
        Returns:
            dict: Dictionary containing all extracted parameters
        """
        # Dictionary to store all extracted parameters
        params = {}
        
        # Keep track of last header line and data start line
        last_header_line     = ""
        last_header_line_num = 0
        data_start_line      = 0
        line_count           = 0
        
        # Special handling for arrays
        array_params = ['latticeconstants', 'ubmatrix'] #, 'plane_normal'
        
        with open(path, 'r') as f:
            for line in f:
                line_count += 1
                
                if line.startswith('#'):
                    last_header_line = line
                    last_header_line_num = line_count
                    
                    # Skip the header line that contains column names
                    if line.strip().startswith('# Pt.') or '# Pt.' in line:
                        continue
                    
                    # Process parameter line
                    clean_line = line[1:].strip()
                    
                    # Skip empty lines or lines without '='
                    if not clean_line or '=' not in clean_line:
                        continue
                    
                    # Split by the first occurrence of '='
                    key, value = clean_line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle special array parameters
                    if key in array_params:
                        # Split by commas and convert to float
                        value = [float(item) for item in value.split()]
                    
                    # Store in dictionary
                    params[key] = value
                else:
                    # First non-comment line is the start of data
                    data_start_line = line_count
                    break
        
        # Add special information about the file structure
        params['_data_start_line'] = data_start_line
        params['_header_line']     = last_header_line
        params['_header_line_num'] = last_header_line_num

        #params['expno']   = params.pop('experiment_number')
        #params['scanno']  = params.pop('scan')
        #params['lattice'] = params.pop('latticeconstants')
        
        # Use .pop() with default to avoid KeyError:
        params['expno'] = params.pop('experiment_number', None)
        params['scanno'] = params.pop('scan', None)
        params['lattice'] = params.pop('latticeconstants', None)

        if params['expno'] is None or params['scanno'] is None or params['lattice'] is None:
            raise ValueError("Missing required header parameters in data file")


        params['scanax1'] = params.get('def_x', '')
        if not params['scanax1']:
            print("Warning: scan axis is empty!")


        return params


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
        #datastart = 0
        #datalines = 0       
     
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


    def taipan_hdf_to_pd(self, path='', scanno = 90000):
        #This function has been tested.
        ppath=Path(path)

        if not ppath.is_dir():
            print("Error: wrong path!")
            return
        if isinstance(scanno, str):
            filename = scanno
        else:
            filename = str(scanno)

        if len(filename) == 5:
            filename = '00'+filename
            #print(filename)
        elif len(filename) == 6:
            filename = '0'+filename
        elif len(filename) >= 7:
            filename = filename
        else:
            print("The scan list is wrong!")


        fileext      = ".nx.hdf"
        fileinitial  = 'TPN'
        fullfilename = fileinitial + filename + fileext
        fullpath= ppath/ Path(fullfilename)

        namelist, col_list= h5dump(fullpath)

        temptitle='no title'
        pd_hdf  = pd.DataFrame([])
        scaninfo    = {}

        with h5py.File(str(fullpath), 'r') as file:
            #print(file["/entry1/experiment/title"])
            temptitle=file.get("/entry1/experiment/title")[0]    #get() is to get the dataset in hdf, you need the full path, it is always an array-like object, use [] to index
            #print(temptitle)
            tempax=None
            for index, name in enumerate(namelist):
                if name.find("data") != -1:
                    xx = np.array(file[name])
                    if np.abs(xx[0]-xx[-1])/len(xx) > 0.0006:
                        tempax = col_list[index]
                else:  
                    new_col         = pd.DataFrame(file[name])
                    new_col.columns = [col_list[index]]
                    pd_hdf       = pd.merge(pd_hdf, new_col, how='outer', left_index=True, right_index=True,suffixes=('', ''))

            if tempax is None:
                raise ValueError("Could not determine scan axis from data")

        neworder=pd_hdf.columns.to_list()
        
        if tempax in neworder:
            neworder.pop(neworder.index(tempax))
            neworder.insert(0, tempax)
        pd_hdf=pd_hdf.reindex(columns=neworder)
        newcolnames=neworder.copy()

        #newcolnames[newcolnames.index('bm1_counts')]= 'monitor'
        #newcolnames[newcolnames.index('bm2_counts')]= 'detector'
        #newcolnames[newcolnames.index('VS_left')] = 'vs_left'
        #newcolnames[newcolnames.index('VS_right')]= 'vs_right'

        # Add existence checks before renaming:
        if 'bm1_counts' in pd_hdf.columns:
            newcolnames[newcolnames.index('bm1_counts')]= 'monitor'
        if 'bm2_counts' in pd_hdf.columns:
            newcolnames[newcolnames.index('bm2_counts')]= 'detector'
        if 'VS_left' in pd_hdf.columns:
            newcolnames[newcolnames.index('VS_left')] = 'vs_left'
        if 'VS_right' in pd_hdf.columns:
            newcolnames[newcolnames.index('VS_right')]= 'vs_right'



        if 'sensorValueA' in pd_hdf:
            newcolnames[newcolnames.index('sensorValueA')]= 'tempVTI' #'T1_Sensor1'
        if 'sensorValueB' in pd_hdf:
            newcolnames[newcolnames.index('sensorValueB')]= 'tempSAMP' #'T1_Sensor2'
        if 'sensorValueC' in pd_hdf:
            newcolnames[newcolnames.index('sensorValueC')]= 'tempSAMPh' #'T1_Sensor3'
        if 'sensorValueD' in pd_hdf:
            newcolnames[newcolnames.index('sensorValueD')]= 'tempIdle' #'T1_Sensor4'
        if 'setpoint1' in pd_hdf:
            newcolnames[newcolnames.index('setpoint1')]= 'tempSP1' #'T1_Setpoint1'
        if 'setpoint2' in pd_hdf:
            newcolnames[newcolnames.index('setpoint2')]= 'tempSP2' #'T1_Setpoint2'

        pd_hdf.columns = newcolnames
        
        scaninfo={"scanno":scanno,"scanax": tempax, "title": temptitle}
        
        print(pd_hdf[['qh','qk','ql', 'en','detector','monitor']])
        print(pd_hdf.columns)
        #there is not enough attrs items 
        pd_hdf.attrs=scaninfo

        return pd_hdf 

    def taipan_scanlist_to_dflist(self, path="", scanlist=None):

        if scanlist is None:
            raise ValueError("scanlist cannot be None")

        dflist=[]

        if all(isinstance(x, (int, np.integer)) for x in scanlist):
            for scanno in scanlist:
                df  = self.taipan_data_to_pd(path, scanno)
                dflist.append(df)

            #print(f"Totally, {len(dflist)} scans in the list are converted into df. ")
            return dflist   # reduce
        else:
            print("Some element in the list is not integer. sublist in the list is not allowed.")
            return
    
    
    def taipan_simpleplot(self, path="", scanlist=None, fit=False, initial=None):
        #quick plot without normalization and combination
        dflist = self.taipan_batch_reduction(path, scanlist)
        parlist,fitlist,fig, ax=self.tas_simpleplot(dflist, fit, initial)

        return parlist,fitlist,fig, ax

    def taipan_combplot(self, path='', scanlist=None, fit=False, norm_mon_count = -999999, overplot=False, offset=1000, initial=None ):
        #scanlist can be [123, 125, [130, 132],115]
        dflist=[]
        for scanno in scanlist:
            if isinstance(scanno, list):
                subscanlist = scanno
                subdflist=self.taipan_scanlist_to_dflist(path, subscanlist)
                comb_df=super().tas_datacombine(subdflist)
                dflist.append(comb_df)
            if isinstance(scanno, (int, np.integer)):
                single_df = self.taipan_data_to_pd(path, scanno)
                dflist.append(single_df)
        
        params_df, fitted_df, fig, ax=super().tas_combplot(dflist, fit, norm_mon_count, overplot, offset, initial)

        return params_df, fitted_df, fig, ax


    def taipan_batch_reduction(self, path='', scanlist=None, motorlist=None):
        
        dflist     = list()

        for scanno in scanlist:
            #print(f"the problem here:{scanno}")
            #print(isinstance(scanno, int))  # this will be false, int is not np.integer the same
            if isinstance(scanno, (int, np.integer)):
                df = self.taipan_data_to_pd(path, scanno)
            
            elif isinstance(scanno, list):
                subscanlist = scanno
                temp_dflist = self.taipan_scanlist_to_dflist(path, subscanlist)
                df = super().tas_datacombine(temp_dflist)
            else:
                print("Warning: The elements in the list is not int or list.")
                return
            if motorlist:
                col_names = motorlist #['h', 'k', 'l', 'e', 'q',  'm1', 'm2','s1','s2', 'a1', 'a2', 'sgl', 'sgu', 'stu', 'stl', 'sd','dd','i3h','i3v','i4h','i4v','Pt.','i1l','i1r', 'mfh', 'mfv', 'c4s','c4r', 'ei', 'ef', 'count', 'beamMon']
                col_notexist=[]
                for col in col_names:
                    if col not in df.columns:
                        col_notexist.append(col)
                # remove those col_names that is not in the data file in case. 
                col_names= [x for x in col_names if x not in col_notexist] 
                if col_notexist:
                    print(f"Warning: These motors are not in the data:{col_notexist}")

                dflist.append(df[col_names])
            else:
                dflist.append(df)  #fulllist
        return dflist  # return reduced columns



    def taipan_reduction_by_row(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'ei','ef', 'm1', 'm2','s1','s2', 'a1', 'a2', 'detector', 'monitor','tempVTI', 'tempSAMP'], sortby="tempSAMP"):
        dflist = self.taipan_batch_reduction(path, scanlist, motorlist)
        df_extend = pd.DataFrame(columns=motorlist)

        for df in dflist:
            if df is not None and not df.empty and not df.isna().all().all():
                df_extend = pd.concat([df_extend, df], ignore_index=True)
        if sortby in motorlist:
            df_extend = df_extend.sort_values(by=sortby, ascending = True).reset_index(drop=True)
        return df_extend
    
    def taipan_reduction_anycolumn(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'ei','ef', 'm1', 'm2','s1','s2', 'a1', 'a2', 'detector', 'monitor','tempVTI', 'tempSAMP'], sortby="tempSAMP"):
        dflist = self.taipan_batch_reduction(path, scanlist, motorlist)
        df_extend = pd.DataFrame(columns=motorlist)

        for df in dflist:
            if df is not None and not df.empty and not df.isna().all().all():
                df_extend = pd.concat([df_extend, df], ignore_index=True)
        if sortby in motorlist:
            df_extend = df_extend.sort_values(by=sortby, ascending = True).reset_index(drop=True)
        return df_extend


    def taipan_random_contour(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'detector', 'monitor'], x_col="qh", y_col="en", xtitle='qh [rlu]', ytitle='en [meV]',title='Contour Map of Measurement Data', vminmax=[0, 1000], output_file=None):
        df_total = self.taipan_reduction_by_row(path, scanlist, motorlist=motorlist)
        #print(dflist[0])
        ax = super().tas_random_contour(df_total, x_col=x_col, y_col=y_col, xtitle=xtitle, ytitle=ytitle, title=title, vminmax=vminmax, output_file=output_file)
        return ax


    def taipan_tidy_contour(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'detector', 'monitor'], x_col='qh', y_col='en', xlabel='Q [rlu]',ylabel='E [meV]', vminmax=None, bRot=False, ax=None):
       
        dflist = self.taipan_batch_reduction(path, scanlist, motorlist=motorlist)
        xx, yy, intt, ax = super().tas_tidy_contour(dflist,  x_col=x_col, y_col=y_col, xlabel=xlabel, ylabel=ylabel, vminmax=vminmax, bRot=bRot, ax=ax)
        return xx, yy, intt, ax


    def taipan_export_hklw(self, path='', scanlist = None, hklw_file=""):
        #this is advanced export function, which can combine data and then export
        #scanlist could be [ 2, [3, 4], 5, 6] or [ 12, 23, 34, 45]
        hklw_df = self.taipan_batch_reduction(path, scanlist, motorlist=['qh', 'qk', 'ql', 'en', 'detector'])
        
        super().tas_export_hklw(hklw_df, hklw_file)

        return hklw_df



    def export_scantitle(self, path='', datafrom_to=None, outputfile=None):
        
        ppath=Path(path)

        if not ppath.is_dir():
            print("Error: wrong path!")
            return

        if datafrom_to is None:
            print("Error: No data from and to number is given")
            return
        if outputfile is None:
            outputfile = "TAIPAN_exp" + self.expnum + "_scanlist.html"

        fileext = ".dat"
        firstno = int(datafrom_to[0])
        lastno  = int(datafrom_to[1])

        filenolist = np.linspace(firstno, lastno, (lastno-firstno+1))

        #print(filenolist)
        scanno_list    = []
        command_list   = []
        scantitle_list = []
        fileinitial    =  "TAIPAN_exp" + self.expnum + "_scan"

        for fileindex, fileno in enumerate (filenolist):

            fullfilename = fileinitial + str(int(fileno)) + fileext

            fullpath= ppath / Path(fullfilename)

            scanno_list.append(int(fileno))
            try:
                with fullpath.open() as f:
                    totallines=list(f)                              #read the lines in the file f, initialize a list
                    for index, line in enumerate (totallines):

                        if (line.find("# command =") != -1):  
                            command_list.append(line[11:-1])
                            #print(command_list)
                            #print(index)
                        if (line.find("# scan_title =")!= -1):
                            scantitle_list.append(line[14:-1])
            except IOError as e:
                print("Couldn't open file (%s)." % e)
                
                
        scan_dict = {"scanno": scanno_list, "command": command_list, "scantitle": scantitle_list}
        
        scanlist  = pd.DataFrame(scan_dict)
        scanlist.to_html( ppath / Path(outputfile))
        #print(scanlist)

        return scanlist


    def export_scanlog(self, path='', logfile=None, outputfile=None):
    
        ppath=Path(path)

        if not ppath.is_dir():
            print("Error: wrong path!")
            return
        if logfile is None:
            print("Error: No logfile name is given")
            return
        if outputfile is None:
            outputfile =  "TAIPAN_exp" + self.expnum + "_scanlist.html"

        fullfilepath = Path(path) / Path(logfile)
        #print(fullfilepath)

        #print(filenolist)
        hdf_filelist   = []
        command_list  = []

        total_line_no = 0
        
        try: 
            with fullfilepath.open() as f:
                for i, l in enumerate(f):
                    total_line_no=i
                    
            total_line_no = total_line_no + 1

            cur_line      = total_line_no + 1

            while cur_line > 0 :
                scancmdfound = 0
                cur_line = cur_line-1
                line = linecache.getline(str(fullfilepath), cur_line)
                if (line.find("nx.hdf updated") != -1):
                    hdf_filelist.append( line[22:32])
                    #print(hdf_filelist)
                    #curscanfootprint = line[22: -1]
                    while (cur_line > 0 ) and (scancmdfound==0):
                        cur_line=cur_line -1
                        line =linecache.getline(str(fullfilepath), cur_line)
                        if line.find("Scanvar:") != -1:
                            cur_line = cur_line-1
                            line1 =linecache.getline(str(fullfilepath), cur_line)
                            cur_line = cur_line-1
                            line2 =linecache.getline(str(fullfilepath), cur_line)
                            cur_line = cur_line-1
                            line3 =linecache.getline(str(fullfilepath), cur_line)
                            if line3[22:32].find("runscan") !=-1 or line3[22:32].find("mscan")   !=-1:
                                command_list.append(line3[22:-1])
                            elif line2[22:32].find("runscan") !=-1 or line2[22:32].find("mscan") !=-1:
                                command_list.append(line2[22:-1])
                            elif line1[22:32].find("runscan") !=-1 or line1[22:32].find("mscan") !=-1:
                                command_list.append(line1[22:-1])
                            else:
                                cur_scan = line[22:76]
                                if line1.find("Scanvar:") !=-1:
                                    cur_scan = cur_scan+line1[22:76]
                                if line2.find("Scanvar:") !=-1:
                                    cur_scan = cur_scan+line2[22:76]
                                if line3.find("Scanvar:") !=-1:
                                    cur_scan = cur_scan+line3[22:76]
                                command_list.append(cur_scan)

                            scancmdfound = 1

            #print(len(hdf_filelist))
            #print(len(command_list))
            scan_dict = {"scanhdf_name": hdf_filelist, "command": command_list}
            scanlist  = pd.DataFrame(scan_dict)
            scanlist  = scanlist[::-1]
            scanlist = scanlist.reset_index(drop=True)
            #scanlist.index = df.index.astype(int)
            #scanlist = scanlist.sort_index(ascendingbool = false)
            outputpath = Path(path) / Path(outputfile)
            #print(outputpath) 
            scanlist.to_html( outputpath )

            #print(pd.index)
            #print(scanlist)
            return scanlist
        
        except FileNotFoundError:
            # Handle the case where the file does not exist
            print("Error: The file does not exist. Please check the file path.")
            return 

        except IOError:
            # Handle other I/O-related errors (e.g., file is corrupted)
            print("Error: The file could not be read. It may be corrupted.")
            return

        except Exception as e:
            # Catch any other unforeseen exceptions
            print(f"An unexpected error occurred: {e}")
            return

    def export_scanlist(self, path='', logfile=None, outputfile=None):
        #this is the best function to use.

        ppath=Path(path)

        if not ppath.is_dir():
            print("Error: wrong path!")
            return

        if outputfile is None:
            outputfile =  "TAIPAN_exp" + self.expnum + "_scanlist.html"

        scanlist = self.export_scanlog(path=path, logfile=logfile, outputfile=outputfile)
        if scanlist is not None:

            scanno_list    = []
            command_list   = []
            scantitle_list = []

            for hdfname in scanlist['scanhdf_name']:

                filename = "TAIPAN_exp" + self.expnum + "_scan" + str(int(hdfname[-7:])) + '.dat'
                datafilepath = Path(path)/Path("Datafiles")/Path(filename)
                #print("file exists? {}".format(datafilepath.is_file()))
                scanno_list.append(int(hdfname[-7:]))
                
                if datafilepath.is_file():
                    try:
                        with datafilepath.open() as f:
                            totallines=list(f)                              #read the lines in the file f, initialize a list
                            for index, line in enumerate (totallines):

                                if (line.find("# command =") != -1):  
                                    command_list.append(line[11:-1])

                                if (line.find("# scan_title =")!= -1):
                                    scantitle_list.append(line[14:-1])
                    except IOError as e:
                        print("Couldn't open file ({}).".format(e))
                else:
                    command_list.append("no-file" )
                    scantitle_list.append("no-file")
                    print("File ({}) does not exist.".format(hdfname) )


            scan_dict = {"scan_no": scanno_list, "command_dat": command_list, "scantitle": scantitle_list}
            scantitlelist  = pd.DataFrame(scan_dict)
            #print(scantitlelist)
            fulllist=pd.merge(scanlist, scantitlelist, how='outer', left_index=True, right_index=True,suffixes=('', ''))
            #print(fulllist.columns)
            fulllist=fulllist[['scan_no','command','scantitle']]
            outputpath = Path(path) / Path(outputfile)
            fulllist.to_html(outputpath)

            return fulllist
        else:
            print("ERROR: no information from the logfile.")
            return pd.DataFrame(columns=['scan_no','command','scantitle'])


    def taipan_calibr_6scans(self, path='', scanlist=None):
        # this function is for the 6 ni scans. each scan has only one peak
        niscan_dflist =   self.taipan_batch_reduction(path, scanlist)

        fitted_data =  pd.DataFrame([])
        params_df   =  pd.DataFrame(columns = ['A', 'A_err', 'w','w_err', 'x0', 'x0_err', 'bg', 'bg_err'])
        fig         =  plt.figure()
        fig, ax     =  plt.subplots(1, 1)

        for ii in range(len(scanlist)):
            col_x_title = niscan_dflist[ii].attrs['scanax1']
            col_y_title = "detector"

            dataX = niscan_dflist[ii][col_x_title].to_numpy()
            dataY = niscan_dflist[ii][col_y_title].to_numpy()

            #dataX = dataX[np.logical_not(np.isnan(dataX))]
            #dataY = dataY[np.logical_not(np.isnan(dataY))]

            cur_fitpar, cur_fitdat= fit_peak(dataX, dataY, func='G')

            plt.plot(dataX, dataY, 'o', cur_fitdat['X'], cur_fitdat['Y_fit'], '-')
            
            params_df     =  pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)

            fitted_data   =  pd.merge(fitted_data, cur_fitdat, how='outer', left_index=True, right_index=True, suffixes=("", "_"+str(ii)))

        params_df = params_df.sort_values(by = ['x0'], ascending=False)  # reverse order
        ni_peakpos     = params_df['x0'].to_numpy()
        calibr_result = self.calibr_fit_offset(ni_peakpos)
        #print(calibr_result)
        
        return params_df, fitted_data, calibr_result, fig, ax


    def calibr_fit_offset(self, peaks=None):
        #this is the fitting for calibration
        if peaks is None:
            raise ValueError("No peaks for fit.")
        
        d_pg002 =   3.355
        dataX   =   np.array([1, 2, 3, 4, 5, 6])
        dataY   =   peaks.flatten() 
        # [-33.063978, -38.440318,-55.727153, -66.547779, -69.950519, -82.977134]
        fit_params  =  Parameters()
        fit_params.add('s2_offset',    value  =  0.01)
        fit_params.add('wavelen',      value  =  2.345)

        out   = minimize(ni_s2_residual, fit_params, args=(dataX, dataY), method='Levenberg-Marquardt')
        #print(fit_report(out))

        #print("The fitted parameters:")
        #for name, param in out.params.items():
            #print(f'{:10}: {:6.8f} +/- {:6.8f}'.format(name, param.value, param.stderr))
            
        m1 = np.arcsin(out.params['wavelen'].value/2.0/d_pg002)*180/np.pi

        print("\n\nThe current m1, m2 and s2 offset values are:\n m1: {:6.8f} \n m2: {:6.8f} \n s2 offset: {:6.8f}".format(m1, 2*m1, out.params['s2_offset'].value))
        calibr_result={"m1": m1, "m2": 2*m1, "s2_offset": out.params['s2_offset'].value, "wavelen": out.params['wavelen'].value, }
        return calibr_result



    def taipan_expconfig(self, ef=14.87, aa=4, bb=5, cc=6, ang_a=90, ang_b=90, ang_c=90, uu=np.array([1, 0, 0]), vv=np.array([0, 0, 1])):
        taipan_exp = super().tas_expconfig(ef)
        taipan_exp.method   =   1  # 1 for Popovici, 0 for Cooper-Nathans
        taipan_exp.moncor   =   1
        taipan_exp.efixed   =   ef
        taipan_exp.infin    =  -1    #const-Ef
        
        taipan_exp.mono.dir =  -1
        taipan_exp.sample.dir =   1
        taipan_exp.ana.dir  =  -1
        

        #Put the sample information below
        taipan_exp.sample.a     = aa
        taipan_exp.sample.b     = bb
        taipan_exp.sample.c     = cc
        taipan_exp.sample.alpha = ang_a
        taipan_exp.sample.beta  = ang_b
        taipan_exp.sample.gamma = ang_c
        taipan_exp.sample.u     = uu
        taipan_exp.sample.v     = vv

        taipan_exp.arms    =  [202, 196, 179, 65, 106]   
        taipan_exp.orient1 =  uu
        taipan_exp.orient2 =  vv                #need to be updated

        self.exp = taipan_exp

        return taipan_exp



    def taipan_conv_init(self, hklw=None, exp = None,  initial=None, fixlist=[0,0,0,0,0,0,0,0], magion= "Mn2", sqw=SqwDemo, pref=PrefDemo, smoothfit=True):
        #numpy array is a dataframe and should be change into a numpy array
        if hklw is None:
            raise ValueError("hklw is None.") 
        if exp is None:
            exp = self.taipan_expconfig(ef = 14.87)

        if initial is None:
            raise ValueError("initial is None.")

        [H, K, L, W, Iobs] =  hklw.to_numpy().T  #split into 1D arrays 
        #dIobs   = np.sqrt(Iobs)

        ffactor=SelFormFactor(magion)
        if ffactor is None:
            print("No Form Factor is used.")
        else:
            AA=ffactor["AA"]
            aa=ffactor["aa"]
            BB=ffactor["BB"]
            bb=ffactor["bb"]
            CC=ffactor["CC"]
            cc=ffactor["cc"]
            DD=ffactor["DD"]
            initial = list(initial) + [AA, aa, BB, bb, CC, cc, DD]  #new initial list
            print(f" {magion} Form Factor is used.")
     

        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)     
        sim_init = exp.ResConv(sqw, pref, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=[5,5], p=initial)
        simhklw=np.column_stack([newH,newK,newL,newW,sim_init])

        return simhklw
    

    def taipan_convfit(self, hklw=None, exp = None,  initial=None, fixlist=[0,0,0,0,0,0,0,0], magion = "none", sqw=SqwDemo, pref=PrefDemo):
        #numpy array is a dataframe and should be change into a numpy array
        if hklw is None:
            raise ValueError("hklw is None.") 
        if exp is None:
            exp = self.taipan_expconfig(ef = 14.87)

        if initial is None:
            raise ValueError("initial is None.") 

        [H, K, L, W, Iobs] =  hklw.to_numpy().T  #split into columns
        dIobs=np.sqrt(Iobs)
        #load the FF parameters if magion is given
        ffactor=SelFormFactor(magion)
        if ffactor is None:
            print("No Form Factor is used.")
        else:
            AA=ffactor["AA"]
            aa=ffactor["aa"]
            BB=ffactor["BB"]
            bb=ffactor["bb"]
            CC=ffactor["CC"]
            cc=ffactor["cc"]
            DD=ffactor["DD"]
            initial = list(initial) + [AA, aa, BB, bb, CC, cc, DD]
            fixlist = fixlist       + [ 0,  0,  0,  0,  0,  0,  0]
            print(f" {magion} Form Factor is used.")

        fitter = UltraFastFitConv(exp, sqw, pref, [H,K,L,W], Iobs, dIobs)
        result = fitter.fit_ultrafast(param_initial=initial, param_fixed_mask=fixlist,maxfev=200,use_analytical_jacobian=True,early_stopping=True,verbose=True)
        
        final_params = result['params']
        # To produce a smooth fitted curve
        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)
        final = exp.ResConv(sqw, pref, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=None, p=final_params)
        fittedHKLW=np.column_stack([newH,newK,newL,newW,final])
        
        return result, fittedHKLW


    def taipan_phonon_convfit(self, hklw=None, exp = None,  initial=None, fixlist=[0,0,0,0,0,0,0,0], sqw=SqwDemo, pref=PrefPhononDemo):
        #numpy array is a dataframe and should be change into a numpy array
        if hklw is None:
            raise ValueError("hklw is None.") 
        if exp is None:
            exp = self.taipan_expconfig(ef = 14.87)

        if not isinstance(initial, (list)):
                raise ValueError("initial must be a list.") 

        [H, K, L, W, Iobs] =  hklw.to_numpy().T  #split into columns
        dIobs=np.sqrt(Iobs)

        fitter     =    FitConv(exp,  sqw,  pref,  [H,K,L,W],  Iobs,  dIobs)

        [final_param, dpa, chisqN, sim, CN, PQ, nit, kvg, details] = fitter.fitwithconv(exp, sqw, pref, [H,K,L,W], Iobs, dIobs, param=initial, paramfixed=fixlist)

        parlist = ['en1','en2','ratio', 'w1','w2','int', 'bg', 'temp']
        str_output = "The fitted parameters:\n"

        for index, (iname, ipar, ierr) in enumerate(zip(parlist, final_param, dpa)):
            str_output=str_output+"P{0}( {1} ): {2:6.8f} +/- {3:6.8f}\n".format(index, iname, ipar, ierr)

        print(str_output)

        # to produce a smooth fitted curve
        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)
        final      = exp.ResConv(sqw, pref, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=None, p=final_param)
        fittedHKLW = np.column_stack([newH,newK,newL,newW,final])
        
        return final_param, fittedHKLW
    

    def taipan_batch_validate_new(self, filename=None, bsim=False, exp=None):
        """
        Reads and validates commands from a file.
        Returns a list of tuples: (line_number, command, is_valid, message)
        """
        if filename is None:
            print("please give the batch files!")
            return
        #valid_cmd_list = []

        results = []
        validator = TaipanCommandValidator()
        
        try:
            with open(filename, 'r') as file:
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
                            print(f"Line {i}: {line} \n Validation Passed.\n")
                            results.append((i + 1, line, is_valid, message))
                            i += 1
                            if bsim :
                                if exp is None:
                                    print("No experiment is provided.")
                                else:
                                    simres=self.taipan_scansim(line,exp=exp)
                                    print(simres)
                        else:
                            print("Invalid cmd!")
                    # Handle drive command
                    elif line.startswith('drive'):
                        # Check if next line exists and is a runscan
                        if i + 1 < len(lines) and lines[i + 1].strip().startswith('runscan'):
                            combined_cmd = f"{line}\n{lines[i + 1]}"
                            is_valid, message = validator.validate_command(combined_cmd)
                            if is_valid:
                                print(f"Line {i} and {i+1}: {combined_cmd} \n Validation Passed.\n")
                                #print(combined_cmd+" is validated.")
                                results.append((i + 1, combined_cmd, is_valid, message))
                                if bsim :
                                    if exp is None:
                                        print("No experiment is provided.")
                                    else:
                                        simres=self.taipan_scansim(combined_cmd, exp=exp)
                                        print(simres)
                                i += 2  # Skip both lines
                            else:
                                print(f"Line {i+1}: {lines[i + 1]} \n Validation Failure: Invalid runscan.\n")
                                i += 1

                        else:
                            # Handle single drive command
                            is_valid, message = validator.validate_command(line)
                            results.append((i + 1, line, is_valid, message))
                            i += 1
                            #print("This is a single drive cmd.")
                            print(f"Line {i}: {line} \n This is a single drive cmd.\n")
                    
                    else:
                        # Invalid command type
                        results.append((i + 1, line, False, "Invalid scan command"))
                        print(f"Line {i+1}: {line} \n Validation False. Invalid scan command.\n")
                        i += 1
                        
        except FileNotFoundError:
            return [(0, "", False, f"File not found: {filename}")]
        except Exception as e:
            return [(0, "", False, f"Error reading file: {str(e)}")]
        
        return results


    def taipan_batch_validate(self, batchfile=None):
        # Now read and process the file
        if batchfile is None:
            print("please give the batch files!")
            return
        valid_cmd_list = []
        validation_result = ""
        validator = TaipanCommandValidator()
        try:
            with open(batchfile, 'r') as file:
                lines = file.readlines()
                
            # Process lines
            i = 0
            while i < len(lines):
                # Skip empty or whitespace-only lines
                if not lines[i].strip():
                    i += 1
                    continue
                    
                current_line = lines[i].strip()
                print(f"\nProcessing line {i + 1}:")
                validation_result = validation_result+ f"\nLine {i + 1}:\n"
                #print(f"Content: {current_line}")
                
                if current_line.startswith('mscan'):
                    # Process mscan command
                    valid, message = validator.validate_command(current_line)
                    #print(f"Command type: mscan")
                    #print(f"Valid: {valid}")
                    #print(f"Message: {message}")
                    print(f"Command: {current_line}")
                    
                    
                    if valid:
                        print(f"Validation passed!")
                        validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Validation passed!\n\n"
                        valid_cmd_list.append(current_line)
                    else: 
                        print(f"Error: mscan error.")
                        validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Error: mscan error!\n\n"
                    i += 1
                    
                elif current_line.startswith('drive'):
                    # Look ahead for runscan
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    
                    if next_line.startswith('runscan'):
                        # Process drive+runscan as a group
                        combined_cmd = f"{current_line}\n{next_line}"
                        valid, message1 = validator.validate_command(combined_cmd)
                        #print(f"Command type: drive+runscan group")
                        #print(f"Message: {message}")
                        print(f"Combined command:\n{combined_cmd}")
                        
                        if valid:
                            print(f"Validation passed!")
                            validation_result = validation_result + f"Combined command:\n{combined_cmd}" + "\n" + f"Validation passed!\n\n"
                            valid_cmd_list.append(current_line)
                            valid_cmd_list.append(next_line)
                        else: 
                            print(f"Error: combined drive-runscan error!")
                            validation_result = validation_result + f"Combined command:\n{combined_cmd}" + "\n" + f"ERROR:combined drive-runscan error!\n\n"
                        i += 2  # Skip both lines
                    else:
                        # Process single drive command
                        valid, message = validator.validate_command(current_line)
                        #print(f"Command type: drive (single)")
                        #print(f"Message: {message}")
                        print(f"Command: {current_line}")
                        
                        if valid:
                            print(f"Validation passed!")
                            validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Validation passed!\n\n"
                        else: 
                            print(f"Error: drive command error.")
                            validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"ERROR: drive command error!\n\n"
                        i += 1

                elif current_line.startswith('title'):   
                    valid, message = validator.validate_command(current_line)
                    #print(f"Command type: title")
                    #print(f"Valid: {valid}")
                    #print(f"Message: {message}")
                    print(f"Command: {current_line}")
                    
                    if valid:
                        print(f"Validation passed!")
                        validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Validation passed!\n\n"
                    else: 
                        print(f"Error: title command error.")
                        validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Error: title command error!\n\n"

                    i += 1

                else:
                    # Invalid command type
                    print(f"Command: {current_line}")
                    print(f"ERROR: unknow command!")
                    validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"ERROR: unknow command!\n\n"
                    i += 1
                    
        except FileNotFoundError:
            print("Error: Test file not found")
        except Exception as e:
            print(f"Error processing file: {str(e)}")

        return validation_result, valid_cmd_list


    def taipan_scansim(self, cmdline="", exp=None, runscanpos=[2, 0, 0, 0]):
        # example: runscan s1 0 10 21 time 1 /monitor  10000
        pd.set_option('display.expand_frame_repr', False)
        if not cmdline:
            print("no cmd")
            return
        
        if exp is None:
            print("There is no experiment.")
            return

        cmd_lines= cmdline.splitlines()
        cmd_items= cmd_lines[0].split()
        print(cmdline)

        if cmd_items[0] == 'drive':
            #drive qh 0 qk 0 ql 2 en 2 \nrunscan s1 0 10 21 time 1 (or monitor  10000)
            if len(cmd_lines) >= 2:
                sim_result = self.taipan_runscansim(cmd_lines[0],cmd_lines[1], exp)
                #print("sim runscan finished")
                return sim_result
            else:
                print("ERROR: This is only a drive command. No runscan is found")
                return

        elif cmd_items[0] == 'mscan':
            # mscan qh 0 0 qk 0 0 ql 0.9 0.005 en 0 0 21 time monitor 10
            sim_result = self.taipan_mscansim(cmd_lines[0], exp)
            #print("sim finished")
            return sim_result
        else:
            print("ERROR: this is not a scan command.")
            return 
 

    def taipan_mscansim(self, mscanline="", exp = None):
        # example: runscan s1 0 10 21 time 1 /monitor  10000

        if not exp:
            print("There is no experiment.")
            return

        if not mscanline:
            print("no cmd")
            return
        
        motorlist = ['m1', 'm2', 's1', 's2', 'a1', 'a2', 'qh', 'qk', 'ql', 'en']
        df_scanpos = pd.DataFrame([])
        validator = TaipanCommandValidator()
        mscan_valid, message = validator.validate_command(mscanline)
        #print(mscan_valid)
        #print(message)
        if mscan_valid:
            cmd_items= mscanline.split()

            if cmd_items[0] == 'mscan':
                # mscan qh 0 0 qk 0 0 ql 0.9 0.005 en 0 0 21 time monitor 10
                if len(cmd_items) <  7:
                    print('The current command is invalid! Please check it if it follows the form: mscan motor1 start1 step1 motor2 start2 step2 numberofsteps time(or monitor) nnnnn.')
                    return
                
                int_part, dec_part = divmod(len(cmd_items)-4, 3)
                if dec_part > 0.000001:
                    print('The number of the parameters is not correct!')
                    return

                for ii in range(int((len(cmd_items)-4)/3)):
                    if cmd_items[3*ii+1] in motorlist:
                        if not strisfloat(cmd_items[3*ii+2]) or  not strisfloat(cmd_items[3*ii+3]):
                            print("Motor {} start position and stepsize is not correct".format(ii+1))
                            return
                    else:
                        print("The motor {} is not in the list".format(cmd_items[ii+1]))
                        return

                if not strisint(cmd_items[-3]):
                    print("The number of steps must be an integer.")
                    return
                if cmd_items[-2] != 'time' and cmd_items[-2] != 'monitor':
                    print("The scan must count against time or monitor.")
                    return
                #df_scanpos = pd.DataFrame([])
                numstep    = int(cmd_items[-3])
                
                for ii in range(int((len(cmd_items)-4)/3)):
                    start      = float(cmd_items[3*ii+2])
                    stepsize   = float(cmd_items[3*ii+3])
                    temppos    = []
                    for jj in range (numstep+1):
                        temppos.append(start+jj*stepsize)
                    #step= (end-start)/(numstep-1)
                    #temppos=np.arange(start, start+numstep*stepsize, stepsize)
                    #print(temppos)
                    df_scanpos[cmd_items[3*ii+1]] = temppos

                # if it is qh qk ql en scan, print the angles too    
                if  'qh' in df_scanpos and 'qk' in df_scanpos and 'ql' in df_scanpos and 'en' in df_scanpos:
                    hkle = df_scanpos[['qh', 'qk','ql','en']].to_numpy().T
                    [M1, M2, S1, S2, A1, A2, Q] = exp.get_spec_angles(hkle)
                    df_scanpos['m1'] = M1
                    df_scanpos['m2'] = M2
                    df_scanpos['s1'] = S1
                    df_scanpos['s2'] = S2
                    df_scanpos['a1'] = A1
                    df_scanpos['a2'] = A2
            
        #pd.set_option('display.expand_frame_repr', False)
        #print(df_scanpos)
        
        return df_scanpos
    
    def taipan_runscansim(self, driveline=None, scanline=None, exp=None):
        # example: runscan s1 0 10 21 time 1 /monitor  10000
        
        if not exp :
            print("There is no experiment.")
            return

        if not driveline:
            print("no drive cmd")
            return
        if not scanline:
            print("no cmd")
            return

        df_scanpos = pd.DataFrame([])

        motorlist = ['m1', 'm2', 's1', 's2', 'a1', 'a2', 'qh', 'qk', 'ql', 'en']

        validator = TaipanCommandValidator()


        #drive_valid, message_a   = validator.validate_command(driveline)
        runscan_valid, message_b = validator.validate_command(driveline + '\n' + scanline) #pass the double line to the validator


        newcmd = "mscan"

        if runscan_valid:

            drive_items = driveline.split()
            scan_items   = scanline.split()

            try:
                motor_index = drive_items.index(scan_items[1])
                #print(f"{target} found at index {index}.")
                for index, item in enumerate(drive_items):
                    if index % 2 == 1:
                        if index != motor_index:
                            newcmd = newcmd + ' ' + drive_items[index] + ' ' + drive_items[index+1] + ' ' + '0.0'

                        else:
                            startpos = float(scan_items[2])
                            stoppos = float(scan_items[3])
                            stepno = float(scan_items[4])
                            if stepno > 0:
                                stepsize= (stoppos-startpos)/stepno
                                newcmd = newcmd + ' ' + drive_items[index] + ' ' + scan_items[2] + ' {}'.format(stepsize)
                            else: 
                                print("the step number must be an positive integer")
                newcmd = newcmd + ' '  + scan_items[4] +  ' '  + scan_items[5] + ' '  + scan_items[6]
                #print(newcmd)

                sim_result = self.taipan_mscansim(newcmd, exp)
                return sim_result

            except ValueError:
                print(f"The scan axis {scan_items[1]} not found in drive command.")
                return





    def taipan_gen_escans(self, hkl_list=[[0,0,1],[0,0,1.5]], e_range=[[0,5], [0,6]], e_step = 0.1, sample_T=1.5, beammon= 1000000, titleTemplate='ABO3 HH0-00L'):
        #generate batch energy scans at a list of Q positions
        batch_str=''
       
        if len(hkl_list) == len(e_range):

            for index, hkl in enumerate( hkl_list ):
                steps = 1 + round((e_range[index][1]- e_range[index][0])/e_step)
                scantitle = 'title \"' + titleTemplate+' Q({}, {}, {}) Escan ({} ~ {}meV) at {} K\"\n'.format(hkl[0],hkl[1], hkl[2], e_range[index][0], e_range[index][1], sample_T)
                drivestr  =  'drive qh {} qk {} ql {} en {}\n'.format(hkl[0],hkl[1], hkl[2], e_range[index][0])
                scanstr   =  'runscan en {} {} {} monitor {} \n\n'.format(e_range[index][0],e_range[index][1], steps, beammon)
                batch_str = batch_str + scantitle +drivestr +scanstr
                print(batch_str)
        else:
            print('error: The numbers of Q and E do not match!')

        return batch_str


    def taipan_gen_qscans(self, startQ=[[0,0,1.1],[0.1,0,1.5]], endQ=[[0,0,1.6],[0.9,0,1.5]], q_step = [[0.00, 0.00, 0.01],[0.01, 0.00, 0.00]], en=2, sample_T=1.5, beammon=1000000, titleTemplate='ABO3 HH0-00L'):
        #generate batch Q scans from the startQ to end Q with q_step size at the energy transfer of en
        batch_str=''
        qNameList=['qh', 'qk', 'ql']

        if len(startQ) == len(endQ):

            for index, hkl in enumerate( startQ ):
                dH = abs(startQ[index][0]  - endQ[index][0])
                dK = abs(startQ[index][1]  - endQ[index][1])
                dL = abs(startQ[index][2]  - endQ[index][2])

                deltaList=[dH, dK, dL]
                #print(deltaList)
                scanQList=[]
                fixQList =[]
                midQ=[(startQ[index][0] + endQ[index][0])/2, (startQ[index][1] + endQ[index][1])/2, (startQ[index][2] + endQ[index][2])/2 ]
                

                for idx, delta in enumerate(deltaList):
                    if delta >= 0.0001:
                        scanQList.append(idx)
                    if delta < 0.0001:
                        fixQList.append(idx)

                if len(scanQList) ==0:
                    print("error: startQ and end Q is the same!")
                    return batch_str
                elif len(scanQList) ==1:
                    if q_step[index][scanQList[0]]== 0:
                        print("error: step could not be zero!")
                    else:
                        steps = 1 + round((endQ[index][scanQList[0]] - startQ[index][scanQList[0]])/q_step[index][scanQList[0]])
                        scantitle = 'title \"' + titleTemplate+' Qscan '+ qNameList[scanQList[0]]
                        scantitle = scantitle + '({}~{}) around '.format(startQ[index][scanQList[0]],endQ[index][scanQList[0]])
                        scantitle = scantitle + 'Q({}, {}, {}) at E = {}meV, {}K\"\n'.format(midQ[0], midQ[1], midQ[2], en, sample_T)
                        drivestr  = 'drive qh  {}  qk  {}  ql  {}  en  {}\n'.format(startQ[index][0],startQ[index][1], startQ[index][2], en)
                        scanstr   = 'runscan ' + qNameList[scanQList[0]] + '  {}  {}  {}  monitor  {}\n\n'.format(startQ[index][scanQList[0]], endQ[index][scanQList[0]], steps, beammon)
                        batch_str = batch_str  + scantitle + drivestr + scanstr
                        print(batch_str)

                elif len(scanQList) ==2 or len(scanQList) ==3:
                    #when we scan two or three axes of HKL, use mscan
                    if q_step[index][0]== 0 and q_step[index][1]== 0 and q_step[index][2]== 0 :
                        print("error: at least one step could not be zero!")
                    else:
                        steps = 1 + round((endQ[index][scanQList[0]] - startQ[index][scanQList[0]])/q_step[index][scanQList[0]])
                        scantitle = 'title \"' + titleTemplate+' Qscan '+ qNameList[scanQList[0]]
                        scantitle = scantitle + '({}~{}) around '.format(startQ[index][scanQList[0]],endQ[index][scanQList[0]])
                        scantitle = scantitle + 'Q({}, {}, {}) at E = {}meV, {}K\"\n'.format(midQ[0], midQ[1], midQ[2], en, sample_T)
                        #drivestr  = 'drive qh {} qk {} ql {} en {}\n'.format(startQ[index][0],startQ[index][1], startQ[index][2], en)
                        scanstr   = 'mscan qh  {}  {}  '.format(startQ[index][0], q_step[index][0])
                        scanstr   = scanstr + 'qk  {}  {}  '.format(startQ[index][1], q_step[index][1])
                        scanstr   = scanstr + 'ql  {}  {}  '.format(startQ[index][2], q_step[index][2])
                        scanstr   = scanstr + 'en  {}  {}  {}  monitor  {}\n\n'.format(en, 0, steps, beammon)
                        batch_str = batch_str  + scantitle  + scanstr
                        print(batch_str)


        return batch_str