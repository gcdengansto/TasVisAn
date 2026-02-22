from pathlib import Path
import pandas as pd
import numpy as np
import re
#import io


from ..base import TasData
from ..utils.toolfunc import (AnglesToQhkl, strisfloat, strisint,  
                gaussian_residual, lorentzian_residual, fit_peak, angle2, SqwDemo, PrefDemo, SelFormFactor, PrefPhononDemo)
from .validator import SikaCommandValidator

import inspy as npy
from inspy import TripleAxisSpectr
from inspy.insfit import FitConv, UltraFastFitConv

class Sika(TasData):
    """Sika class that extends TasData."""
    
    def __init__(self, expnum, title, sample, user):
        super().__init__("Sika", expnum, title, sample, user)
        #self.specific_value = specific_value
    

    

    def _sika_data_header_info(self, path):
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
        last_header_line = ""
        last_header_line_num = 0
        data_start_line = 0
        line_count = 0
        
        # Special handling for arrays
        array_params = ['latticeconstants', 'ubmatrix', 'plane_normal']
        
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
                        value = [float(item) for item in value.split(',')]
                    
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

        
        if 'steps' in params:
            str_steps = params['steps']

            # Use regex to find all substrings before the colons
            matches = re.findall(r'([^:]+):', str_steps)
             
            if len(matches) == 1:
                params['scanax1']=matches[0]
                params['scanax2']=None
            elif len(matches) == 2:
                params['scanax1']=matches[0]
                params['scanax2']=matches[1]
            else:
                params['scanax1'] = params.get('def_x', 'Pt.')
                params['scanax2']=None
        else:
            params['scanax1'] = params.get('def_x', 'Pt.')
            params['scanax2'] = None

        if params['scanax1'] == "":
            params['scanax1'] = "Pt."
            print("Scan axis is empty! Please check. it is temperarily set to Pt.")
        if params['scanax1'] in ['h', 'k', 'l']:
            params['scanax1']= "q"+ params['scanax1']
        elif params['scanax1'] in ['e']:
            params['scanax1']= "en"

            #print("no or too many scanning axis is found")

        return params


    def sika_data_to_pd(self, path='', scanno=None):
        """
        Load a single neutron data file into a pandas DataFrame with metadata in attrs.
        Args:
            path (str): Path to the data file
        Returns:
            pandas.DataFrame: DataFrame containing the data with metadata in attrs
        """
        ppath=Path(path)

        if not ppath.is_dir():
            print(ppath)
            print("Error: wrong path!")
            return

        if scanno is None:
            print("Error: No scan no. is given.")
            return

        basestr = "000000"
        basestrexp="0000"
        
        scanno = str(int(scanno)) 
        expno = str(int(self.expnum)) 
        filename = basestrexp[:-len(expno)]+expno  + "_" + basestr[:-len(scanno)] + scanno + '.dat'
        fullfilepath = ppath / Path(filename)

        # Parse header information
        params = self._sika_data_header_info(fullfilepath)
        #print(params)
        
        # Get file structure information
        data_start_line = params.pop('_data_start_line', 0)
        header_line = params.pop('_header_line', '')
        header_line_num = params.pop('_header_line_num', 0)
        
        # Process the header line to extract column names
        if header_line.startswith('#'):
            # Remove the leading # and split by whitespace
            header_line = header_line[1:].strip()
        # Create column names from the header line
        if header_line:
            # Clean and split the header line
            sika_column_names = [col.strip() for col in header_line.split() if col.strip()]
        #print(sika_column_names)
        sika_motorname_list = ['h', 'k', 'l','e', 'count', 'beamMon','tc1a','tc2a','magneta']
        stdtas_motorname_list = ['qh', 'qk', 'ql','en', 'detector', 'monitor', 'tempVTI', 'tempSAMP','magField']

        # Create a mapping from old to new
        replace_map = dict(zip(sika_motorname_list, stdtas_motorname_list))

        # change into standard column names
        std_column_names = [replace_map.get(item, item) for item in sika_column_names]

        # Try to read the data part using pandas
        try:
            # No header line found, use pandas to read with default column names
            df = pd.read_csv(fullfilepath, sep=r'\s+', header=None, comment='#')
            df.columns = std_column_names

        except Exception as e:
            print(f"Warning: Error parsing data section. Trying alternative method. Error: {e}")
            # Create an empty DataFrame as last resort
            df = pd.DataFrame()
    
        # Store all metadata in the DataFrame's attrs
        df.attrs = params
        
        return df  


    def sika_scanlist_to_dflist(self, path="", scanlist=None):
        if scanlist is None:
            raise ValueError("scanlist cannot be None")

        dflist=[]

        if all(isinstance(x, (int, np.integer)) for x in scanlist):
            for scanno in scanlist:
                df  = self.sika_data_to_pd(path, scanno)
                dflist.append(df)

            #print(f"Totally, {len(dflist)} scans in the list are converted into df.")
            return dflist   
        else:
            print("Some element in the list is not integer. sublist in the list is not allowed.")
            return

    def sika_simpleplot(self, path="", scanlist=None, fit=False, initial=None):
        #quick plot without normalization and combination
        dflist = self.sika_batch_reduction(path, scanlist)
        #print(dflist)
        parlist, fitlist, fig, ax=self.tas_simpleplot(dflist, fit, initial)

        return parlist, fitlist, fig, ax


    def sika_combplot(self, path='', scanlist=None, fit=False, norm_mon_count = -999999, overplot=False, offset=1000, initial=None ):
        #scanlist can be [123, 125, [130, 132],115]
        dflist   = list()
        for scanno in scanlist:
            if isinstance(scanno, list):
                subscanlist = scanno
                subdflist   = self.sika_scanlist_to_dflist(path, subscanlist)
                comb_df  =  super().tas_datacombine(subdflist)
                dflist.append(comb_df)
            if isinstance(scanno, (int, np.integer)):
                single_df = self.sika_data_to_pd(path, scanno)
                dflist.append(single_df)
        
        params_df, fitted_df, fig, ax = super().tas_combplot(dflist, fit, norm_mon_count, overplot, offset, initial)

        return params_df, fitted_df, fig, ax


    def sika_batch_reduction(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'ei','ef', 'm1', 'm2','s1','s2', 'a1', 'a2', 'detector', 'monitor','tempVTI', 'tempSAMP']):
        #scanlist can be [123, 125, [130, 132],115]
        dflist     = list()
        col_names = motorlist #['h', 'k', 'l', 'e', 'q',  'm1', 'm2','s1','s2', 'a1', 'a2', 'sgl', 'sgu', 'stu', 'stl', 'sd','dd','i3h','i3v','i4h','i4v','Pt.','i1l','i1r', 'mfh', 'mfv', 'c4s','c4r', 'ei', 'ef', 'count', 'beamMon']
 
        for scanno in scanlist:
            #print(f"the problem here:{scanno}")
            #print(isinstance(scanno, int))  # this will be false, int is not np.integer the same
            if isinstance(scanno, (int, np.integer)):
                df = self.sika_data_to_pd(path, scanno)
            
            elif isinstance(scanno, list):
                subscanlist = scanno
                temp_dflist=self.sika_scanlist_to_dflist(path, subscanlist)
                df=super().tas_datacombine(temp_dflist)
            else:
                print("the elemenet in the list is not int or list.")
                return
            col_notexist=[]
            
            #print(df.columns)
            for col in col_names:
                if col not in df.columns:
                    col_notexist.append(col)
            # remove those col_names that is not in the data file in case. 
            col_names= [x for x in col_names if x not in col_notexist] 
            if col_notexist:
                print(f"These motors are not in the data:{col_notexist}")
            dflist.append(df[col_names])
                
        return dflist  # return reduced columns

    def sika_reduction_by_row(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'ei','ef', 'm1', 'm2','s1','s2', 'a1', 'a2', 'detector', 'monitor','tempVTI', 'tempSAMP'], sortby="tempSAMP"):
        dflist = self.sika_batch_reduction(path, scanlist, motorlist)
        df_extend = pd.DataFrame(columns=motorlist)

        for df in dflist:
            if df is not None and not df.empty and not df.isna().all().all():
                df_extend=pd.concat([df_extend, df], ignore_index=True)
        if sortby in motorlist:
            df_extend = df_extend.sort_values(by=sortby, ascending = True).reset_index(drop=True)
        return df_extend

    def sika_reduction_anycolumn(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'ei','ef', 'm1', 'm2','s1','s2', 'a1', 'a2', 'detector', 'monitor','tempVTI', 'tempSAMP'], sortby="tempSAMP"):
        dflist = self.sika_batch_reduction(path, scanlist, motorlist)
        df_extend = pd.DataFrame(columns=motorlist)

        for df in dflist:
            if df is not None and not df.empty and not df.isna().all().all():
                df_extend=pd.concat([df_extend, df], ignore_index=True)
        if sortby in motorlist:
            df_extend = df_extend.sort_values(by=sortby, ascending = True).reset_index(drop=True)
        return df_extend  

    def sika_random_contour(self, path='', scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'detector', 'monitor'], x_col="qh", y_col="en", xtitle='qh [rlu]', ytitle='en [meV]',title='Contour Map of Measurement Data', vminmax=[0, 1000], output_file=None):
        df_total = self.sika_reduction_by_row(path, scanlist, motorlist=motorlist)
        ax = super().tas_random_contour(df_total, x_col=x_col, y_col=y_col, xtitle=xtitle, ytitle=ytitle, title=title, vminmax=vminmax, output_file=output_file)
        return ax

    def sika_tidy_contour(self, path='',  scanlist=None, motorlist=['qh', 'qk', 'ql', 'en', 'detector', 'monitor'], x_col='qh', y_col='en', xlabel='Q [rlu]',ylabel='E [meV]', vminmax=None, ax=None):
       
        dflist = self.sika_batch_reduction(path, scanlist, motorlist=motorlist)
        qq, ee, intt, ax = super().tas_tidy_contour(dflist,  x_col=x_col, y_col=y_col, xlabel=xlabel, ylabel=ylabel, vminmax=vminmax, ax=ax)
        return qq, ee, intt, ax


    def sika_export_hklw(self, path='', scanlist = None, hklw_file=""):
        #this is advanced export function, which can combine data and then export
        #scanlist could be [ 2, [3, 4], 5, 6] or [ 12, 23, 34, 45]
        dflist = self.sika_batch_reduction(path, scanlist, motorlist=['qh', 'qk', 'ql', 'en', 'detector'])
        
        super().tas_export_hklw(dflist, hklw_file)

        return dflist


    def export_scanlist(self, path='', datafromto=None, outputfile=''):
        # export the scan list from sika
        # path is the Datafiles folder
        # datafromto is a list with two elements, integers
        # outputfile is the filename for the list
        
        ppath=Path(path)

        if not ppath.is_dir():
            print("Error: wrong path!")
            return

        if datafromto is None:
            print("Error: No data from and to number is given")
            return
        elif datafromto[0] <= 0 or not isinstance(datafromto[0], int) or datafromto[1] <= 0 or not isinstance(datafromto[1], int):
            print("Error: Please specify positive integer for datafrom and to numbers!")
            return
        else:
            pass

        if outputfile:
            outputfile = ppath / Path(outputfile)
        else:
            outputfile = ppath / Path( "Sika_exp" + self.expnum  + "_scanfrom" + str(datafromto[0]) + "to" + str(datafromto[1]) + "list.html")

        fileext    = ".dat"

        filenolist = np.linspace(datafromto[0], datafromto[1], (datafromto[1]-datafromto[0]+1))

        #print(filenolist)
        scanno_list    = []
        command_list   = []
        scantitle_list = []

        basestr = "000000"

        for fileindex, fileno in enumerate (filenolist):
            filename = str(int(fileno))
            filename = basestr[:-len(filename)] + filename
            expnumstr = "0000"

            expnumstr=expnumstr[0:4-len(str(self.expnum))] + str(self.expnum)
                
            fullpath = ppath / Path(expnumstr + "_" + filename + fileext)
            #print(fullpath)

            scanno_list.append(int(fileno))
            try:
            with fullpath.open() as f:
                totallines = list(f)                              #read the lines in the file f, initialize a list
                for index, line in enumerate (totallines):

                    if (line.find("# command =") != -1):
                        command_list.append(line[11:-1])
                        #print(command_list)
                        #print(index)
                    if (line.find("# scan_title =")!= -1):
                        scantitle_list.append(line[14:-1])

            except IOError as e:
                print(f"Couldn't open file: {e}")
                command_list.append("error")
                scantitle_list.append("error")


        scan_dict ={"scanno": scanno_list, "command": command_list, "scantitle": scantitle_list}
        scanlist=pd.DataFrame(scan_dict)
        scanlist.to_html(str(outputfile))
        #print(scanlist)
        ### end of the function###
        return scanlist


    def sika_multichannel_config(self, total_channel_number=0, total_det_number=0, all_detname_list=None, all_det_eff_list=None, s2_offset_list=None, det_channel_list=None):
        if total_channel_number <3:
            total_channel_number = 9
        if total_det_number <3: 
            total_det_number = 48
        if all_detname_list is None:
            tubenumber=list(range(total_det_number))
            all_detname_list=[]
            for ii in tubenumber:
                all_detname_list.append("tube"+str(int(ii+1)))
            #print(all_detname_list)
        if all_det_eff_list is None:
            all_det_eff_list = [1.492784922, 0.679198979, 0.749288309, 0.916560322, 0.933051929, 0.700696967,
                                0.764798272, 1.005497202, 1.022577795, 0.675959556, 0.879257878, 0.954648081,
                                1.113968784, 0.59890056,  0.984490036, 0.839893983, 0.852851674, 0.734269167,
                                0.735250810, 0.991557868, 0.855207618, 0.744576421, 0.721409640, 1.000000000,
                                1.041229017, 0.779326593, 0.862569942, 0.926376755, 1.073328752, 0.654068911,
                                1.000588986, 0.945322470, 0.875134976, 0.822518897, 0.722685776, 0.982035928,
                                0.967605772, 0.826838127, 0.703936390, 0.948365564, 0.935898694, 0.726808678,
                                0.892019240, 0.928536370, 1.061058211, 0.711789536, 1.114459605, 1.278099539 ]
        if s2_offset_list is None:
            s2_offset_list = [-10, -7.5, -5.0, -2.50, 0, 2.5, 5.0, 7.5, 10.0]
        if det_channel_list is None:
            det_channel_list = [[7], [11], [15], [20], [24], [28], [32], [37], [41, 42]]

        super().tas_multichannel_config(total_channel_number, total_det_number,all_detname_list,all_det_eff_list,
        s2_offset_list,det_channel_list)
        return

    
    def sika_multichannel_reduction(self, path="", scanno=None):

        if self.multichannel == True:
            df = self.sika_data_to_pd(path, scanno)
            multichannel_dflist = super().tas_multichannel_reduction(df)
            return multichannel_dflist
        else:
            print("The multichannel has not been configured.")
            return


    def sika_multichannel_plot3d(self, path="", scanlist=None, xcol='qh', ycol='ql', zcol="en", perc_low=5, perc_high=95, xlim=None, ylim=None, zlim=None,xlabel="QH [r.l.u]", ylabel="QL [r.l.u]",zlabel="E [meV]"):
        """ put all the data points in dflist into X, Y, Z, Intensity, then plot in the 3d space with scatter. 
        """
        if scanlist is not None:
            alldflist=[]
            for scan in scanlist:
                df = self.sika_data_to_pd(path, scan)
                dflist = super().tas_multichannel_reduction(df)
                alldflist = alldflist + dflist
            X, Y, Z, A, ax= super().tas_multichannel_plot3d(alldflist,xcol='qh', ycol='ql', zcol="en", perc_low=5, perc_high=95, xlim=None, ylim=None, zlim=None,xlabel="QH [r.l.u]", ylabel="QL [r.l.u]",zlabel="E [meV]")
            return X, Y, Z, A, ax
        else:
            print("No scanlist is provided.")
            return


    def sika_expconfig(self, ef=5, aa=4, bb=5, cc=6, ang_a=90, ang_b=90, ang_c=90, uu=np.array([1, 0, 0]), vv=np.array([0, 0, 1])):
        sika_exp = super().tas_expconfig(ef)

        sika_exp.method   =   1  # 1 for Popovici, 0 for Cooper-Nathans
        sika_exp.moncor   =   1
        sika_exp.efixed   =   ef
        sika_exp.infin    =  -1    #const-Ef
        
        sika_exp.mono.dir =  1
        sika_exp.sample.dir = -1
        sika_exp.ana.dir  =  1
        

        #Put the sample information below
        sika_exp.sample.a    = aa
        sika_exp.sample.b    = bb
        sika_exp.sample.c    = cc
        sika_exp.sample.alpha = ang_a
        sika_exp.sample.beta  = ang_b
        sika_exp.sample.gamma = ang_c
        sika_exp.sample.u       = uu
        sika_exp.sample.v       = vv
        sika_exp.hcol =  [60, 60, 60, 60]
        sika_exp.arms =  [230, 203, 179, 39.5, 106]
        sika_exp.orient1 = uu
        sika_exp.orient2 = vv                #need to be updated


        return sika_exp


    def sika_conv_init(self, hklw=None, exp = None,  initial=None, fixlist=[0,0,0,0,0,0,0,0], magion= "Mn2", sqw=SqwDemo, pref=PrefDemo, smoothfit=True):
        #numpy array is a dataframe and should be change into a numpy array
        if hklw is None:
            raise ValueError("hklw is None.") 
        if exp is None:
            exp = self.sika_expconfig(ef = 5.0)

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
     

        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)     
        sim_init = exp.ResConv(sqw, pref, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=[5,5], p=initial)
        simhklw=np.column_stack([newH,newK,newL,newW,sim_init])
        
        return simhklw

    def sika_convfit(self, hklw=None, exp = None,  initial=None, fixlist=[0,0,0,0,0,0,0,0], magion = "none", sqw=SqwDemo, pref=PrefDemo):
        #numpy array is a dataframe and should be change into a numpy array
        if hklw is None:
            raise ValueError("hklw is None.") 
        if exp is None:
            exp = self.sika_expconfig(ef = 5)

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
            print(f"{magion} Form Factor is used.")

        fitter = UltraFastFitConv(exp, sqw, pref, [H,K,L,W], Iobs, dIobs)
        result = fitter.fit_ultrafast(param_initial=initial, param_fixed_mask=fixlist,maxfev=200,use_analytical_jacobian=True,early_stopping=True,verbose=True)
        
        final_params = result['params']
        # To produce a smooth fitted curve
        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)
        final      = exp.ResConv(sqw=sqw, pref=pref, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=None, p=final_params)
        fittedHKLW = np.column_stack([newH,newK,newL,newW,final])
        
        return result, fittedHKLW


    def sika_batchfile_validate(self, batchfile=None):
        # Now read and process the file
        if batchfile ==None:
            print("please give the batch files!")
            return
        valid_cmd_list    = []
        validation_result = ""
        validator         = SikaCommandValidator()
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
                print(f"Content: {current_line}")
                
                # Process mscan command
                valid, message = validator.validate_command(current_line)

                print(f"Command: {current_line}")
                
                
                if valid:
                    print(f"Validation passed!")
                    validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Validation passed"+f": {message}:\n\n"
                    valid_cmd_list.append(current_line)
                else: 
                    print(f"Error: Invalid")
                    validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Error: "+f"Message: {message}\n\n"
                i += 1
                    
        except FileNotFoundError:
            print("Error: Test file not found")
        except Exception as e:
            print(f"Error processing file: {str(e)}")

        return validation_result, valid_cmd_list

    def sika_batchcmd_validate(self, cmdlist=None):
        # Now read and process the file
        if cmdlist ==None:
            print("please give the batch files!")
            return
        valid_cmd_list = []
        validation_result = ""
        validator=SikaCommandValidator()
        try:

            i = 0
            for cmdline in cmdlist:
            #while i < len(lines):
                # Skip empty or whitespace-only lines
                if not cmdline.strip():
                    i += 1
                    continue
                    
                current_line = cmdline.strip()
                print(f"\nProcessing line {i + 1}:")
                validation_result = validation_result+ f"\nLine {i + 1}:\n"
                print(f"Content: {current_line}")
                

                # Process mscan command
                valid, message = validator.validate_command(current_line)
                #print(f"Command type: mscan")
                #print(f"Valid: {valid}")
                #print(f"Message: {message}")
                print(f"Command: {current_line}")
                
                
                if valid:
                    print(f"Validation passed!")
                    validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Validation passed"+f": {message}:\n\n"
                    valid_cmd_list.append(current_line)
                else: 
                    print(f"Error: Invalid")
                    validation_result = validation_result+ f"Command: {current_line}"+"\n"+f"Error: "+f"Message: {message}\n\n"
                i += 1
                    
        except FileNotFoundError:
            print("Error: Test file not found")
        except Exception as e:
            print(f"Error processing file: {str(e)}")

        return validation_result, valid_cmd_list
    

    def sika_scansim(self, command=None, exp=None):
        """
        Parse a scan command and return array of [h, k, l, e] positions
        Returns None if command is invalid
        """
        # First validate the command
        if not command:
            print("please give the command!")
            return
        if not exp:
            print("please give the exp!")
            return

        validator=SikaCommandValidator()

        # Initialize default values
        motor_values = {'h': None, 'k': None, 'l': None, 'e': None}
        scan_motor = None
        scan_range = None
        hpos = []
        kpos = []
        lpos = []
        epos = []
        n_steps    =   0
        last_steps = -99
        scan_motor=[]
        scan_range=[]
        fix_motor=[]
        fix_pos=[]

        df_scanpos = pd.DataFrame([])
        try:
            # Split command into parts
            parts = command.split()

            i = 1
            while i < len(parts):
                motor = parts[i].lower()
                if motor not in motor_values:
                    raise ValueError(f"Invalid motor: {motor}")
                    
                # Parse values
                if len(parts) > i + 3 and all(validator.is_number(parts[j]) for j in range(i+1, i+4)):
                    # This is the scanning motor (start, end, step)
                    scan_motor.append( motor)
                    start = float(parts[i+1])
                    end = float(parts[i+2])
                    step = float(parts[i+3])
                    scan_range.append([start, end, step])
                    if step == 0:
                        print("error: step size can not be zero.")
                        return None
                    
                    i += 4

                else:
                    # This is a fixed position
                    if i+1 > len(parts):
                        print("Error: Missing value for motor")
                        return None
                    fix_motor.append(motor)
                    fix_pos.append(float(parts[i+1]))
                    #motor_values[motor] = float(parts[i+1])
                    i += 2
            
            if not scan_motor:
                raise ValueError("No scanning motor specified")
            #print(f"scan_motor:{scan_motor}")

            for idx, motor in enumerate(scan_motor):

                n_steps = abs(int(round((scan_range[idx][1] - scan_range[idx][0]) / scan_range[idx][2]))) + 1
                    
                if last_steps == -99 or last_steps == n_steps:
                    if motor == 'h':
                        hpos = np.linspace(scan_range[idx][0], scan_range[idx][1], n_steps)
                    elif motor == 'k':
                        kpos = np.linspace(scan_range[idx][0], scan_range[idx][1], n_steps)
                    elif motor == 'l':
                        lpos = np.linspace(scan_range[idx][0], scan_range[idx][1], n_steps)
                    elif motor == 'e':
                        epos = np.linspace(scan_range[idx][0], scan_range[idx][1], n_steps)
                    last_steps = n_steps
                else:
                    #print("last_steps:{}".format(last_steps))
                    #print("n_steps:{}".format(n_steps))
                    print("ERROR: the step numbers in the multi-motor scan do not match!")
                    return None
            
            for idx, motor in enumerate(fix_motor):
                if   motor == 'h':
                    hpos = np.linspace(fix_pos[idx], fix_pos[idx], n_steps)
                elif motor == 'k':
                    kpos = np.linspace(fix_pos[idx], fix_pos[idx], n_steps)
                elif motor == 'l':
                    lpos = np.linspace(fix_pos[idx], fix_pos[idx], n_steps)
                elif motor == 'e':
                    epos = np.linspace(fix_pos[idx], fix_pos[idx], n_steps)


            if len(hpos) >0:
                if len(hpos) == len(kpos)  and  len(hpos) == len(lpos)   and   len(hpos) == len(epos):    
                    hkle = np.stack([hpos, kpos,lpos,epos])  # doesn't need to transpose
                    #print(hkle)
                    [M1, M2, S1, S2, A1, A2, Q] = exp.get_spec_angles(hkle)

                    df_scanpos['h'] = hkle[0]
                    df_scanpos['k'] = hkle[1]
                    df_scanpos['l'] = hkle[2]
                    df_scanpos['e'] = hkle[3]
                    
                    df_scanpos['m1'] = M1
                    df_scanpos['m2'] = M2
                    df_scanpos['s1'] = S1
                    df_scanpos['s2'] = S2
                    df_scanpos['a1'] = A1
                    df_scanpos['a2'] = A2
                    #print(df_scanpos)
                    return df_scanpos
            else:
                print("There are some error!")
                return  None
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing command: {str(e)}")
            return None
