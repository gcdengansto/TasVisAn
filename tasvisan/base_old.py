
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.interpolate import griddata
from .utils.toolfunc import ni_s2_residual, gaussian_residual, lorentzian_residual, fit_peak, angle2, SqwDemo, PrefDemo, SelFormFactor, PrefDemoQscan, strisfloat
from .utils.toolfunc import AnglesToQhkl

import inspy as npy
from inspy import TripleAxisSpectr
from inspy.insfit import FitConv




class TasData:
    def __init__(self, tasname, expnum, title, sample, user):
        """TasData does not deal with original data.
           It will use pandas dataframe for all the data.
           The pandas dataframe should be the results after the data reduction from indivisual TAS instrument, such as Taipan and Sika.
           The standard column names: 'qh', 'qk', 'ql', 'en', 'ei', 'ef', 'q', 'm1', 'm2', 's1', 's2', 'a1', 'a2', 'sgu', 'sgl', 'det', 'mon', 'time', 'sampleT1', 'sampleT2', 'magField', 
        """
        if not all(isinstance(x, str) for x in [tasname, title, sample]): 
            raise ValueError("tasname, title, sample must be strings")
        if not isinstance(user, (list, tuple)) or not user: 
            raise ValueError("user must be a non-empty sequence")
        if not isinstance(expnum, (str, int)):
            raise ValueError("expnum must be a number or string")
        # configuration
        self.tasname= tasname
        self.expnum = str(expnum)
        self.title  = title
        self.sample = sample
        self.user   = user
        self.exp    = None
        self.multichannel          = False
        self.multidetector_config  = None
        self.multichannel_config   = None
        self.total_channel_number  = 1
        self.total_det_number      = 1 

    def print_exp(self):
        print('This is a Tas experiment '+ self.expnum + ' entitled with ' + self.title + ' using ' + self.sample + ' by ' + self.user[0] + ' et al.')
        return
    
    def tas_datanormalize(self, dflist = None, norm_mon_count= 1000000):

        if not isinstance(dflist, list): 
            raise TypeError("dflist must be a list")

        for index, df in enumerate(dflist):
            if isinstance(df, pd.DataFrame):
                df['detector'] = df['detector'].astype(float)                    #detector count can be decimal after normalization
                df = df[df["monitor"]!=0]                                          # remove the row with monitor zero
                if df["monitor"].iloc[-1] < df['monitor'].mean()*0.90:             # if the last monitor is less than 90% of the avrage count
                    df = df.iloc[:,-1]  #drop the last row
                df['detector'] = df['detector']*norm_mon_count/df['monitor']       # normalized to the given monitor number
                
        return dflist


    def tas_datacombine(self, dflist = None, norm_mon_count= 1000000):
        """
        scanlist is a int list for the scan number to be combined together
        [101, 103, 106] is acceptable
        [[101, 103], 105] is not acceptable
        """
        if not isinstance(dflist, list): 
            raise TypeError("dflist must be a list")
            
        elif len(dflist) == 1:
            print("only one data set, no need to combine.")
            return dflist[0]
        dflist = self.tas_datanormalize(dflist,norm_mon_count)   #normalize first
        comb_df  =  pd.DataFrame([])

        xaxis    =   ""
        for index, df in enumerate(dflist):
            if isinstance(df, pd.DataFrame):                
                curxaxis  = df.attrs['scanax1'] 

                if index == 0:                                            # the first data
                    xaxis    = df.attrs['scanax1']                      # take the first data axis as the axis current axis is a list of string
                    comb_df  = df.copy()
                    if 'scanax1' not in df.attrs: 
                        raise ValueError("DataFrame missing scanax1 attribute")
                    comb_df.attrs=df.attrs.copy()

                elif curxaxis == xaxis:

                    for jj in df.index:
                        point_exist = False
                        for nn in comb_df.index:
                            if np.abs(df[xaxis].iloc[jj] - comb_df[xaxis].iloc[nn]) < 0.001: # if they are the same position
                                point_exist = True                        # the same position in two different scans
                                comb_df.loc[nn,'detector']    =  (df.loc[jj,'detector'] + comb_df.loc[nn,'detector'])/2 # all scans has been normalized to the same monitor
                        
                        if point_exist == False:
                            current_row  = df.iloc[jj].tolist()           # change the record into a list
                            comb_df.loc[len(comb_df)] = current_row       # add the new row to the end.
                    comb_df.attrs['scanno']=comb_df.attrs['scanno'] + "+" + df.attrs['scanno']
                else:
                    print("The scan #{} axis is not the same as the first scan!".format(index))
        
        if not comb_df.empty:
            comb_df = comb_df.sort_values(by=[xaxis])
            comb_df.reset_index(drop=True, inplace=True)

        return comb_df # this is a single dataframe



    def tas_export_hklw(self, dflist = None, hklw_file=""):
        #this is advanced export function, which can combine data and then export
        # dflist is simple list. if you want to combine, it is better to combine when you create dflist.
        if not isinstance(dflist, list): 
            raise TypeError("dflist must be a list")
    
        
        for df in dflist:
            if not all(col in df.columns for col in ['h', 'k', 'l', 'e', 'detector']): 
                raise ValueError("Missing required columns")
            hklw = df[['h','k','l','e','detector']]
            scanno=df.attrs['scanno']
            
            if hklw_file == "" :
                hklw_file = self.tasname+"_exp" + str(self.expnum) + "_HKLW_#" + scanno + ".csv"
            hklw.to_csv(Path(hklw_file), index=False)

        return hklw


    def tas_simpleplot(self, dflist = None, fit = False, initial = None ):
        """
        Simply plot and fit the data
        """
        if not dflist or not all(isinstance(df, pd.DataFrame) for df in dflist): 
            raise ValueError("dflist must be a non-empty list of DataFrames")

        params_list     = list()   
        fitdata_list    = list()   

        

        if len(dflist) > 1:
            fig, axs = plt.subplots(len(dflist), 1, figsize=(6, 4*len(dflist)))
            for ii in range(len(dflist)):
        
                col_x_title = dflist[ii].attrs['scanax1']  
                col_y_title = "detector"
                dataX = dflist[ii][:][[col_x_title]].to_numpy()
                dataY = dflist[ii][:][[col_y_title]].to_numpy()
                
                if fit: 
                    cur_fitpar, cur_fitdat= fit_peak(dataX, dataY, func='G', initial=initial)
                    axs[ii].plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')   # the scan is normalized only in the plot.
                    params_list.append(cur_fitpar)
                    fitdata_list.append(cur_fitdat) 

                else: 
                    axs[ii].plot(dataX, dataY, 'o')
                    
                axs[ii].set_xlabel(col_x_title)
                axs[ii].set_ylabel('Intensity [counts]')
            return params_list, fitdata_list, fig, axs

        elif len(dflist) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4*len(dflist)))
            col_x_title = dflist[0].attrs['scanax1']
            col_y_title = "detector"
            dataX = dflist[0][:][[col_x_title]].to_numpy()
            dataY = dflist[0][:][[col_y_title]].to_numpy()
                
            if fit : 
                cur_fitpar, cur_fitdat= fit_peak(dataX, dataY, func='G', initial=initial)
                ax.plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')   # the scan is normalized only in the plot.
                params_list.append(cur_fitpar)
                fitdata_list.append(cur_fitdat) 
            else:
                ax.plot(dataX, dataY, 'o') 

            ax.set_xlabel(col_x_title)
            ax.set_ylabel('Intensity [counts]')

            return params_list, fitdata_list, fig, ax
        else:
            print("The given scanlist is emtpy!")
            return None
        

    def tas_combplot(self, dflist=None, fit=False, norm_mon_count=1000000, overplot=False, offset=1000, initial=None ):
        #new simple plot
        if not dflist or not all(isinstance(df, pd.DataFrame) for df in dflist): 
            raise ValueError("dflist must be a non-empty list of DataFrames")

        params_list     = list()   
        fitdata_list    = list()   

        if overplot == False and len(dflist) > 1:
            fig, axs = plt.subplots(len(dflist), 1, figsize=(6, 4*len(dflist)))

            for ii, scan in enumerate (dflist):
                if isinstance(scan, list):
                    hklw   = self.tas_datacombine( scan, norm_mon_count)  # set to the first scan monitor

                    x_axis = hklw.attrs['scanax1']
                else:
                    hklw   = scan.copy()
                    hklw.attrs = hklw.attrs.copy()
                    x_axis = hklw.attrs['scanax1']

                if ii == 0 :

                    #monitor0 = scan['monitor'][0]                                          #the first point of monitor
                    dataX = hklw[[x_axis]].to_numpy()
                    dataY = hklw[["detector"]].to_numpy()

                    if fit:
                        cur_fitpar, cur_fitdat= fit_peak(dataX, dataY, func='G', initial=initial)
                        axs[ii].plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')   
                        params_list.append(cur_fitpar)
                        fitdata_list.append(cur_fitdat) 
                    else:
                        axs[ii].plot(dataX,dataY, 'o')  #dataY*monitor0/hklw['beamMon'][0]
                else:    
                    dataX = hklw[[x_axis]].to_numpy()
                    dataY = hklw[["detector"]].to_numpy()

                    if fit:
                        cur_fitpar, cur_fitdat= fit_peak(dataX, dataY, func='G', initial=initial)
                        axs[ii].plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')   
                        params_list.append(cur_fitpar)
                        fitdata_list.append(cur_fitdat) 
                    else:
                        axs[ii].plot(dataX, dataY, 'o')

                axs[ii].set_xlabel(x_axis)
                axs[ii].set_ylabel('Intensity [counts]')
        else:

            fig, axs = plt.subplots(1, 1, figsize=(6, 8))

            for ii, scan in enumerate (dflist):
                if isinstance(scan, list):
                    hklw   = self.tas_datacombine( scan, norm_mon_count)  # set to the first scan monitor
                    x_axis = hklw.attrs['scanax1']
                else:
                    hklw   = scan.copy()
                    hklw.attrs=hklw.attrs.copy()
                    x_axis = hklw.attrs['scanax1']

                if ii == 0 :
                    dataX = hklw[[x_axis]].to_numpy()
                    dataY = hklw[["detector"]].to_numpy()

                    if fit:
                        cur_fitpar, cur_fitdat= fit_peak(dataX, dataY, func='G', initial=initial)
                        axs.plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')   
                        params_list.append(cur_fitpar)
                        fitdata_list.append(cur_fitdat) 
                    else:
                        axs.plot(dataX, dataY, 'o')
                else:    
                    dataX = hklw[[x_axis]].to_numpy()
                    dataY = hklw[["detector"]].to_numpy()
                    dataY = dataY + offset*ii                           #offset along Y
                    
                    if fit:
                        cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                        axs.plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')   
                        params_list.append(cur_fitpar)
                        fitdata_list.append(cur_fitdat) 
                    else:
                        axs.plot(dataX, dataY, 'o')

                axs.set_xlabel(x_axis)
                axs.set_ylabel('Intensity [counts]')

        return params_list, fitdata_list, fig, axs  

    # Create a contour plot using matplotlib
    def tas_random_contour(self, df_reduced, xtitle='x', ytitle= 'y', ztitle= 'detector', zminmax=[0, 1000], output_file=None):
        #df_reduced  = df with column [h, k l, e, detector monitor] reduced by row. all the data are packed in.
        #df_reduced = sika.sika_reduction_by_row()   kkl could be not the same, just list all collected value in the contour range
        # Extract data

        if not all(col in df_reduced.columns for col in [xtitle, ytitle, ztitle]): 
            raise ValueError("Missing required columns")

        x = df_reduced[xtitle].values
        y = df_reduced[ytitle].values
        z = df_reduced[ztitle].values
        if len(x) < 3: 
            raise ValueError("Insufficient points for contour")
        # Create a regular grid for contour plot
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate z values on the regular grid
        #zi = griddata((x, y), z, (xi, yi), method='cubic')
        # Interpolate z values on the regular grid
        zi = griddata((x, y), z, (xi, yi), method='linear')  # Change to 'linear' to avoid extrapolation issues
        zi = np.clip(zi, zminmax[0], zminmax[1])  # Clip values to defined range
        zi = np.nan_to_num(zi, nan=zminmax[0])  # Replace NaN with min value
        
        
        # Create the contour plot
        fig, ax = plt.subplots(figsize=(10,8))
        contour = ax.contourf(xi, yi, zi, levels=15, vmin=zminmax[0], vmax=zminmax[1], cmap='viridis')
        plt.colorbar(contour, label='Counts')
        plt.scatter(x, y, s=1, color='black', alpha=0.5)  # Show original data points
        plt.title('Contour Map of Measurement Data')
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.grid(True, linestyle='--', alpha=0.7)
        if output_file:
            plt.savefig('contour_plot_matplotlib.png', dpi=300, bbox_inches='tight')

        return ax


    def tas_tidy_contour(self, dflist=None,  vminmax=None, x_col='h', y_col='e', scan_range=[0, 1], xlabel='Q [rlu]',ylabel='E [meV]',title=None, ax=None):
        # to combine scans into a contour :
        if not dflist or not all(isinstance(df, pd.DataFrame) for df in dflist): 
            raise ValueError("dflist must be non-empty list of DataFrames")

        if vminmax is None:
            vminmax = [0, 1000]

        contourdata =  pd.DataFrame([])

        lowrange = scan_range[0]
        uprange  = scan_range[1]
        xrange   = []
        yrange   = []
        xaxis    = ""
        delta0   = 0
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        for scanindex, scan in enumerate (dflist):
                
            hklw       = scan[['qh','qk','ql','en','detector','monitor']]                #only choose the most important data
            hklw.attrs = scan.attrs.copy()
            datalines  = len(hklw.index)
            xaxis = hklw.attrs['scanax1']   
            #print(f"the scanning axis is {xaxis}") 

            if(scanindex==0):
                #print(hklw)
                delta0   = np.abs(hklw[xaxis][0]-hklw[xaxis][datalines-1])/(datalines-1)  #determine the step size
                points   = int(np.around((uprange-lowrange)/delta0)+1)                     #determine how many steps in whole range
                xrange   = np.linspace(lowrange, uprange, points)                           #generate an array with step size of delta
                #print(xrange)
                #monitor0 = hklw['monitor'][0]                                               #the first point of monitor
                contourx = pd.DataFrame(data=xrange)                                      #create a DataFrame with first column xrange
                contourx.columns=[xaxis]                                                  #give a name to this column as the scan axis
                contourdata = pd.merge(contourdata, contourx, how='outer', left_index=True, right_index=True)
            else:
                delta    = np.abs(hklw[xaxis][0]-hklw[xaxis][datalines-1])/(datalines-1)
                if np.abs(delta-delta0) > 0.002:
                    print("Error: the step size of the scan no. {} is not the same!".format(scanindex))
                    return 

            curCol       = hklw[xaxis].to_numpy()                                                #get the range of the real data from file

            curColCount  = hklw['detector'].to_numpy()         #*monitor0/hklw['monitor'][0]                  # not necessary, normalization has done.
            #curColCount  = curColCount.astype("float64")      # this has been done earlier in normalization

            steps        = int(np.around((curCol[datalines-1]-xrange[0])/delta0))                         #calculate the difference between max range and the last data point
            
            if (points-steps-1) < 0:
                print("Error: the up range is too small! please make sure the up range covering the whole scan area!")
            elif (points-steps-1) == 0:
                print("The up range is just on the edge.")
            else:
                temp = np.zeros(points-steps-1)
                temp = temp + 0.1                                       #generate a zero array of the size of the end
                #np.insert(curCol, datalines-1, temp)
                curColCount=np.insert(curColCount, datalines, temp)     #insert the zero array to the end of the real data

            steps = int(np.around((hklw[xaxis][0]-xrange[0])/delta0))      #calculate the missing points at the beginning of real data

            if steps < 0:
                print("Error: the low range is too big! please make sure the low range covering the whole scan area!")
            elif steps == 0:
                print("The low range is just on the edge.")
            else:
                temp = np.zeros(steps)                                    #generate a zero arry of size of the beginning
                temp = temp + 0.1
                curColCount = np.insert(curColCount, 0, temp)             #insert it

            tidyarray = np.vstack((xrange,curColCount)).T                 # now it has the same size as the low and high range
            tempdf    =  pd.DataFrame(tidyarray, columns=[xaxis,"count_{}".format(scanindex)])
            #insert the data frame into the 
            contourdata = pd.merge(contourdata, tempdf, on=xaxis, how='outer')  
            
            if   xaxis == x_col:
                yrange.append(hklw[y_col][0])                          # generate an array along energy   
            elif xaxis == y_col:                         
                yrange.append(hklw[x_col][0])                          # generate an array along energy   
            else:
                print("ERROR:The x_col and y_col are wrong!")
                
        if   xaxis == x_col:
            qq        = np.linspace(lowrange,uprange,points)
            print(qq)
            ypoints   = int(np.around(np.abs((yrange[0]-yrange[-1])/(yrange[-2]-yrange[-1]))+1))  #uneven steps in the scans cause problems
            ee        = np.linspace(yrange[0], yrange[-1],ypoints)

            intensity = contourdata.drop(columns=[xaxis]).to_numpy().T
            cs = ax.contourf(qq, ee, intensity, levels=900,   vmin=vminmax[0], vmax=vminmax[1]) #
        elif  xaxis == y_col:
            ee        = np.linspace(lowrange,uprange,points)
            ypoints   = int(np.around(np.abs((yrange[0]-yrange[-1])/(yrange[0]-yrange[1]))+1))
            qq        = np.linspace(yrange[0], yrange[-1],ypoints)
            

            intensity = contourdata.drop(columns=[xaxis]).to_numpy()
            cs        = ax.contourf(qq, ee, intensity, levels=900,  vmin=vminmax[0], vmax=vminmax[1]) #

        #contourdata.to_csv("C:\\Data\\NeutronData\\debug_contour.csv")   
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        return qq, ee, intensity, ax

    def tas_expconfig(self, ef=5):
        
        tas_exp = TripleAxisSpectr(efixed = ef)
        tas_exp.method   =   1  # 1 for Popovici, 0 for Cooper-Nathans
        tas_exp.moncor   =   1
        tas_exp.efixed   =   ef
        tas_exp.infin    =  -1    #const-Ef
        
        tas_exp.mono.dir =  1
        tas_exp.ana.dir  =  1
        
        tas_exp.mono.tau      = 'PG(002)'
        tas_exp.mono.mosaic   = 30
        tas_exp.mono.vmosaic  = 30
        tas_exp.mono.height   = 10            #no need for /sqrt(12)
        tas_exp.mono.width    = 10
        tas_exp.mono.depth    = 0.2
        tas_exp.mono.rh       = 100
        tas_exp.mono.rv       = 100
        
        tas_exp.ana.tau     = 'PG(002)'
        tas_exp.ana.mosaic  = 30
        tas_exp.ana.vmosaic = 30
        tas_exp.ana.height  = 10
        tas_exp.ana.width   = 10
        tas_exp.ana.depth   = 0.2
        tas_exp.ana.rh      = 100
        tas_exp.ana.rv      = 100
        
        #Put the sample information below
        tas_exp.sample.a    = 4
        tas_exp.sample.b    = 5
        tas_exp.sample.c    = 6
        tas_exp.sample.alpha = 90
        tas_exp.sample.beta  = 90
        tas_exp.sample.gamma = 90
        tas_exp.sample.mosaic  = 30
        tas_exp.sample.vmosaic = 30
        tas_exp.sample.u       = np.array([1, 0, 0])
        tas_exp.sample.v       = np.array([0, 0, 1])
        tas_exp.sample.shape_type =   'rectangular'
        tas_exp.sample.shape      = np.diag([0.6, 0.6, 10])**2
        tas_exp.hcol = [60, 60, 60, 60]
        tas_exp.vcol = [500, 500, 500, 500]
        tas_exp.arms = [200, 200, 200, 100, 100]             #need to be updated sika [230, 203, 179, 39.5, 106] 
        tas_exp.orient1 = np.array([1, 0, 0])
        tas_exp.orient2 = np.array([0, 0, 1])                #need to be updated
        
        tas_exp.guide.height = 15
        tas_exp.guide.width  =  5
        
        tas_exp.detector.height  =   15
        tas_exp.detector.width   =  2.5
        #print(tas_exp)
        return tas_exp


    def tas_conv_init(self, hklw=None, exp = None,  initial=None, fixlist=[0,0,0,0,0,0,0,0], magion= "Mn2", sqw=SqwDemo, pref=PrefDemo, smoothfit=True):
        #numpy array is a dataframe and should be change into a numpy array
        if hklw is None:
            print("No data is provided to fit!")
            return
        if exp is None:
            exp = self.tas_expconfig(ef = 5)
        
        if initial is not None:
            if not isinstance(initial, dict):
                print("initial is not a dict type")

        [H, K, L, W, Iobs] =  hklw.to_numpy().T  #split into 1D arrays 
        dIobs   = np.sqrt(Iobs)

        ffactor=SelFormFactor(magion)
        if ffactor is None:
            ffactor=SelFormFactor("Mn2")
            print("The given magnetic ion was not found. Instead, Mn2 is used.")

        AA=ffactor["AA"]
        aa=ffactor["aa"]
        BB=ffactor["BB"]
        bb=ffactor["bb"]
        CC=ffactor["CC"]
        cc=ffactor["cc"]
        DD=ffactor["DD"]
        initial_new = list(initial.values()) + [AA, aa, BB, bb, CC, cc, DD]
        fixlist_new = fixlist + [ 0,  0,  0,  0,  0,  0,  0]
     
        sim_inital = exp.ResConv(sqw=SqwDemo, pref=PrefDemo, nargout=2, hkle=[H,K,L,W], METHOD='fix', ACCURACY=[5,5], p=initial_new)

        return sim_inital

    def tas_convfit(self, hklw=None, exp = None,  initial=None, fixlist=[0,0,0,0,0,0,0,0], magion= "Mn2", sqw=SqwDemo, pref=PrefDemo, smoothfit=True):
        #numpy array is a dataframe and should be change into a numpy array
        if hklw is None:
            print("No data is provided to fit!")
            return
        if exp is None:
            exp = self.sika_expconfig(ef = 14.87)
        
        if initial is not None:
            if not isinstance(initial, dict):
                print("Initial is not a dict type!")

        [H, K, L, W, Iobs] =  hklw.to_numpy().T  #split into 1D arrays 
        dIobs   = np.sqrt(Iobs)

        ffactor=SelFormFactor(magion)
        if ffactor is None:
            ffactor=SelFormFactor("Mn2")
            print("The given magnetic ion was not found. Instead, Mn2 is used.")

        AA=ffactor["AA"]
        aa=ffactor["aa"]
        BB=ffactor["BB"]
        bb=ffactor["bb"]
        CC=ffactor["CC"]
        cc=ffactor["cc"]
        DD=ffactor["DD"]
        initial_new = list(initial.values()) + [AA, aa, BB, bb, CC, cc, DD]
        fixlist_new = fixlist + [ 0,  0,  0,  0,  0,  0,  0]
     
        fitter     =    FitConv(exp,  sqw,  pref,  [H,K,L,W],  Iobs,  dIobs)

        [final_param, dpa, chisqN, sim, CN, PQ, nit, kvg, details] = fitter.fitwithconv(exp, sqw, pref, [H,K,L,W], Iobs, dIobs, param=initial_new, paramfixed=fixlist_new)

        str_output = "The fitted parameters:\n"
        parlist=list()
        for key in initial.keys():
            parlist = parlist + [key]

        for index, (iname, ipar, ierr) in enumerate(zip(parlist,final_param,dpa)):
            str_output=str_output+"P{0}({1}):\t {2:8f}\t {3:8f}\n".format(index, iname, ipar, ierr)

        param_err=pd.DataFrame(final_param[0:8].reshape(1,-1), columns=['e1','e2','ratio', 'w1', 'w2', 'int', 'bg', 'T'])
        err=pd.DataFrame(dpa[0:8].reshape(1,-1), columns=['e1err','e2err','rtoerr', 'w1err', 'w2err', 'interr', 'bgerr', 'Terr'])
        param_err=pd.concat([param_err, err],axis=1)
        param_err=param_err[['e1','e1err','e2','e2err','ratio','rtoerr', 'w1','w1err', 'w2','w2err', 'int','interr', 'bg','bgerr',  'T', 'Terr']]
        #print(param_err)
        
        if smoothfit:
            newH=np.linspace(H[0],H[-1], 5*(len(H)-1)+1)
            newK=np.linspace(K[0],K[-1], 5*(len(K)-1)+1)
            newL=np.linspace(L[0],L[-1], 5*(len(L)-1)+1)
            newW=np.linspace(W[0],W[-1], 5*(len(H)-1)+1)
            fittedcurve= exp.ResConv(sqw=sqw, pref=pref, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=[5,5], p=final_param)
            hkle_fit = np.vstack((newH,newK,newL,newW,fittedcurve))
        else:
            hkle_fit = np.vstack((H,K,L,W,fittedcurve))

        return param_err, hkle_fit


    def tas_multichannel_config(self, total_channel_number=21, total_det_number=48, all_detname_list=None, all_det_eff_list=None, s2_offset_list=None, det_channel_list=None):
        #s2_offset_list is the each channel angle offset from the central blade
        #det_channel_list is the col_names of each channel detector, users much to combine the detector counts if one channel correponding to several tubes
        #det_eff_list, this should be 
        if not all(isinstance(x, (list, tuple)) for x in [all_detname_list, all_det_eff_list, s2_offset_list, det_channel_list]): 
            raise ValueError("All list inputs must be non-None lists")
        if len(all_detname_list) != total_det_number or len(s2_offset_list) != total_channel_number: 
            raise ValueError("List lengths mismatch with counts")

        self.total_channel_number     = total_channel_number
        self.total_det_number         = total_det_number 

        if all (x is not None for x in (all_detname_list, all_det_eff_list, s2_offset_list, det_channel_list)):
            #all the detector should do the normalization first
            self.multidetector_config = pd.DataFrame({"all_detname": all_detname_list, "all_det_eff": all_det_eff_list})
            #with the corrected counts on each detector, find the counts corresponding to each channel
            #channel number is less than detector number
            self.multichannel_config = pd.DataFrame({"s2_offset": s2_offset_list, "det_channel": det_channel_list})
            self.multichannel = True
            print("The multichannel configuration setup is completed.")
        else:
            self.multichannel = False
            print("The multichannel configuration could not be completed.")
        
        return

    def tas_multichannel_reduction(self, df=None):
        #All the multi-ana config is set to property by the function above. 

        if not df: 
            raise ValueError("Missing required columns")

        if 'ubmatrix' not in df.attrs: 
            raise ValueError("ubmatrix missing in df.attrs")

        if self.multichannel == True:
            multichannel_dflist=[]
            UBMat=np.matrix(df.attrs['ubmatrix']).reshape(3,3)
            
            #print(UBMat)

            multi_detname_list=self.multidetector_config['all_detname'].tolist()
            multi_deteff_list =self.multidetector_config['all_det_eff'].tolist()
            multi_channels2_list = self.multichannel_config['s2_offset'].tolist()
            multi_channeldet_list = self.multichannel_config['det_channel'].tolist()

            for tubeindex, tubename in enumerate(multi_detname_list):
                df[tubename]=df[tubename].astype(float)
                df[tubename]=df[tubename]/multi_deteff_list[tubeindex]

            for channelindex, s2_offset in enumerate(multi_channels2_list):
                df["channel_s2_"+str(channelindex)]   = df['s2'] + s2_offset
                df["channel_det_"+str(channelindex)]  = 0            # set the starting value as zero
                for tubenum in multi_channeldet_list[channelindex]: # the channelindex-th tube has a list
                    df["channel_det_"+str(channelindex)] +=  df["tube"+str(int(tubenum))]  # add all the tube-results in the current list multi_channeldet_list[channelindex]
                
                newhklarray=np.zeros((len(df),3))
                for i in range(len(df)):
                    newhkl=AnglesToQhkl(df["m2"].iloc[i],
                                        df["s1"].iloc[i],
                                        df["channel_s2_"+str(channelindex)].iloc[i],
                                        df["a2"].iloc[i],
                                        df["sgu"].iloc[i],
                                        df["sgl"].iloc[i],
                                        UBMat)

                    newhklarray[i][0] = newhkl[0]
                    newhklarray[i][1] = newhkl[1]
                    newhklarray[i][2] = newhkl[2]
                    #print(type(hkelist))
                df["channel_h_"+str(channelindex)] = newhklarray[:,0]
                df["channel_k_"+str(channelindex)] = newhklarray[:,1]
                df["channel_l_"+str(channelindex)] = newhklarray[:,2]
                #print(df.columns)
                df_channel=df[["channel_h_"+str(channelindex),"channel_k_"+str(channelindex),"channel_l_"+str(channelindex),
                                'en', "channel_det_"+str(channelindex), 'monitor']]
                df_channel.columns = ['qh', 'qk', 'ql', 'en', 'detector', 'monitor']
                df_channel.attrs["channel"] = "channel#" + str(channelindex)
                multichannel_dflist.append(df_channel)
            return multichannel_dflist
        else:
            print("The multichannel has not been configured.")
            return
        
    def tas_multichannel_plot3d(self, dflist=None, xcol='qh', ycol='ql', zcol="en", perc_low=5, perc_high=95, xlim=None, ylim=None, zlim=None,xlabel="QH [r.l.u]", ylabel="QL [r.l.u]",zlabel="E [meV]"):
        """ put all the data points in dflist into X, Y, Z, Intensity, then plot in the 3d space with scatter. 
        """
        if not all(col in df.columns for col in [xcol, ycol, zcol, 'detector']): 
            raise ValueError("Missing required columns")

        X=np.array([])
        Y=np.array([])
        Z=np.array([]) 
        A=np.array([])

        for df in dflist:
            X=np.append(X, df[xcol].to_numpy())
            Y=np.append(Y, df[ycol].to_numpy())
            Z=np.append(Z,df[zcol].to_numpy())
            A=np.append(A,df['detector'].to_numpy())   #peak intensity,

        logA = np.log10(np.maximum(A, 1e-5))  # change into logscale and avoid log(0)
        vminA, vmaxA  = np.percentile(logA, [perc_low, perc_high])
        norm          = Normalize(vmin=vminA, vmax =vmaxA, clip=True)
        scaled        = norm(logA)
        colors        = cm.viridis(scaled)

        colors[:,-1]  = scaled

        fig           =  plt.figure()
        ax  = fig.add_subplot(111, projection ='3d')
        sc  = ax.scatter(X, Y, Z, color=colors)
        if xlim is not None and len(xlim) == 2:
            ax.set_xlim( xlim[0],  xlim[1])
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim( ylim[0],  ylim[1])
        if zlim is not None and len(zlim) == 2:
            ax.set_zlim( zlim[0],  zlim[1])

        # Labels
        ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
        ax.set_zlabel(zlabel, fontsize=14, labelpad=10)
        
        # Tick labels
        ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        
        plt.show()
        return X, Y, Z, A, ax