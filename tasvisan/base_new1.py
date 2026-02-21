from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.interpolate import griddata
from .utils.toolfunc import (ni_s2_residual, gaussian_residual, lorentzian_residual, 
                           fit_peak, angle2, SqwDemo, PrefDemo, SelFormFactor, 
                           PrefDemoQscan, strisfloat, AnglesToQhkl)

import inspy as npy
from inspy import TripleAxisSpectr
from inspy.insfit import FitConv


class TasData:
    """
    TAS (Triple Axis Spectrometer) Data analysis class.
    
    This class handles pandas DataFrames containing reduced TAS data with standardized
    column names including: 'qh', 'qk', 'ql', 'en', 'ei', 'ef', 'q', 'm1', 'm2', 
    's1', 's2', 'a1', 'a2', 'sgu', 'sgl', 'detector', 'monitor', 'time', 'sampleT1', 
    'sampleT2', 'magField'
    """
    
    def __init__(self, tasname, expnum, title, sample, user):
        """
        Initialize TasData instance.
        
        Args:
            tasname (str): Name of the TAS instrument
            expnum (str or int): Experiment number
            title (str): Experiment title
            sample (str): Sample description
            user (list or tuple): List of users (non-empty)
        """
        if not all(isinstance(x, str) for x in [tasname, title, sample]): 
            raise ValueError("tasname, title, sample must be strings")
        if not isinstance(user, (list, tuple)) or not user: 
            raise ValueError("user must be a non-empty sequence")
        if not isinstance(expnum, (str, int)):
            raise ValueError("expnum must be a number or string")
            
        # Configuration
        self.tasname = tasname
        self.expnum = str(expnum)
        self.title = title
        self.sample = sample
        self.user = user
        self.exp = None
        self.multichannel = False
        self.multidetector_config = None
        self.multichannel_config = None
        self.total_channel_number = 1
        self.total_det_number = 1 

    def print_exp(self):
        """Print experiment information."""
        print(f'This is a TAS experiment {self.expnum} entitled with {self.title} '
              f'using {self.sample} by {self.user[0]} et al.')

    def tas_datanormalize(self, dflist=None, norm_mon_count=1000000):
        """
        Normalize detector counts by monitor counts.
        
        Args:
            dflist (list): List of pandas DataFrames to normalize
            norm_mon_count (int): Target monitor count for normalization
            
        Returns:
            list: List of normalized DataFrames
        """
        if not isinstance(dflist, list): 
            raise TypeError("dflist must be a list")

        normalized_dflist = []
        for index, df in enumerate(dflist):
            if isinstance(df, pd.DataFrame):
                df = df.copy()  # Avoid modifying original data
                df['detector'] = df['detector'].astype(float)
                
                # Remove rows with zero monitor counts
                df = df[df["monitor"] != 0]
                
                # Check if last monitor count is significantly lower
                if len(df) > 0 and df["monitor"].iloc[-1] < df['monitor'].mean() * 0.90:
                    df = df.iloc[:-1]  # Drop the last row
                    
                # Normalize to target monitor count
                df['detector'] = df['detector'] * norm_mon_count / df['monitor']
                normalized_dflist.append(df)
            else:
                normalized_dflist.append(df)
                
        return normalized_dflist

    def tas_datacombine(self, dflist=None, norm_mon_count=1000000):
        """
        Combine multiple scans into a single DataFrame.
        
        Args:
            dflist (list): List of DataFrames to combine
            norm_mon_count (int): Monitor count for normalization
            
        Returns:
            pd.DataFrame: Combined DataFrame
        """
        if not isinstance(dflist, list): 
            raise TypeError("dflist must be a list")
            
        if len(dflist) == 1:
            print("Only one dataset, no need to combine.")
            return dflist[0]
            
        # Normalize first
        dflist = self.tas_datanormalize(dflist, norm_mon_count)
        comb_df = pd.DataFrame()

        xaxis = ""
        for index, df in enumerate(dflist):
            if isinstance(df, pd.DataFrame):                
                curxaxis = df.attrs.get('scanax1', '')
                
                if index == 0:
                    xaxis = curxaxis
                    comb_df = df.copy()
                    if 'scanax1' not in df.attrs: 
                        raise ValueError("DataFrame missing scanax1 attribute")
                    comb_df.attrs = df.attrs.copy()

                elif curxaxis == xaxis:
                    for jj in df.index:
                        point_exist = False
                        for nn in comb_df.index:
                            # Check if positions are close enough to be considered the same
                            if np.abs(df[xaxis].iloc[jj] - comb_df[xaxis].iloc[nn]) < 0.001:
                                point_exist = True
                                # Average the detector counts
                                comb_df.loc[nn, 'detector'] = (df.loc[jj, 'detector'] + 
                                                              comb_df.loc[nn, 'detector']) / 2
                        
                        if not point_exist:
                            current_row = df.iloc[jj].to_dict()
                            new_index = len(comb_df)
                            for col, val in current_row.items():
                                comb_df.loc[new_index, col] = val
                    
                    # Update scan number attribute
                    comb_df.attrs['scanno'] = (comb_df.attrs.get('scanno', '') + 
                                             "+" + df.attrs.get('scanno', ''))
                else:
                    print(f"The scan #{index} axis is not the same as the first scan!")
        
        if not comb_df.empty:
            comb_df = comb_df.sort_values(by=[xaxis])
            comb_df.reset_index(drop=True, inplace=True)

        return comb_df

    def tas_export_hklw(self, dflist=None, hklw_file=""):
        """
        Export HKLW data to CSV file.
        
        Args:
            dflist (list): List of DataFrames to export
            hklw_file (str): Output filename
            
        Returns:
            pd.DataFrame: Exported HKLW data
        """
        if not isinstance(dflist, list): 
            raise TypeError("dflist must be a list")
    
        hklw = None
        for df in dflist:
            required_cols = ['h', 'k', 'l', 'e', 'detector']
            if not all(col in df.columns for col in required_cols): 
                raise ValueError(f"Missing required columns: {required_cols}")
                
            hklw = df[required_cols].copy()
            scanno = df.attrs.get('scanno', 'unknown')
            
            if hklw_file == "":
                hklw_file = f"{self.tasname}_exp{self.expnum}_HKLW_#{scanno}.csv"
            hklw.to_csv(Path(hklw_file), index=False)

        return hklw

def tas_simpleplot(self, dflist=None, fit=False, initial=None):
    """
    Simple plot and fit of TAS data.
    
    Args:
        dflist (list): List of DataFrames to plot
        fit (bool): Whether to perform fitting
        initial (dict): Initial parameters for fitting
        
    Returns:
        tuple: (params_list, fitdata_list, fig, axs)
    """
    if not dflist or not all(isinstance(df, pd.DataFrame) for df in dflist): 
        raise ValueError("dflist must be a non-empty list of DataFrames")

    params_list = []
    fitdata_list = []

    if len(dflist) > 1:
        fig, axs = plt.subplots(len(dflist), 1, figsize=(6, 4*len(dflist)))
        # Convert axs to list for consistency (in case it's a NumPy array)
        axs = list(axs)  # NEW: Ensure axs is a list
                
        for ii in range(len(dflist)):
            col_x_title = dflist[ii].attrs.get('scanax1', 'x')
            col_y_title = "detector"
            dataX = dflist[ii][col_x_title].values.reshape(-1, 1)
            dataY = dflist[ii][col_y_title].values.reshape(-1, 1)
            
            if fit: 
                cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                axs[ii].plot(dataX.flatten(), dataY.flatten(), 'o', 
                            cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
                params_list.append(cur_fitpar)
                fitdata_list.append(cur_fitdat) 
            else: 
                axs[ii].plot(dataX.flatten(), dataY.flatten(), 'o')
                
            axs[ii].set_xlabel(col_x_title)
            axs[ii].set_ylabel('Intensity [counts]')
            
        return params_list, fitdata_list, fig, axs

    elif len(dflist) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        col_x_title = dflist[0].attrs.get('scanax1', 'x')
        col_y_title = "detector"
        dataX = dflist[0][col_x_title].values.reshape(-1, 1)
        dataY = dflist[0][col_y_title].values.reshape(-1, 1)
            
        if fit: 
            cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
            ax.plot(dataX.flatten(), dataY.flatten(), 'o', 
                    cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
            params_list.append(cur_fitpar)
            fitdata_list.append(cur_fitdat) 
        else:
            ax.plot(dataX.flatten(), dataY.flatten(), 'o') 

        ax.set_xlabel(col_x_title)
        ax.set_ylabel('Intensity [counts]')

        # NEW: Return ax as a single-item list for consistency
        return params_list, fitdata_list, fig, [ax]
    else:
        print("The given dflist is empty!")
        return None

    def tas_combplot(self, dflist=None, fit=False, norm_mon_count=1000000, 
                    overplot=False, offset=1000, initial=None):
        """
        Combined plot of multiple TAS scans.
        
        Args:
            dflist (list): List of DataFrames or lists of DataFrames to plot
            fit (bool): Whether to perform fitting
            norm_mon_count (int): Monitor count for normalization
            overplot (bool): Whether to plot all scans on same axes
            offset (float): Y-offset for overplotted scans
            initial (dict): Initial parameters for fitting
            
        Returns:
            tuple: (params_list, fitdata_list, fig, axs)
        """
        if not dflist or not all(isinstance(df, (pd.DataFrame, list)) for df in dflist): 
            raise ValueError("dflist must be a non-empty list of DataFrames or lists")

        params_list = []
        fitdata_list = []

        if not overplot and len(dflist) > 1:
            fig, axs = plt.subplots(len(dflist), 1, figsize=(6, 4*len(dflist)))
            if len(dflist) == 1:
                axs = [axs]

            for ii, scan in enumerate(dflist):
                if isinstance(scan, list):
                    hklw = self.tas_datacombine(scan, norm_mon_count)
                    x_axis = hklw.attrs.get('scanax1', 'x')
                else:
                    hklw = scan.copy()
                    hklw.attrs = scan.attrs.copy()
                    x_axis = hklw.attrs.get('scanax1', 'x')

                dataX = hklw[x_axis].values.reshape(-1, 1)
                dataY = hklw["detector"].values.reshape(-1, 1)

                if fit:
                    cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                    axs[ii].plot(dataX.flatten(), dataY.flatten(), 'o', 
                               cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
                    params_list.append(cur_fitpar)
                    fitdata_list.append(cur_fitdat) 
                else:
                    axs[ii].plot(dataX.flatten(), dataY.flatten(), 'o')

                axs[ii].set_xlabel(x_axis)
                axs[ii].set_ylabel('Intensity [counts]')
        else:
            fig, axs = plt.subplots(1, 1, figsize=(6, 8))

            for ii, scan in enumerate(dflist):
                if isinstance(scan, list):
                    hklw = self.tas_datacombine(scan, norm_mon_count)
                    x_axis = hklw.attrs.get('scanax1', 'x')
                else:
                    hklw = scan.copy()
                    hklw.attrs = scan.attrs.copy()
                    x_axis = hklw.attrs.get('scanax1', 'x')

                dataX = hklw[x_axis].values.reshape(-1, 1)
                dataY = hklw["detector"].values.reshape(-1, 1)
                
                if ii > 0:
                    dataY = dataY + offset * ii  # Apply Y-offset

                if fit:
                    cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                    axs.plot(dataX.flatten(), dataY.flatten(), 'o', 
                           cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
                    params_list.append(cur_fitpar)
                    fitdata_list.append(cur_fitdat) 
                else:
                    axs.plot(dataX.flatten(), dataY.flatten(), 'o')

            axs.set_xlabel(x_axis)
            axs.set_ylabel('Intensity [counts]')

        return params_list, fitdata_list, fig, axs

    def tas_random_contour(self, df_reduced, xtitle='x', ytitle='y', ztitle='detector', 
                          zminmax=[0, 1000], output_file=None):
        """
        Create a contour plot using matplotlib.
        
        Args:
            df_reduced (pd.DataFrame): DataFrame with reduced data
            xtitle (str): X-axis column name
            ytitle (str): Y-axis column name
            ztitle (str): Z-axis (intensity) column name
            zminmax (list): Min and max values for intensity
            output_file (str): Optional output filename
            
        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        required_cols = [xtitle, ytitle, ztitle]
        if not all(col in df_reduced.columns for col in required_cols): 
            raise ValueError(f"Missing required columns: {required_cols}")

        x = df_reduced[xtitle].values
        y = df_reduced[ytitle].values
        z = df_reduced[ztitle].values
        
        if len(x) < 3: 
            raise ValueError("Insufficient points for contour (need at least 3)")

        # Create regular grid for contour plot
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate z values on the regular grid
        zi = griddata((x, y), z, (xi, yi), method='linear')
        zi = np.clip(zi, zminmax[0], zminmax[1])  # Clip values to defined range
        zi = np.nan_to_num(zi, nan=zminmax[0])  # Replace NaN with min value
        
        # Create the contour plot
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(xi, yi, zi, levels=15, vmin=zminmax[0], vmax=zminmax[1], 
                             cmap='viridis')
        plt.colorbar(contour, label='Counts')
        plt.scatter(x, y, s=1, color='black', alpha=0.5)  # Show original data points
        plt.title('Contour Map of Measurement Data')
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')

        return ax

    def tas_tidy_contour(self, dflist=None, vminmax=None, x_col='h', y_col='e', 
                        scan_range=[0, 1], xlabel='Q [rlu]', ylabel='E [meV]', 
                        title=None, ax=None):
        """
        Combine scans into a contour plot.
        
        Args:
            dflist (list): List of DataFrames
            vminmax (list): Min/max values for contour
            x_col (str): X-axis column name
            y_col (str): Y-axis column name
            scan_range (list): Range for scanning axis
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            title (str): Plot title
            ax (matplotlib.axes.Axes): Existing axes to plot on
            
        Returns:
            tuple: (qq, ee, intensity, ax)
        """
        if not dflist or not all(isinstance(df, pd.DataFrame) for df in dflist): 
            raise ValueError("dflist must be non-empty list of DataFrames")

        if vminmax is None:
            vminmax = [0, 1000]

        contourdata = pd.DataFrame()
        lowrange, uprange = scan_range
        xrange = []
        yrange = []
        xaxis = ""
        delta0 = 0
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        for scanindex, scan in enumerate(dflist):
            required_cols = ['qh', 'qk', 'ql', 'en', 'detector', 'monitor']
            if not all(col in scan.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
                
            hklw = scan[required_cols].copy()
            hklw.attrs = scan.attrs.copy()
            datalines = len(hklw.index)
            xaxis = hklw.attrs.get('scanax1', '')

            if scanindex == 0:
                if datalines < 2:
                    raise ValueError("Insufficient data points in first scan")
                delta0 = abs(hklw[xaxis].iloc[0] - hklw[xaxis].iloc[-1]) / (datalines - 1)
                points = int(round((uprange - lowrange) / delta0) + 1)
                xrange = np.linspace(lowrange, uprange, points)
                contourx = pd.DataFrame(data=xrange, columns=[xaxis])
                contourdata = pd.concat([contourdata, contourx], axis=1)
            else:
                if datalines < 2:
                    continue
                delta = abs(hklw[xaxis].iloc[0] - hklw[xaxis].iloc[-1]) / (datalines - 1)
                if abs(delta - delta0) > 0.002:
                    print(f"Warning: step size of scan no. {scanindex} differs significantly!")

            curCol = hklw[xaxis].to_numpy()
            curColCount = hklw['detector'].to_numpy()

            if len(curCol) == 0:
                continue

            steps = int(round((curCol[-1] - xrange[0]) / delta0))
            
            if (points - steps - 1) < 0:
                print("Error: upper range too small! Please ensure upper range covers whole scan area!")
                continue
            elif (points - steps - 1) > 0:
                temp = np.full(points - steps - 1, 0.1)
                curColCount = np.append(curColCount, temp)

            steps = int(round((hklw[xaxis].iloc[0] - xrange[0]) / delta0))
            
            if steps < 0:
                print("Error: lower range too large! Please ensure lower range covers whole scan area!")
                continue
            elif steps > 0:
                temp = np.full(steps, 0.1)
                curColCount = np.insert(curColCount, 0, temp)

            # Ensure arrays have same length
            if len(curColCount) != len(xrange):
                min_len = min(len(curColCount), len(xrange))
                curColCount = curColCount[:min_len]
                xrange_trimmed = xrange[:min_len]
            else:
                xrange_trimmed = xrange

            tidyarray = np.column_stack((xrange_trimmed, curColCount))
            tempdf = pd.DataFrame(tidyarray, columns=[xaxis, f"count_{scanindex}"])
            contourdata = pd.merge(contourdata, tempdf, on=xaxis, how='outer')
            
            if xaxis == x_col:
                yrange.append(hklw[y_col].iloc[0])
            elif xaxis == y_col:
                yrange.append(hklw[x_col].iloc[0])
            else:
                raise ValueError("ERROR: x_col and y_col are wrong!")

        if len(yrange) < 2:
            raise ValueError("Insufficient scans for contour plot")

        if xaxis == x_col:
            qq = np.linspace(lowrange, uprange, points)
            ypoints = max(2, int(round(abs((yrange[0] - yrange[-1]) / 
                                         (yrange[-1] - yrange[-2])) + 1)))
            ee = np.linspace(yrange[0], yrange[-1], ypoints)
            intensity = contourdata.drop(columns=[xaxis]).to_numpy().T
            cs = ax.contourf(qq, ee, intensity, levels=900, vmin=vminmax[0], vmax=vminmax[1])
        elif xaxis == y_col:
            ee = np.linspace(lowrange, uprange, points)
            ypoints = max(2, int(round(abs((yrange[0] - yrange[-1]) / 
                                         (yrange[0] - yrange[1])) + 1)))
            qq = np.linspace(yrange[0], yrange[-1], ypoints)
            intensity = contourdata.drop(columns=[xaxis]).to_numpy()
            cs = ax.contourf(qq, ee, intensity, levels=900, vmin=vminmax[0], vmax=vminmax[1])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        
        return qq, ee, intensity, ax

    def tas_expconfig(self, ef=5):
        """
        Configure TAS experiment parameters.
        
        Args:
            ef (float): Fixed final energy in meV
            
        Returns:
            TripleAxisSpectr: Configured TAS experiment object
        """
        tas_exp = TripleAxisSpectr(efixed=ef)
        tas_exp.method = 1  # 1 for Popovici, 0 for Cooper-Nathans
        tas_exp.moncor = 1
        tas_exp.efixed = ef
        tas_exp.infin = -1    # const-Ef
        
        tas_exp.mono.dir = 1
        tas_exp.ana.dir = 1
        
        # Monochromator configuration
        tas_exp.mono.tau = 'PG(002)'
        tas_exp.mono.mosaic = 30
        tas_exp.mono.vmosaic = 30
        tas_exp.mono.height = 10
        tas_exp.mono.width = 10
        tas_exp.mono.depth = 0.2
        tas_exp.mono.rh = 100
        tas_exp.mono.rv = 100
        
        # Analyzer configuration
        tas_exp.ana.tau = 'PG(002)'
        tas_exp.ana.mosaic = 30
        tas_exp.ana.vmosaic = 30
        tas_exp.ana.height = 10
        tas_exp.ana.width = 10
        tas_exp.ana.depth = 0.2
        tas_exp.ana.rh = 100
        tas_exp.ana.rv = 100
        
        # Sample information
        tas_exp.sample.a = 4
        tas_exp.sample.b = 5
        tas_exp.sample.c = 6
        tas_exp.sample.alpha = 90
        tas_exp.sample.beta = 90
        tas_exp.sample.gamma = 90
        tas_exp.sample.mosaic = 30
        tas_exp.sample.vmosaic = 30
        tas_exp.sample.u = np.array([1, 0, 0])
        tas_exp.sample.v = np.array([0, 0, 1])
        tas_exp.sample.shape_type = 'rectangular'
        tas_exp.sample.shape = np.diag([0.6, 0.6, 10]) ** 2
        
        # Collimation and arms
        tas_exp.hcol = [60, 60, 60, 60]
        tas_exp.vcol = [500, 500, 500, 500]
        tas_exp.arms = [200, 200, 200, 100, 100]
        tas_exp.orient1 = np.array([1, 0, 0])
        tas_exp.orient2 = np.array([0, 0, 1])
        
        # Guide and detector
        tas_exp.guide.height = 15
        tas_exp.guide.width = 5
        tas_exp.detector.height = 15
        tas_exp.detector.width = 2.5
        
        return tas_exp

    def tas_conv_init(self, hklw=None, exp=None, initial=None, 
                     fixlist=[0,0,0,0,0,0,0,0], magion="Mn2", sqw=SqwDemo, 
                     pref=PrefDemo, smoothfit=True):
        """
        Initialize TAS convolution calculation.
        
        Args:
            hklw (pd.DataFrame): HKLW data
            exp (TripleAxisSpectr): Experiment configuration
            initial (dict): Initial parameters
            fixlist (list): Fixed parameter flags
            magion (str): Magnetic ion type
            sqw (function): Scattering function
            pref (function): Prefactor function
            smoothfit (bool): Whether to use smooth fitting
            
        Returns:
            np.ndarray: Initial simulation result
        """
        if hklw is None:
            print("No data is provided to fit!")
            return
        if exp is None:
            exp = self.tas_expconfig(ef=5)
        
        if initial is not None and not isinstance(initial, dict):
            print("initial is not a dict type")
            return

        H, K, L, W, Iobs = hklw.to_numpy().T  # split into 1D arrays 
        dIobs = np.sqrt(np.maximum(Iobs, 0))  # Avoid sqrt of negative numbers

        ffactor = SelFormFactor(magion)
        if ffactor is None:
            ffactor = SelFormFactor("Mn2")
            print("The given magnetic ion was not found. Instead, Mn2 is used.")

        AA = ffactor["AA"]
        aa = ffactor["aa"]
        BB = ffactor["BB"]
        bb = ffactor["bb"]
        CC = ffactor["CC"]
        cc = ffactor["cc"]
        DD = ffactor["DD"]
        
        initial_new = list(initial.values()) + [AA, aa, BB, bb, CC, cc, DD]
        fixlist_new = fixlist + [0, 0, 0, 0, 0, 0, 0]
     
        sim_initial = exp.ResConv(sqw=SqwDemo, pref=PrefDemo, nargout=2, 
                                 hkle=[H, K, L, W], METHOD='fix', 
                                 ACCURACY=[5, 5], p=initial_new)

        return sim_initial

    def tas_convfit(self, hklw=None, exp=None, initial=None, 
                   fixlist=[0,0,0,0,0,0,0,0], magion="Mn2", sqw=SqwDemo, 
                   pref=PrefDemo, smoothfit=True):
        """
        Perform TAS convolution fitting.
        
        Args:
            hklw (pd.DataFrame): HKLW data
            exp (TripleAxisSpectr): Experiment configuration
            initial (dict): Initial parameters
            fixlist (list): Fixed parameter flags
            magion (str): Magnetic ion type
            sqw (function): Scattering function
            pref (function): Prefactor function
            smoothfit (bool): Whether to use smooth fitting
            
        Returns:
            tuple: (param_err, hkle_fit) - fitted parameters and smooth curve
        """
        if hklw is None:
            print("No data is provided to fit!")
            return
        if exp is None:
            exp = self.tas_expconfig(ef=14.87)  # Fixed typo: was sika_expconfig
        
        if initial is not None and not isinstance(initial, dict):
            print("Initial is not a dict type!")
            return

        H, K, L, W, Iobs = hklw.to_numpy().T  # split into 1D arrays 
        dIobs = np.sqrt(np.maximum(Iobs,  0))  # Avoid sqrt of negative numbers
        
        ffactor = SelFormFactor(magion)
        if ffactor is None:
            ffactor = SelFormFactor("Mn2")
            print("The given magnetic ion was not found. Instead, Mn2 is used.")

        AA = ffactor["AA"]
        aa = ffactor["aa"]
        BB = ffactor["BB"]
        bb = ffactor["bb"]
        CC = ffactor["CC"]
        cc = ffactor["cc"]
        DD = ffactor["DD"]
        
        initial_new = list(initial.values()) + [AA, aa, BB, bb, CC, cc, DD]
        fixlist_new = fixlist + [0, 0, 0, 0, 0, 0, 0]
     
        fitter = FitConv(exp, sqw, pref, [H, K, L, W], Iobs, dIobs)

        result = fitter.fitwithconv(exp, sqw, pref, [H, K, L, W], Iobs, dIobs, 
                                   param=initial_new, paramfixed=fixlist_new)
        final_param, dpa, chisqN, sim, CN, PQ, nit, kvg, details = result

        str_output = "The fitted parameters:\n"
        parlist = list(initial.keys())

        for index, (iname, ipar, ierr) in enumerate(zip(parlist, final_param, dpa)):
            str_output += f"P{index}({iname}):\t {ipar:8f}\t {ierr:8f}\n"

        param_err = pd.DataFrame(final_param[0:8].reshape(1, -1), 
                                columns=['e1', 'e2', 'ratio', 'w1', 'w2', 'int', 'bg', 'T'])
        err = pd.DataFrame(dpa[0:8].reshape(1, -1), 
                          columns=['e1err', 'e2err', 'rtoerr', 'w1err', 'w2err', 
                                  'interr', 'bgerr', 'Terr'])
        param_err = pd.concat([param_err, err], axis=1)
        param_err = param_err[['e1', 'e1err', 'e2', 'e2err', 'ratio', 'rtoerr', 
                              'w1', 'w1err', 'w2', 'w2err', 'int', 'interr', 
                              'bg', 'bgerr', 'T', 'Terr']]
        
        if smoothfit:
            newH = np.linspace(H[0], H[-1], 5*(len(H)-1)+1)
            newK = np.linspace(K[0], K[-1], 5*(len(K)-1)+1)
            newL = np.linspace(L[0], L[-1], 5*(len(L)-1)+1)
            newW = np.linspace(W[0], W[-1], 5*(len(H)-1)+1)
            fittedcurve = exp.ResConv(sqw=sqw, pref=pref, nargout=2, 
                                     hkle=[newH, newK, newL, newW], METHOD='fix', 
                                     ACCURACY=[5, 5], p=final_param)
            hkle_fit = np.vstack((newH, newK, newL, newW, fittedcurve))
        else:
            fittedcurve = exp.ResConv(sqw=sqw, pref=pref, nargout=2, 
                                     hkle=[H, K, L, W], METHOD='fix', 
                                     ACCURACY=[5, 5], p=final_param)
            hkle_fit = np.vstack((H, K, L, W, fittedcurve))

        return param_err, hkle_fit

    def tas_multichannel_config(self, total_channel_number=21, total_det_number=48, 
                               all_detname_list=None, all_det_eff_list=None, 
                               s2_offset_list=None, det_channel_list=None):
        """
        Configure multichannel detector setup.
        
        Args:
            total_channel_number (int): Total number of channels
            total_det_number (int): Total number of detector tubes
            all_detname_list (list): List of detector names
            all_det_eff_list (list): List of detector efficiencies
            s2_offset_list (list): List of S2 angle offsets for each channel
            det_channel_list (list): List of detector-to-channel mappings
        """
        if not all(isinstance(x, (list, tuple)) and x is not None 
                  for x in [all_detname_list, all_det_eff_list, s2_offset_list, det_channel_list]): 
            raise ValueError("All list inputs must be non-None lists")
            
        if (len(all_detname_list) != total_det_number or 
            len(all_det_eff_list) != total_det_number or
            len(s2_offset_list) != total_channel_number or 
            len(det_channel_list) != total_channel_number): 
            raise ValueError("List lengths mismatch with specified counts")

        self.total_channel_number = total_channel_number
        self.total_det_number = total_det_number 

        # Set up detector configuration
        self.multidetector_config = pd.DataFrame({
            "all_detname": all_detname_list, 
            "all_det_eff": all_det_eff_list
        })
        
        # Set up channel configuration
        self.multichannel_config = pd.DataFrame({
            "s2_offset": s2_offset_list, 
            "det_channel": det_channel_list
        })
        
        self.multichannel = True
        print("The multichannel configuration setup is completed.")

    def tas_multichannel_reduction(self, df=None):
        """
        Reduce multichannel detector data.
        
        Args:
            df (pd.DataFrame): Input DataFrame with detector data
            
        Returns:
            list: List of DataFrames for each channel
        """
        if df is None or df.empty: 
            raise ValueError("DataFrame is required and cannot be empty")

        if 'ubmatrix' not in df.attrs: 
            raise ValueError("ubmatrix missing in df.attrs")

        if not self.multichannel:
            print("The multichannel has not been configured.")
            return None

        multichannel_dflist = []
        UBMat = np.matrix(df.attrs['ubmatrix']).reshape(3, 3)

        multi_detname_list = self.multidetector_config['all_detname'].tolist()
        multi_deteff_list = self.multidetector_config['all_det_eff'].tolist()
        multi_channels2_list = self.multichannel_config['s2_offset'].tolist()
        multi_channeldet_list = self.multichannel_config['det_channel'].tolist()

        # Normalize detector counts by efficiency
        df = df.copy()  # Avoid modifying original data
        for tubeindex, tubename in enumerate(multi_detname_list):
            if tubename in df.columns:
                df[tubename] = df[tubename].astype(float)
                if multi_deteff_list[tubeindex] != 0:
                    df[tubename] = df[tubename] / multi_deteff_list[tubeindex]

        # Process each channel
        for channelindex, s2_offset in enumerate(multi_channels2_list):
            df[f"channel_s2_{channelindex}"] = df['s2'] + s2_offset
            df[f"channel_det_{channelindex}"] = 0  # Initialize to zero
            
            # Sum detector counts for this channel
            for tubenum in multi_channeldet_list[channelindex]:
                tube_col = f"tube{int(tubenum)}"
                if tube_col in df.columns:
                    df[f"channel_det_{channelindex}"] += df[tube_col]
            
            # Calculate HKL for each data point
            newhklarray = np.zeros((len(df), 3))
            for i in range(len(df)):
                try:
                    newhkl = AnglesToQhkl(
                        df["m2"].iloc[i],
                        df["s1"].iloc[i],
                        df[f"channel_s2_{channelindex}"].iloc[i],
                        df["a2"].iloc[i],
                        df["sgu"].iloc[i],
                        df["sgl"].iloc[i],
                        UBMat
                    )
                    newhklarray[i] = newhkl[:3].flatten()  #if not flatten, Warning: could not broadcast input array from shape (3,1) into shape (3,)
                except (IndexError, KeyError, ValueError) as e:
                    print(f"Warning: Error calculating HKL for point {i}: {e}")
                    continue

            df[f"channel_h_{channelindex}"] = newhklarray[:, 0]
            df[f"channel_k_{channelindex}"] = newhklarray[:, 1]
            df[f"channel_l_{channelindex}"] = newhklarray[:, 2]

            # Create channel-specific DataFrame
            required_cols = [f"channel_h_{channelindex}", f"channel_k_{channelindex}", 
                           f"channel_l_{channelindex}", 'en', 
                           f"channel_det_{channelindex}", 'monitor']
            
            if all(col in df.columns for col in required_cols):
                df_channel = df[required_cols].copy()
                df_channel.columns = ['qh', 'qk', 'ql', 'en', 'detector', 'monitor']
                df_channel.attrs = df.attrs.copy()
                df_channel.attrs["channel"] = f"channel#{channelindex}"
                multichannel_dflist.append(df_channel)

        return multichannel_dflist

    def tas_multichannel_plot3d(self, dflist=None, xcol='qh', ycol='ql', zcol="en", 
                               perc_low=5, perc_high=95, xlim=None, ylim=None, zlim=None,
                               xlabel="QH [r.l.u]", ylabel="QL [r.l.u]", zlabel="E [meV]"):
        """
        Create 3D scatter plot of multichannel data.
        
        Args:
            dflist (list): List of DataFrames
            xcol (str): X-axis column name
            ycol (str): Y-axis column name
            zcol (str): Z-axis column name
            perc_low (float): Lower percentile for color scaling
            perc_high (float): Upper percentile for color scaling
            xlim (list): X-axis limits
            ylim (list): Y-axis limits
            zlim (list): Z-axis limits
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            zlabel (str): Z-axis label
            
        Returns:
            tuple: (X, Y, Z, A, ax) - coordinates, intensities, and axes
        """
        if not dflist:
            raise ValueError("dflist cannot be empty")
            
        required_cols = [xcol, ycol, zcol, 'detector']
        for df in dflist:
            if not all(col in df.columns for col in required_cols): 
                raise ValueError(f"Missing required columns: {required_cols}")

        X = np.array([])
        Y = np.array([])
        Z = np.array([]) 
        A = np.array([])

        for df in dflist:
            X = np.append(X, df[xcol].to_numpy())
            Y = np.append(Y, df[ycol].to_numpy())
            Z = np.append(Z, df[zcol].to_numpy())
            A = np.append(A, df['detector'].to_numpy())

        # Handle zero or negative values for log scale
        logA = np.log10(np.maximum(A, 1e-5))
        vminA, vmaxA = np.percentile(logA, [perc_low, perc_high])
        norm = Normalize(vmin=vminA, vmax=vmaxA, clip=True)
        scaled = norm(logA)
        colors = cm.viridis(scaled)
        colors[:, -1] = scaled  # Set alpha channel

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X, Y, Z, c=colors)
        
        # Set axis limits if provided
        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])
        if zlim is not None and len(zlim) == 2:
            ax.set_zlim(zlim[0], zlim[1])

        # Labels and formatting
        ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
        ax.set_zlabel(zlabel, fontsize=14, labelpad=10)
        
        # Tick formatting
        ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        
        plt.tight_layout()
        plt.show()
        
        return X, Y, Z, A, ax