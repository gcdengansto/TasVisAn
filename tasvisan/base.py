from pathlib import Path
import pandas as pd
import numpy as np
import math

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
from inspy.insfit import FitConv, UltraFastFitConv


class TasData:
    """
    TAS (Triple Axis Spectrometer) Data analysis class.
    
    This class handles pandas DataFrames containing reduced TAS data with standardized
    column names including: 'qh', 'qk', 'ql', 'en', 'ei', 'ef', 'q', 'm1', 'm2', 
    's1', 's2', 'a1', 'a2', 'sgu', 'sgl', 'det', 'mon', 'time', 'sampleT1', 
    'sampleT2', 'magField'
    """
    # Constants
    PG002_D_SPACING = 3.355  # Angstroms
    #DEFAULT_NORM_COUNT = 1000000
    VARIANCE_THRESHOLD = 0.0006
    
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

    def tas_datanormalize2(self, dflist=None, norm_mon_count=-999999):
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
        if len(dflist[0]) == 0: 
            raise TypeError("the first dataframe in dflist is empty.")

        normalized_dflist = []

        for index, df in enumerate(dflist):
            if isinstance(df, pd.DataFrame):
                df = df.copy()  # Avoid modifying original data
                df['detector'] = df['detector'].astype(float)
                
                # Use the monitor counts from first dataframe if not provided
                if index == 0:  
                    if len(df) > 1:
                        # Use the single row's monitor value if only 1 row exists
                        avg_monitor =  df["monitor"][:-1].mean() 
                    else :  #len(df) == 1;   len(df) ==0 will not reach here. it stops at the beginning.
                        avg_monitor = df["monitor"].iloc[0] 


                
                if norm_mon_count < 0 : # if norm_mon_count was not given, use the monitor value of the first df
                    norm_mon_count = avg_monitor
                
                # Remove rows where monitor count is too low (incomplete counting)
                df = df[df["monitor"] >= 0.9 * avg_monitor]
                
                # Skip empty dataframes after filtering
                if len(df) == 0:
                    normalized_dflist.append(df)  # Add empty df
                    print(f"Warning: DataFrame at index {index} became empty after filtering")
                    continue
                
                # Normalize to target monitor count
                df['detector'] = df['detector'] * norm_mon_count / df['monitor']
                normalized_dflist.append(df)
            else:
                print(f"Error: the {index} element in dflist is not a DataFrame.")
                
        return normalized_dflist
        
    def tas_datanormalize(self, dflist=None, norm_mon_count=-999999):
        """
        Normalize detector counts by monitor counts.
        
        Args:
            dflist (list): List of pandas DataFrames to normalize
            norm_mon_count (int): Target monitor count for normalization
                If negative, uses the average monitor count from first dataframe
            
        Returns:
            list: List of normalized DataFrames
        """
        # Validate dflist is a non-empty list
        if not isinstance(dflist, list) or len(dflist) == 0: 
            raise TypeError("dflist must be a non-empty list.")
        
        # Validate first element is a non-empty DataFrame
        if not isinstance(dflist[0], pd.DataFrame):
            raise TypeError("the first element in dflist must be a DataFrame.")
        if len(dflist[0]) == 0: 
            raise ValueError("the first DataFrame in dflist is empty.")
        
        normalized_dflist = []
        
        # Determine the normalization reference value from first dataframe if not provided
        if norm_mon_count < 0:
            first_df = dflist[0]
            if len(first_df) >= 2:
                norm_mon_count = first_df["monitor"][:-1].mean()
            else:  # len(first_df) == 1, since we already checked it's not empty
                norm_mon_count = first_df["monitor"].iloc[0]

        # Now process all dataframes
        for index, df in enumerate(dflist):
            if isinstance(df, pd.DataFrame):
                df = df.copy()  # Avoid modifying original data
                df['detector'] = df['detector'].astype(float)
                
                # Calculate average monitor for THIS dataframe (excluding last row for filtering)
                if len(df) >= 2:
                    avg_monitor_this_df = df["monitor"][:-1].mean()
                elif len(df) == 1:
                    avg_monitor_this_df = df["monitor"].iloc[0]
                else:
                    # Empty dataframe - keep it to maintain indices
                    print(f"Warning: DataFrame at index {index} is empty")
                    normalized_dflist.append(df)
                    continue
                
                # Remove rows where monitor count is too low (incomplete counting)
                # Filter based on THIS dataframe's average, not norm_mon_count
                df = df[df["monitor"] >= 0.9 * avg_monitor_this_df]
                
                # If dataframe becomes empty after filtering, keep it to maintain indices
                if len(df) == 0:
                    print(f"Warning: DataFrame at index {index} became empty after filtering")
                    normalized_dflist.append(df)
                    continue
                
                # Normalize to target monitor count (uses norm_mon_count, not avg_monitor_this_df)
                #print("XXXXXXXXXXXXXXXXX")
                #print(len(df['detector']))
                #print(len(df['monitor']))
                #print(norm_mon_count)
                #print("XXXXXXXXXXXXXXXXX")
                df['detector'] = df['detector'] * norm_mon_count / df['monitor']
                normalized_dflist.append(df)
            else:
                print(f"Error: the {index} element in dflist is not a DataFrame.")
                
        return normalized_dflist

    def tas_datacombine(self, dflist=None, norm_mon_count=-999999):
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

        Xaxis = ""
        for index, df in enumerate(dflist):
            if isinstance(df, pd.DataFrame):                
                curXaxis = df.attrs.get('scanax1')
                if curXaxis is None:
                    raise TypeError("The scan axis is unknown from the data header.")
                
                if index == 0:
                    Xaxis = curXaxis
                    comb_df = df.copy()
                    if 'scanax1' not in df.attrs: 
                        raise ValueError("DataFrame missing scanax1 attribute")
                    comb_df.attrs = df.attrs.copy()

                elif curXaxis == Xaxis:
                    for jj in df.index:
                        point_exist = False
                        for nn in comb_df.index:
                            # Check if positions are close enough to be considered the same
                            if np.abs(df[Xaxis].iloc[jj] - comb_df[Xaxis].iloc[nn]) < 0.001:
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
                    comb_df.attrs['scanno'] = (comb_df.attrs.get('scanno') + 
                                             "+" + df.attrs.get('scanno'))
                else:
                    print(f"The scan #{index} axis is not the same as the first scan!")
        
        if not comb_df.empty:
            comb_df = comb_df.sort_values(by=[Xaxis])
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
            required_cols = ['qh', 'qk', 'ql', 'en', 'detector']
            if not all(col in df.columns for col in required_cols): 
                raise ValueError(f"Missing required columns: {required_cols}")
                
            hklw = df[required_cols].copy()
            scanno = df.attrs.get('scanno')
            
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
            tuple: (fitted_df, params_df, fig, axs)
        """
        if not dflist or not all(isinstance(df, pd.DataFrame) for df in dflist): 
            raise ValueError("dflist must be a non-empty list of DataFrames")
        
        params_df  =  pd.DataFrame(columns = ['A', 'A_err', 'w','w_err', 'x0', 'x0_err', 'bg', 'bg_err'])
        fitted_df  =  pd.DataFrame([])
        Xaxis      = "" 
        Yaxis      = "detector"

        if len(dflist) > 1:
            fig, axs = plt.subplots(len(dflist), 1, figsize=(6, 4*len(dflist)))
            # Convert axs to list for consistency
            axs = list(axs)    # NEW: Ensure axs is a list
                
            for ii in range(len(dflist)):
                if dflist[ii].empty:
                    print(f"The {ii}th DataFrame in the list is empty! No plot.")
                else:
                    Xaxis = dflist[ii].attrs.get('scanax1')
                    if Xaxis is None:
                        raise TypeError("The scan axis is unknown from the data header.")

                    dataX = dflist[ii][Xaxis].to_numpy()
                    dataY = dflist[ii][Yaxis].to_numpy()
                    
                    if fit and len(dflist[ii]) > 4:
                        cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                        axs[ii].plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')

                        params_df   =  pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)
                        fitted_df   =  pd.merge(fitted_df, cur_fitdat, how='outer', left_index=True, right_index=True, suffixes=("", "_"+str(ii)))
                    else:
                        axs[ii].plot(dataX, dataY, 'o')
                        if fit and len(dflist[ii]) <= 4:
                            print("Not enough data points for fitting!")
                    axs[ii].set_xlabel(Xaxis)
                    axs[ii].set_ylabel('Intensity [counts]')

            return params_df, fitted_df, fig, axs

        elif len(dflist) == 1:
            if dflist[0].empty:
                print(f"The only one DataFrame in the list is empty! No plot.")
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                Xaxis = dflist[0].attrs.get('scanax1')
                if Xaxis is None:
                    raise TypeError("The scan axis is unknown from the data header.")

                dataX = dflist[0][Xaxis].to_numpy()
                dataY = dflist[0][Yaxis].to_numpy()
        
                if fit and len(dflist[0]) > 4:
                    cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                    ax.plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')

                    params_df   =  pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)
                    fitted_df   =  pd.merge(fitted_df, cur_fitdat, how='outer', left_index=True, right_index=True, suffixes=("", "_"))
                else:
                    ax.plot(dataX, dataY, 'o')
                    if fit and len(dflist[0]) <= 4 :
                        print("Not enough data points for fitting!")

                ax.set_xlabel(Xaxis)
                ax.set_ylabel('Intensity [counts]')

            return params_df, fitted_df, fig, [ax]  # NEW: Return ax as a single-item list


    def tas_combplot_old(self, dflist=None, fit=False, norm_mon_count = -999999, overplot=False, offset=1000, initial=None):
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
            tuple: (fitted_df, params_df, fig, axs) where axs is always a list of Axes
        """
        if not dflist or not all(isinstance(df, (pd.DataFrame, list)) for df in dflist): 
            raise ValueError("dflist must be a non-empty list of DataFrames or lists")

        params_df  =  pd.DataFrame(columns = ['A', 'A_err', 'w','w_err', 'x0', 'x0_err', 'bg', 'bg_err'])
        fitted_df  =  pd.DataFrame([])
        Xaxis      =  ""
        Yaxis      = "detector"

        if not overplot and len(dflist) > 1:
            fig, axs = plt.subplots(len(dflist), 1, figsize=(6, 4*len(dflist)))
            # Convert axs to list for consistency (in case it's a NumPy array)
            axs = list(axs)  # NEW: Ensure axs is a list

            for ii, scan in enumerate(dflist):
                if isinstance(scan, list):
                    hklw = self.tas_datacombine(scan, norm_mon_count)
                    Xaxis = hklw.attrs.get('scanax1')
                    if Xaxis is None:
                        raise TypeError("The scan axis is unknown from the data header.")
                else:
                    hklw = scan.copy()
                    hklw.attrs = scan.attrs.copy()
                    Xaxis = hklw.attrs.get('scanax1')
                    if Xaxis is None:
                        raise TypeError("The scan axis is unknown from the data header.")
                    
                dataX = hklw[Xaxis].to_numpy()
                dataY = hklw[Yaxis].to_numpy()

                if ii > 0:
                    dataY = dataY + offset * ii  # Apply Y-offset

                if fit and len(hklw) > 4:
                    cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                    axs[ii].plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
                    params_df   =  pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)
                    fitted_df   =  pd.merge(fitted_df, cur_fitdat, how='outer', left_index=True, right_index=True, suffixes=("", "_"+str(ii)))
                else:
                    axs[ii].plot(dataX, dataY, 'o')
                    if fit and len(hklw) < 4 :
                        print("Not enough data points for fitting!")

                axs[ii].set_xlabel(Xaxis)
                axs[ii].set_ylabel('Intensity [counts]')
            
            return params_df, fitted_df, fig, axs  # NEW: Return axs as list
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))

            #for ii, scan in enumerate(dflist):
            #dflist[0] could be a Dataframe or a df list
            if isinstance(dflist[0], list):  # if dflist[0] is a DF list, we combine the element
                hklw = self.tas_datacombine(dflist[0], norm_mon_count)
                Xaxis = hklw.attrs.get('scanax1')
                if Xaxis is None:
                    raise TypeError("The scan axis is unknown from the data header.")
            else:
                hklw = dflist[0].copy()
                hklw.attrs = dflist[0].attrs.copy()
                Xaxis = hklw.attrs.get('scanax1')
                if Xaxis is None:
                    raise TypeError("The scan axis is unknown from the data header.")
                
            dataX = hklw[Xaxis].to_numpy()
            dataY = hklw[Yaxis].to_numpy()
            
            if fit and len(hklw) > 4:
                cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                ax.plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
 
                params_df   =  pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)
                fitted_df   =  pd.merge(fitted_df, cur_fitdat, how='outer', left_index=True, right_index=True, suffixes=("", "_"))
            else:
                ax.plot(dataX, dataY, 'o')
                if fit and len(hklw) < 4 :
                        print("Not enough data points for fitting!")

            ax.set_xlabel(Xaxis)
            ax.set_ylabel('Intensity [counts]')

            return params_df, fitted_df, fig, [ax]  # NEW: Return ax as a single-item list
        
    def tas_combplot(self, dflist=None, fit=False, norm_mon_count = -999999, overplot=False, offset=1000, initial=None):
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
            tuple: (fitted_df, params_df, fig, axs) where axs is always a list of Axes
        """
        if not dflist or not all(isinstance(df, (pd.DataFrame, list)) for df in dflist): 
            raise ValueError("dflist must be a non-empty list of DataFrames or lists")

        params_df  =  pd.DataFrame(columns = ['A', 'A_err', 'w','w_err', 'x0', 'x0_err', 'bg', 'bg_err'])
        fitted_df  =  pd.DataFrame([])
        Xaxis      =  ""
        Yaxis      = "detector"

        dflist = self.tas_datanormalize(dflist, norm_mon_count)

        if not overplot and len(dflist) > 1:
            fig, axs = plt.subplots(len(dflist), 1, figsize=(6, 4*len(dflist)))
            # Convert axs to list for consistency (in case it's a NumPy array)
            axs = list(axs)

            for ii, scan in enumerate(dflist):
                if isinstance(scan, list):
                    hklw = self.tas_datacombine(scan, norm_mon_count)
                    Xaxis = hklw.attrs.get('scanax1')
                    if Xaxis is None:
                        raise TypeError("The scan axis is unknown from the data header.")
                else:
                    hklw = scan.copy()
                    hklw.attrs = scan.attrs.copy()
                    Xaxis = hklw.attrs.get('scanax1')
                    if Xaxis is None:
                        raise TypeError("The scan axis is unknown from the data header.")
                    
                dataX = hklw[Xaxis].to_numpy()
                dataY = hklw[Yaxis].to_numpy()

                if ii > 0:
                    dataY = dataY + offset * ii  # Apply Y-offset

                if fit and len(hklw) > 4:
                    cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                    axs[ii].plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
                    params_df   =  pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)
                    fitted_df   =  pd.merge(fitted_df, cur_fitdat, how='outer', left_index=True, right_index=True, suffixes=("", "_"+str(ii)))
                else:
                    axs[ii].plot(dataX, dataY, 'o')
                    if fit and len(hklw) < 4 :
                        print("Not enough data points for fitting!")

                axs[ii].set_xlabel(Xaxis)
                axs[ii].set_ylabel('Intensity [counts]')
            
            return params_df, fitted_df, fig, axs
        else:
            # overplot=True OR single scan case
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))
            
            # Loop through all scans in dflist
            for ii, scan in enumerate(dflist):
                if isinstance(scan, list):
                    hklw = self.tas_datacombine(scan, norm_mon_count)
                    Xaxis = hklw.attrs.get('scanax1')
                    if Xaxis is None:
                        raise TypeError("The scan axis is unknown from the data header.")
                else:
                    hklw = scan.copy()
                    hklw.attrs = scan.attrs.copy()
                    Xaxis = hklw.attrs.get('scanax1')
                    if Xaxis is None:
                        raise TypeError("The scan axis is unknown from the data header.")
                
                dataX = hklw[Xaxis].to_numpy()
                dataY = hklw[Yaxis].to_numpy()
                
                # Apply offset when overplotting multiple scans
                if overplot and ii > 0:
                    dataY = dataY + offset * ii
                
                if fit and len(hklw) > 4:
                    cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G', initial=initial)
                    ax.plot(dataX, dataY, 'o', cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
    
                    params_df   =  pd.concat([params_df, cur_fitpar], axis=0, ignore_index=True)
                    fitted_df   =  pd.merge(fitted_df, cur_fitdat, how='outer', left_index=True, right_index=True, suffixes=("", "_"+str(ii)))
                else:
                    ax.plot(dataX, dataY, 'o')
                    if fit and len(hklw) < 4:
                        print("Not enough data points for fitting!")

            ax.set_xlabel(Xaxis)
            ax.set_ylabel('Intensity [counts]')

            return params_df, fitted_df, fig, [ax]

    def tas_random_contour(self, df_reduced, x_col="qh", y_col="en", xtitle='qh [rlu]', ytitle='en [meV]',title='Contour Map of Measurement Data', vminmax=[0, 1000], output_file=None):
        """
        Create a contour plot using matplotlib.
        
        Args:
            df_reduced (pd.DataFrame): DataFrame with reduced data
            x_col (str): X-axis column name
            y_col (str): Y-axis column name
            xtitle (str): X-axis label
            ytitle (str): Y-axis label
            vminmax (list): Min and max values for intensity
            output_file (str): Optional output filename
            
        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        required_cols = [x_col, y_col, 'detector']
        if not all(col in df_reduced.columns for col in required_cols): 
            raise ValueError(f"Missing required columns: {required_cols}")

        x = df_reduced[x_col].values
        y = df_reduced[y_col].values
        z = df_reduced['detector'].values
        
        if len(x) < 3: 
            raise ValueError("Insufficient points for contour (need at least 3)")

        # Create regular grid for contour plot
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate z values on the regular grid
        zi = griddata((x, y), z, (xi, yi), method='linear')
        zi = np.clip(zi, vminmax[0], vminmax[1])  # Clip values to defined range
        zi = np.nan_to_num(zi, nan=vminmax[0])  # Replace NaN with min value
        
        # Create the contour plot
        fig, ax = plt.subplots(figsize=(8, 6))
        cs = ax.contourf(xi, yi, zi, levels=100, vmin=vminmax[0], vmax=vminmax[1], 
                             cmap='viridis')
        plt.colorbar(cs, label='Counts')
        plt.scatter(x, y, s=1, color='black', alpha=0.5)  # Show original data points
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')

        return ax



    def tas_tidy_contour(self, dflist=None, x_col='qh', y_col='en', xlabel='Q [rlu]',ylabel='E [meV]', title=None,  vminmax=None, bRot=False, ax=None):
        """
        Combine scans into a contour plot.
        
        Args:
            dflist (list): List of DataFrames
            vminmax (list): Min/max values for contour
            x_col (str): X-axis column name
            y_col (str): Y-axis column name
            scan_range (list): Range for x-axis
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            title (str): Plot title
            ax (matplotlib.axes.Axes): Optional axes to plot on
            
        Returns:
            tuple: (xx, yy, intensity, ax) - grid coordinates, intensity, and axes
        """
        if not dflist or not all(isinstance(df, pd.DataFrame) for df in dflist): 
            raise ValueError("dflist must be non-empty list of DataFrames")

        if vminmax is None:
            vminmax = [0, 1000]

        contourdata =  pd.DataFrame([])

        scan_range  = []
        pos_range   = []
        Xaxis       = ""
        delta0      = 0


        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        for scan in dflist:
            Xaxis = scan.attrs['scanax1']   
            if Xaxis != y_col:
                raise ValueError("The scan axis of all data must be the same as y_col.")
        
        low_lim = min(df[y_col].min() for df in dflist) #use the min as the low range
        up_lim  = max(df[y_col].max() for df in dflist)  #use the max as the up range
        #print(f"lowlim:{low_lim}; uplim:{up_lim}")

        for scanindex, scan in enumerate (dflist):
                
            hklw       = scan[['qh','qk','ql','en','detector','monitor']]                #only choose the most important data
            hklw.attrs = scan.attrs.copy()
            datalines  = len(hklw.index)
            Xaxis = hklw.attrs['scanax1']   
            #print(f"the scanning axis is {Xaxis}") 

            if(scanindex==0):
                #get stepsize from the first data
                delta0      = np.abs(hklw[Xaxis][0]-hklw[Xaxis][datalines-1])/(datalines-1)    #determine the step size
                points      = int(np.around((up_lim-low_lim)/delta0)) +1                       #determine how many steps in whole range
                scan_range  = np.linspace(low_lim, up_lim, points)                             #generate an array with step size of delta
                contourx    = pd.DataFrame(data=scan_range)                                    #create a DataFrame with first column scan_range
                contourx.columns=[Xaxis]                                                       #give a name to this column as the scan axis
                contourdata = pd.merge(contourdata, contourx, how='outer', left_index=True, right_index=True)
            else:
                delta    = np.abs(hklw[Xaxis][0]-hklw[Xaxis][datalines-1])/(datalines-1)
                if np.abs(delta-delta0) > 0.002:
                    print("Error: the step size of the scan no. {} is not the same!".format(scanindex))
                    return 

            curCol       = hklw[Xaxis].to_numpy()                                                # get the range of the real data from file
            curColCount  = hklw['detector'].to_numpy()                                           # not necessary, normalization has done.
            pos_range.append(hklw[x_col][0])                                                     # generate an array along energy  y_col 

            step_deci, step_int = math.modf((curCol[datalines-1]-scan_range[0])/delta0)

            if np.abs(step_deci) > 0.45 and np.abs(step_deci) < 0.55:
                print(f"Warning: Scan #{hklw.attrs.get('scanno')}: half step detected!")
                curCol = curCol + delta0/2                                        # shift half step for all
                 
            steps        = int(np.around((curCol[datalines-1]-scan_range[0])/delta0))                #calculate the difference between max range and the last data point
            if (points-steps-1) > 0:
                temp = np.zeros(points-steps-1) + 0.1                        #generate a zero array of the size of the end
                curColCount = np.insert(curColCount, datalines, temp)        #insert the zero array to the end of the real data

            steps        = int(np.around((curCol[0]-scan_range[0])/delta0)) #calculate the missing points at the beginning of real data

            if steps > 0:
                temp = np.zeros(steps)  + 0.1                                  #generate a zero arry of size of the beginning
                curColCount = np.insert(curColCount, 0, temp)                  #insert it

            tidyarray = np.vstack((scan_range, curColCount)).T                      # now it has the same size as the low and high range
            tempdf    =  pd.DataFrame(tidyarray, columns=[Xaxis,"count_{}".format(scanindex)])
            
            contourdata = pd.merge(contourdata, tempdf, on=Xaxis, how='outer')  


        Yscan       = np.linspace(low_lim, up_lim, points)
        Xpos        = np.linspace(pos_range[0], pos_range[-1], len(pos_range) )
        
        intensity   = contourdata.drop(columns=[Xaxis]).to_numpy()

        if  bRot == False:
            cs  = ax.contourf(Xpos, Yscan, intensity, levels=900,  vmin=vminmax[0], vmax=vminmax[1]) #
        else:
            cs  = ax.contourf(Yscan, Xpos, intensity.T, levels=900,  vmin=vminmax[0], vmax=vminmax[1]) #


        ax.set_xlabel(xlabel,fontsize=18, labelpad=10)
        ax.set_ylabel(ylabel,fontsize=18, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6, pad=5)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        return Xpos, Yscan, intensity, ax



    
    def tas_bosecorrect(self,dflist=None, sample_T=1.5):
        """
        Apply Bose-Einstein population correction to inelastic neutron scattering data.
        
        Args:
            path (str): Directory path containing scan files
            dflist: List of scan numbers to process
            sample_T (float): Sample temperature in Kelvin
            
        Returns:
            list: List of corrected DataFrames
        """
        KB_MEV_K = 0.08617  # Boltzmann constant in meV/K (more accurate value)
        
        if sample_T <= 0:
            print('The sample temperature is incorrect, it should be given in Kelvin')
            return
        
        newdflist = []
        
        for index, df in enumerate(dflist):
            # Keep only positive energy transfers
            newdf = df.loc[df['en'] > 0].copy()  # Use .copy() to avoid SettingWithCopyWarning 
            newdf.reset_index(drop=True, inplace=True)  
            
            if df.attrs['scanax1'] == 'en':
                deltae = newdf['en'].to_numpy()
                counts = newdf['detector'].to_numpy()
                # Apply Bose population correction
                # Standard formula: I_corrected = I_measured / [1 - exp(-E/kT)]
                bose_factor = 1 - np.exp(-deltae / (sample_T * KB_MEV_K))     #from claude 
                # Avoid division by zero for very small energies
                bose_factor = np.where(np.abs(bose_factor) < 1e-10, 1e-10, bose_factor)
                counts_corrected = counts / bose_factor
                
                newdf['detector'] = counts_corrected
            else:
                print(f"Warning: Scan {index} is not an energy scan, skipping correction.")
            
            newdflist.append(newdf)
        
        return newdflist

    def tas_expconfig(self, ef=5):
        """
        Configure TAS experiment parameters.
        
        Args:
            ef (float): Fixed energy (default 5 meV)
            
        Returns:
            TripleAxisSpectr: Configured experiment object
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

    def tas_conv_init(self, hklw=None, exp=None, initial=None, fixlist=[0,0,0,0,0,0,0,0], magion="none", sqw=SqwDemo, pref=PrefDemo):
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

    def tas_convfit(self, hklw=None, exp=None, initial=None, fixlist=[0,0,0,0,0,0,0,0], magion="none", sqw=SqwDemo, pref=PrefDemo):
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
                    newhklarray[i] = newhkl[:3].flatten()
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