

    def tas_tidy_contour_old(self, dflist=None, x_col='qh', y_col='en', xlabel='Q [rlu]',ylabel='E [meV]', title=None,  vminmax=None, ax=None):
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

        xrange   = []
        yrange   = []
        Xaxis    = ""
        delta0   = 0

        lowrange = min(df[y_col].min() for df in dflist) #use the min as the low range
        uprange  = max(df[y_col].max() for df in dflist)  #use the max as the up range


        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        for scanindex, scan in enumerate (dflist):
                
            hklw       = scan[['qh','qk','ql','en','detector','monitor']]                #only choose the most important data
            hklw.attrs = scan.attrs.copy()
            datalines  = len(hklw.index)
            Xaxis = hklw.attrs['scanax1']   
            #print(f"the scanning axis is {Xaxis}") 

            if(scanindex==0):
                #print(hklw)
                delta0   = np.abs(hklw[Xaxis][0]-hklw[Xaxis][datalines-1])/(datalines-1)  #determine the step size
                points   = int(np.around((uprange-lowrange)/delta0)+1)                     #determine how many steps in whole range
                xrange   = np.linspace(lowrange, uprange, points)                           #generate an array with step size of delta
                #print(xrange)
                #monitor0 = hklw['monitor'][0]                                               #the first point of monitor
                contourx = pd.DataFrame(data=xrange)                                      #create a DataFrame with first column xrange
                contourx.columns=[Xaxis]                                                  #give a name to this column as the scan axis
                contourdata = pd.merge(contourdata, contourx, how='outer', left_index=True, right_index=True)
            else:
                delta    = np.abs(hklw[Xaxis][0]-hklw[Xaxis][datalines-1])/(datalines-1)
                if np.abs(delta-delta0) > 0.002:
                    print("Error: the step size of the scan no. {} is not the same!".format(scanindex))
                    return 

            curCol       = hklw[Xaxis].to_numpy()                                                #get the range of the real data from file

            curColCount  = hklw['detector'].to_numpy()         #*monitor0/hklw['monitor'][0]                  # not necessary, normalization has done.
            #curColCount  = curColCount.astype("float64")      # this has been done earlier in normalization

            steps        = int(np.around((curCol[datalines-1]-xrange[0])/delta0))                         #calculate the difference between max range and the last data point
            
            if (points-steps-1) < 0:
                print("Error: the up range is too small! please make sure the up range covering the whole scan area!")
            #elif (points-steps-1) == 0:
                #print("The up range is just on the edge.")
            else:
                temp = np.zeros(points-steps-1)
                temp = temp + 0.1                                       #generate a zero array of the size of the end
                #np.insert(curCol, datalines-1, temp)
                curColCount=np.insert(curColCount, datalines, temp)     #insert the zero array to the end of the real data

            steps = int(np.around((hklw[Xaxis][0]-xrange[0])/delta0))      #calculate the missing points at the beginning of real data

            if steps < 0:
                print("Error: the low range is too big! please make sure the low range covering the whole scan area!")
            #elif steps == 0:
                #print("The low range is just on the edge.")
            else:
                temp = np.zeros(steps)                                    #generate a zero arry of size of the beginning
                temp = temp + 0.1
                curColCount = np.insert(curColCount, 0, temp)             #insert it

            tidyarray = np.vstack((xrange,curColCount)).T                 # now it has the same size as the low and high range
            tempdf    =  pd.DataFrame(tidyarray, columns=[Xaxis,"count_{}".format(scanindex)])
            #insert the data frame into the 
            contourdata = pd.merge(contourdata, tempdf, on=Xaxis, how='outer')  
            
            if   Xaxis == x_col:
                yrange.append(hklw[y_col][0])                          # generate an array along energy   
            elif Xaxis == y_col:                         
                yrange.append(hklw[x_col][0])                          # generate an array along energy  y_col 
            else:
                print("ERROR:The x_col and y_col are wrong!")
                
        if   Xaxis == x_col:
            xx        = np.linspace(lowrange,uprange,points)
            ypoints   = len(yrange)  #old way has a bug using np.around()
            yy        = np.linspace(yrange[0], yrange[-1],ypoints)

            intensity = contourdata.drop(columns=[Xaxis]).to_numpy().T
            cs = ax.contourf(xx, yy, intensity, levels=900,   vmin=vminmax[0], vmax=vminmax[1]) #

        elif  Xaxis == y_col:
            yy        = np.linspace(lowrange,uprange,points)
            ypoints   = len(yrange) #old way has a bug using np.around()
            xx        = np.linspace(yrange[0], yrange[-1],ypoints)
            
            intensity = contourdata.drop(columns=[Xaxis]).to_numpy()
            cs        = ax.contourf(xx, yy, intensity, levels=900,  vmin=vminmax[0], vmax=vminmax[1]) #


        ax.set_xlabel(xlabel,fontsize=18, labelpad=10)
        ax.set_ylabel(ylabel,fontsize=18, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6, pad=5)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        return xx, yy, intensity, ax


    def tas_tidy_contour2(self, dflist=None,  x_col='qh', y_col='en', xlabel='Q [rlu]', ylabel='E [meV]', title='Contour Map',vminmax=None, ax=None):
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
            raise ValueError("dflist must be a non-empty list of DataFrames")

        if vminmax is None:
            vminmax = [0, 1000]

        contourdata = pd.DataFrame([])

        lowrange = min(df[y_col].min() for df in dflist) #use the min as the low range
        uprange  = max(df[y_col].max() for df in dflist)  #use the max as the up range
        xrange = []
        yrange = []
        Xaxis = ""
        delta0 = 0
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        for scanindex, scan in enumerate(dflist):
            hklw = scan[['qh', 'qk', 'ql', 'en', 'detector', 'monitor']].copy()
            hklw.attrs = scan.attrs.copy()
            datalines = len(hklw.index)
            Xaxis = hklw.attrs.get('scanax1')
            if Xaxis is None:
                raise TypeError("The scan axis is unknown from the data header.")
            
            if not Xaxis:
                raise ValueError(f"Scan #{scanindex} missing scanax1 attribute")

            if scanindex == 0:
                delta0 = np.abs(hklw[Xaxis][0] - hklw[Xaxis][datalines-1]) / (datalines-1) if datalines > 1 else 0
                points = int(np.around((uprange - lowrange) / delta0) + 1) if delta0 != 0 else 1
                xrange = np.linspace(lowrange, uprange, points)
                contourx = pd.DataFrame(data=xrange, columns=[Xaxis])
                contourdata = contourx.copy()

            curColCount = hklw['detector'].to_numpy()
            steps = int(np.around((uprange - hklw[Xaxis][datalines-1]) / delta0)) if delta0 != 0 else 0

            if steps < 0:
                print(f"Warning: Scan #{scanindex} ends beyond scan_range upper limit")
            elif steps > 0:
                temp = np.zeros(steps) + 0.1
                curColCount = np.append(curColCount, temp)

            steps = int(np.around((hklw[Xaxis][0] - lowrange) / delta0)) if delta0 != 0 else 0

            if steps < 0:
                print(f"Warning: Scan #{scanindex} starts below scan_range lower limit")
            elif steps > 0:
                temp = np.zeros(steps) + 0.1
                curColCount = np.insert(curColCount, 0, temp)

            tidyarray = np.vstack((xrange, curColCount)).T
            tempdf = pd.DataFrame(tidyarray, columns=[Xaxis, f"count_{scanindex}"])
            contourdata = pd.merge(contourdata, tempdf, on=Xaxis, how='outer')
            
            if Xaxis == x_col:
                yrange.append(hklw[y_col][0])
            elif Xaxis == y_col:
                yrange.append(hklw[x_col][0])
            else:
                print(f"ERROR: Scan #{scanindex} axis {Xaxis} does not match x_col ({x_col}) or y_col ({y_col})")
                continue

        if not yrange:
            raise ValueError("No valid yrange values collected")

        if len(yrange) < 2:
            raise ValueError("Insufficient unique y values for contour")

        if Xaxis == x_col:
            xx = np.linspace(lowrange, uprange, points)
            ypoints = len(yrange)
            yy = np.linspace(min(yrange), max(yrange), ypoints)
            intensity = contourdata.drop(columns=[Xaxis]).to_numpy().T
            cs = ax.contourf(xx, yy, intensity, levels=900, vmin=vminmax[0], vmax=vminmax[1])
        elif Xaxis == y_col:
            yy = np.linspace(lowrange, uprange, points)
            ypoints = len(yrange)
            xx = np.linspace(yrange[0], yrange[-1],ypoints)   #np.linspace(min(yrange), max(yrange), ypoints)
            intensity = contourdata.drop(columns=[Xaxis]).to_numpy()
            cs = ax.contourf(xx, yy, intensity, levels=900, vmin=vminmax[0], vmax=vminmax[1])
        else:
            raise ValueError("Xaxis does not match x_col or y_col")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        
        return xx, yy, intensity, ax