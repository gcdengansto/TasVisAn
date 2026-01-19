import sys
import time
import os, csv
from pathlib import Path
import numpy as np
import pandas as pd   

from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtCore import QTimer, QRegExp, QAbstractTableModel, Qt
from qtpy.QtWidgets import (QApplication, QMainWindow, QSizePolicy, QTextEdit, QVBoxLayout, QMessageBox, QFileDialog,)
from qtpy.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import  NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from lmfit import Parameters, minimize, fit_report

from ..tas.taipan import Taipan
from ..tas.sika import Sika
from ..utils.toolfunc import fit_peak, three_peaks_on_slope_residual, strisfloat, strisint



def parse_number_string(s):
    elements = s.split(",")  # Split by comma to get individual elements
    result = []
    
    for element in elements:
        sub_elements = list(map(int, element.split("+")))  # Split by "+" and convert to integers
        if len(sub_elements) == 1:
            result.append(sub_elements[0])  # If there's only one number, add it as an integer
        else:
            result.append(sub_elements)  # Otherwise, add as a list
    
    return result


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class TASCanvas(FigureCanvas):
    def __init__(self, parent=None, width=451, height=281, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', edgecolor='k')
        self.fig.subplots_adjust(bottom=0.15, left=0.15)

        self.axes = self.fig.add_subplot(111)
        AxPos  = [0.16, 0.16, 0.82, 0.82]
        self.axes.set_position(AxPos)
        self.axes.set_xlabel("E[meV]")
        self.axes.set_ylabel("Count")
        
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)



class BrowseCanvas(TASCanvas):
    def __init__(self, *args, **kwargs):
        super(BrowseCanvas, self).__init__(*args, **kwargs)
        

class GraphCanvas(TASCanvas):
    def __init__(self, *args, **kwargs):
        super(GraphCanvas, self).__init__(*args, **kwargs)
        AxPos  = [0.10, 0.10, 0.82, 0.82]
        self.axes.set_position(AxPos)
        TASCanvas.updateGeometry(self)
        
class ContourCanvas(FigureCanvas):
    def __init__(self, parent=None, width=531, height=531, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', edgecolor='k')
        self.fig.subplots_adjust(bottom=0.15, left=0.15)

        self.axes = self.fig.add_subplot(111)
        #AxPos  = [0.16, 0.16, 0.82, 0.82]
        #self.axes.set_position(AxPos)
        #self.axes.set_xlabel("E[meV]")
        #self.axes.set_ylabel("Count")
        
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)







class MainWindow(QtWidgets.QDialog):
    
    def closeEvent(self, event):
        QtWidgets.QApplication.quit()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        


        uic.loadUi(os.path.join(os.path.dirname(__file__), 'ui\\DataBrowser.ui'), self)
        self.expPath='C:/'
        self.dataPath=''
        self.filelist=[]
        self.scanlist=[]
        self.curData=pd.DataFrame()
        self.curScanInf= {}
        
        self.cs = None
        #self.
        
        self.initialcurve=None
        self.fitcurve=None
        
        self.taipan= Taipan( expnum='1200', title="no decided", sample="Unknown Sample", user=['UnknowA', 'UnknowB'])
        self.sika= Sika( expnum='100', title="no decided", sample="Unknown Sample", user=['UnknowA', 'UnknowB'])

        self.browsePlot = QVBoxLayout(self.browsePlotCanvas)
        self.graphPlot  = QVBoxLayout(self.graphPlotCanvas)
        self.contourPlot = QVBoxLayout(self.contourPlotCanvas)



        
 
        
        self.initUI()


    def initUI(self):
        
        
        # TEST CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        try:
            self.clearLayout(self.browsePlot)
            self.clearLayout(self.graphPlot)

        except:
            pass
        
        self.radioTaipan.setChecked(True)

        self.browseCanvas = TASCanvas(self.browsePlotCanvas, width=451, height=281, dpi=100)
        self.browsePlot.addWidget(self.browseCanvas)

        self.graphCanvas = TASCanvas(self.graphPlotCanvas, width=531, height=341, dpi=100)
        self.graphPlot.addWidget(self.graphCanvas)
        
        self.contourCanvas = ContourCanvas(self.contourPlotCanvas, width=531, height=531, dpi=100)
        self.contourPlot.addWidget(self.contourCanvas)
        
        self.btnExpPath.clicked.connect(self.exp_path_selection)
        self.spinboxScanNum.valueChanged.connect(self.scan_number_changed)
        self.editExpPath.textChanged.connect(self.path_changed)
        self.radioTaipan.clicked.connect(self.instr_changed)
        self.radioSika.clicked.connect(self.instr_changed)
        self.mainTab.currentChanged.connect(self.tabChanged)
        self.spinNoPeaks.valueChanged.connect(self.peakmum_changed)
        self.btnGraphFit.clicked.connect(self.btnGraphFit_clicked)
        self.btnGraphInitial.clicked.connect(self.btnGraphInitial_clicked)
        self.spinBoxGraph.valueChanged.connect(self.graph_number_changed)
        self.btnPlotContour.clicked.connect(self.btnPlotContour_clicked)  
        self.btnGraphPlot.clicked.connect(self.btnGraphPlot_clicked) 
        
        
        self.btnScanSim.clicked.connect(self.btnScanSim_clicked)
        
        self.btnSaveScanList.clicked.connect(self.btnSaveScanList_clicked)
        
        self.btnSaveContour.clicked.connect(self.btnSaveContour_clicked)
        #self.yMinMax.returnPressed.connect(self.yMinMax_return)
        self.btnChooseExtDataFile.clicked.connect(self.choose_ext_data_file)
        self.btnBatchSim.clicked.connect(self.btnBatchSim_clicked)
        
        
        self.btnSaveContour.clicked.connect(self.btnSaveContour_clicked)
        
        self.btnSikaScanSim.clicked.connect(self.btnSikaScanSim_clicked)
        self.btnSikaBatchSim.clicked.connect(self.btnSikaBatchSim_clicked)
        
        return
    
    
    def update(self):

        return
    


    def exp_path_selection(self):
        self.expPath  = QFileDialog.getExistingDirectory(self, "Select Directory", "C:\\Data\\NeutronData\\Taipan")
        self.dataPath = self.expPath+'/Datafiles/'
        
        if not os.path.isdir(self.dataPath):
            errorbox=QMessageBox()
            errorbox.setText("The path does not exist!")
            errorbox.exec_()
        else: 
    
            self.filelist = os.listdir(self.expPath+'/Datafiles')
            
            self.scanlist = []
            self.bTaipan = False
            nonDatafileList=list([])
    
            for filename in self.filelist:
    
                if filename[-4:]=='.dat' and filename[0:10]=="TAIPAN_exp":
                    self.bTaipan = True
                    self.taipan.expnum=filename[10:14]
                    self.radioTaipan.setChecked(True)
                    self.radioSika.setChecked(False)
                    if len(filename) == 29:
                        self.scanlist.append(int(filename[-10:-4]))
                    elif len(filename) == 28:
                        self.scanlist.append(int(filename[-9:-4]))
                    else:
                        print("Error with the filename.")
                    
                elif len(filename) == 15 and filename.find('_')==4 and filename[-4:]=='.dat':
                    self.bTaipan = False
                    self.sika.expnum=filename[0:4]
                    self.radioTaipan.setChecked(False)
                    self.radioSika.setChecked(True)
                    self.scanlist.append(int(filename[-10:-4]))
                else:
                    nonDatafileList.append(filename)
                
            for filename in nonDatafileList:
                self.filelist.remove(filename)
                    
            #print(self.scanlist)
            self.spinboxScanNum.setRange(min(self.scanlist), max(self.scanlist))
    
            
            if self.bTaipan:
                expnumpos=self.filelist[0].find('exp')+3
                self.editExpNum.setText(self.filelist[0][expnumpos:expnumpos+4])  #expNNNN
                #print(self.scanlist[0])
                dflist=self.taipan.taipan_scanlist_to_dflist(path=self.dataPath, scanlist=[self.scanlist[0]])
                self.curData=dflist[0].copy()
                self.curScanInf=dflist[0].attrs.copy()
                self.browseCanvas.axes.clear()
                self.browseCanvas.axes.plot(dflist[0][:][self.curScanInf['scanax1']],dflist[0][:]['detector'],'o-')
                self.browseCanvas.axes.grid(axis='both', which='major')
                self.browseCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                self.browseCanvas.axes.set_xlabel(self.curScanInf['scanax1'])
                self.browseCanvas.axes.set_ylabel("Count")
                self.browseCanvas.draw()
                
                self.taipanlog=self.taipan.export_scanlist(path=self.expPath, logfile='LogFile.txt') #export_scantitle
                #print(taipanlog.columns.to_list())
                self.logTable.setModel(pandasModel(self.taipanlog))
                for col in range(self.logTable.horizontalHeader().count()):
                    if col ==0:
                        self.logTable.setColumnWidth(col, 80)
                    if col ==1:
                        self.logTable.setColumnWidth(col, 200)
                    if col ==2:
                        self.logTable.setColumnWidth(col, 400)
                        
                self.editExpPath.setText(str(self.expPath))
                self.spinboxScanNum.setValue(self.scanlist[0])
            
                with open(self.dataPath + self.filelist[0], "r") as f:
                    #print("open file:")
                    #print(self.filelist[0])
                    datafiletxt = f.read()
            
                self.txtDataFile.setPlainText(datafiletxt)
                self.txtHKLW.setPlainText(dflist[0].to_string())
                
                ####GraphTab
                self.curData=dflist[0].copy()
                self.curScanInf=dflist[0].attrs.copy()
                self.graphCanvas.axes.clear()
                self.graphCanvas.axes.plot(dflist[0][:][self.curScanInf['scanax1']],dflist[0][:]['detector'],'o-')
                self.graphCanvas.axes.grid(axis='both', which='major')
                self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                self.graphCanvas.axes.set_xlabel(self.curScanInf['scanax1'])
                self.graphCanvas.axes.set_ylabel("Count")
                self.graphCanvas.draw()
                motorlist=dflist[0].columns.tolist()
                self.comboXAxis.clear()
                self.comboXAxis.addItems(motorlist)
                curIndex=self.comboXAxis.findText(self.curScanInf['scanax1'])
                self.comboXAxis.setCurrentIndex(curIndex)
                
                ####GraphTab
                
            else:
    
                self.editExpNum.setText(self.filelist[0][0:4])  #NNNN
    
                dflist=self.sika.sika_batch_reduction(path=self.dataPath, scanlist=[self.scanlist[0]])
                if dflist is None:
                    del self.scanlist[0]
                    del self.filelist[0]
                    errorbox=QMessageBox()
                    errorbox.setText("There is no data in this data file.")
                    errorbox.exec_()
                    
                else:
                    self.curData=dflist[0].copy()
                    self.curScanInf=dflist[0].attrs.copy()  
                    self.browseCanvas.axes.clear()
                    self.browseCanvas.axes.plot(dflist[0][:][self.curScanInf['scanax1']],dflist[0][:]['detector'],'o-')
                    self.browseCanvas.axes.grid(axis='both', which='major')
                    self.browseCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                    self.browseCanvas.axes.set_xlabel(self.curScanInf['scanax1'])
                    self.browseCanvas.axes.set_ylabel("Count")
                    self.browseCanvas.draw()
                    
                    self.sikalog=self.sika.export_scanlist(path=self.dataPath,datafromto=[self.scanlist[0], self.scanlist[-1]])
                    self.logTable.setModel(pandasModel(self.sikalog))
                    for col in range(self.logTable.horizontalHeader().count()):
                        if col ==0:
                            self.logTable.setColumnWidth(col, 100)
                        else:
                            self.logTable.setColumnWidth(col, 250)
                    
                    self.editExpPath.setText(str(self.expPath))
                    self.spinboxScanNum.setValue(self.scanlist[0])
                    
                    with open(self.dataPath + self.filelist[0], "r") as f:
                        datafiletxt = f.read()
                
                    self.txtDataFile.setPlainText(datafiletxt)
                    self.txtHKLW.setPlainText(dflist[0].to_string())
                    
                    ##########GraphTab
                    self.curData=dflist[0].copy()
                    self.curScanInf=dflist[0].attrs.copy() 
                    self.graphCanvas.axes.clear()
                    self.graphCanvas.axes.plot(dflist[0][:][self.curScanInf['scanax1']],dflist[0][:]['detector'],'o-')
                    self.graphCanvas.axes.grid(axis='both', which='major')
                    self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                    self.graphCanvas.axes.set_xlabel(self.curScanInf['scanax1'])
                    self.graphCanvas.axes.set_ylabel("Count")
                    self.graphCanvas.draw()
                    motorlist=dflist[0].columns.tolist()
                    self.comboXAxis.clear()
                    self.comboXAxis.addItems(motorlist)
                    curIndex=self.comboXAxis.findText(self.curScanInf['scanax1'])
                    self.comboXAxis.setCurrentIndex(curIndex)
                    
                    ##########GraphTab
                    
                    
    def choose_ext_data_file(self):
        extDataFilePath,_  = QFileDialog.getOpenFileName(self, "Open File", "C:\\Data\\NeutronData\\Taipan", "taipan data files (*.dat)")
        #self.dataPath = self.expPath+'/Datafiles/'

        
        extDataname = os.path.basename(extDataFilePath)
        extDataPath = os.path.dirname(extDataFilePath)
        
        expnum=extDataname[10:14]  #TAIPAN_exp1295_scan103860
        #print(expnum)
        
        strScanNo = extDataname[extDataname.find('scan')+4:-4]
        #print(strScanNo)
        if strisint(strScanNo):
            extScanNo=int(strScanNo)
            self.extExp=Taipan( expnum=expnum, title="extData", sample="Unknown Sample", user=['UnknowA', 'UnknowB'])
            extdflist=self.extExp.taipan_scanlist_to_dflist(path=extDataPath, scanlist=[extScanNo])
            #self.extData=extdflist[0].copy()
            #self.extScanInf=extinfolist[0].copy()
            #self.browseCanvas.axes.clear()
            #print(extdflist)
            self.browseCanvas.axes.plot(extdflist[0][:][extdflist[0].attrs['scanax1']],extdflist[0][:]['detector'],'r-')
            #self.browseCanvas.axes.grid(axis='both', which='major')
            #self.browseCanvas.axes.grid(axis='both', which='minor', linestyle='--')
            #self.browseCanvas.axes.set_xlabel(extinfolist[0]['scanax'])
            #self.browseCanvas.axes.set_ylabel("Count")
            self.browseCanvas.draw()
    

        return


    def path_changed(self):
        
        self.expPath = self.editExpPath.text()
        if not os.path.isdir(self.expPath):
            errorbox=QMessageBox()
            errorbox.setText("The path does not exist!")
            errorbox.exec_()
            return
        else:
                
            self.dataPath=self.expPath+'/Datafiles/'
    
            self.filelist=os.listdir(self.dataPath)
            self.scanlist=[]
            self.bTaipan=False
            nonDatafileList=list([])
    
            for filename in self.filelist:
    
                if  filename[-4:]=='.dat' and filename[0:10]=="TAIPAN_exp":
                    self.bTaipan=True
                    self.taipan.expnum=filename[10:14]
                    self.radioTaipan.setChecked(True)
                    self.radioSika.setChecked(False)
                    #print(filename[-10:-4])
                    
                    if len(filename) == 29:
                        self.scanlist.append(int(filename[-10:-4]))
                    elif len(filename) == 28:
                        self.scanlist.append(int(filename[-9:-4]))
                    else:
                        print("Error with the filename.")
                elif len(filename) == 15 and filename.find('_')==4 and filename[-4:]=='.dat':
                    self.bTaipan=False
                    self.sika.expnum=filename[0:4]
                    self.radioTaipan.setChecked(False)
                    self.radioSika.setChecked(True)
                    self.scanlist.append(int(filename[-10:-4]))
                else:
                    nonDatafileList.append(filename)
                
            for filename in nonDatafileList:
                self.filelist.remove(filename)
                    
            self.spinboxScanNum.setRange(min(self.scanlist), max(self.scanlist))
            self.spinBoxGraph.setRange(min(self.scanlist), max(self.scanlist))
            
            
            if self.bTaipan:
                expnumpos=self.filelist[0].find('exp')+3
                self.editExpNum.setText(self.filelist[0][expnumpos:expnumpos+4])  #expNNNN
    
                dflist=self.taipan.taipan_scanlist_to_dflist(path=self.dataPath, scanlist=[self.scanlist[0]])
                self.curData=dflist[0].copy()
                self.browseCanvas.axes.clear()
                self.browseCanvas.axes.plot(dflist[0][:][dflist[0].attrs['scanax1']],dflist[0][:]['detector'],'o-')
                self.browseCanvas.axes.grid(axis='both', which='major')
                self.browseCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                self.browseCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
                self.browseCanvas.axes.set_ylabel("Count")
                #minY, maxY=self.browseCanvas.axes.get_ylim()
                #self.yMinMax.setText("{:.2f}:{:.2f}".format(minY, maxY))
    
                self.browseCanvas.draw()
                self.taipanlog=self.taipan.export_scanlist(path=self.expPath, logfile='LogFile.txt') #export_scantitle
                #self.txtExpLog.setPlainText(taipanlog.to_string())
                #taipanlog.drop(['scanno'], axis=1)
                #taipanlogB=taipanlog[['scanno', 'scan_num', 'command', 'scantitle']].copy()
                #print("path changed:")
                #print(taipanlog.columns.to_list())
                self.logTable.setModel(pandasModel(self.taipanlog))
                #print(self.logTable.horizontalHeader().count())
                for col in range(self.logTable.horizontalHeader().count()):
                    if col ==0:
                        self.logTable.setColumnWidth(col, 100)
                    if col ==1:
                        self.logTable.setColumnWidth(col, 100)
                    if col ==2:
                        self.logTable.setColumnWidth(col, 400)
                
                self.editExpPath.setText(str(self.expPath))
                self.spinboxScanNum.setValue(self.scanlist[0])
                
                with open(self.dataPath + self.filelist[0], "r") as f:
                    datafiletxt = f.read()
            
                self.txtDataFile.setPlainText(datafiletxt)
                self.txtHKLW.setPlainText(dflist[0].to_string())
                
                ####GraphTab
                self.curData=dflist[0].copy()
                self.curScanInf=dflist[0].attrs.copy()
                self.graphCanvas.axes.clear()
                self.graphCanvas.axes.plot(dflist[0][:][dflist[0].attrs['scanax1']],dflist[0][:]['detector'],'o-')
                self.graphCanvas.axes.grid(axis='both', which='major')
                self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                self.graphCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
                self.graphCanvas.axes.set_ylabel("Count")
                self.graphCanvas.draw()
                motorlist=dflist[0].columns.tolist()
                self.comboXAxis.clear()
                self.comboXAxis.addItems(motorlist)
                curIndex=self.comboXAxis.findText(dflist[0].attrs['scanax1'])
                self.comboXAxis.setCurrentIndex(curIndex)
                
                ####GraphTab
    
            else:
    
                
                self.editExpNum.setText(self.filelist[0][0:4])  #NNNN
    
                dflist=self.sika.sika_scanlist_to_dflist(path=self.dataPath, scanlist=[self.scanlist[0]])
                if dflist is None:
                    del self.scanlist[0]
                    del self.filelist[0]
                    errorbox=QMessageBox()
                    errorbox.setText("There is no data in this data file.")
                    errorbox.exec_()
                    
                else:
                    self.curData=dflist[0].copy()
                    self.curScanInf=dflist[0].attrs.copy()
                    self.browseCanvas.axes.clear()
                    self.browseCanvas.axes.plot(dflist[0][:][dflist[0].attrs['scanax1']], dflist[0][:]['detector'],'o-')
                    self.browseCanvas.axes.grid(axis='both', which='major')
                    self.browseCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                    self.browseCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
                    self.browseCanvas.axes.set_ylabel("Count")
        
                    self.browseCanvas.draw()
                    
                    sikalog=self.sika.export_scanlist(path=self.dataPath,datafromto=[self.scanlist[0], self.scanlist[-1]])
                    self.logTable.setModel(pandasModel(sikalog))
                   
                    for col in range(self.logTable.horizontalHeader().count()):
                        if col ==0:
                            self.logTable.setColumnWidth(col, 80)
                        else:
                            self.logTable.setColumnWidth(col, 250)
                    
                    self.editExpPath.setText(str(self.expPath))
                    self.spinboxScanNum.setValue(self.scanlist[0])
                    
                    with open(self.dataPath + self.filelist[0], "r") as f:
                        datafiletxt = f.read()
                
                    self.txtDataFile.setPlainText(datafiletxt)
                    self.txtHKLW.setPlainText(dflist[0].to_string())

                    ##########GraphTab
                    self.curData=dflist[0].copy()
                    self.curScanInf=dflist[0].attrs.copy()
                    self.graphCanvas.axes.clear()
                    self.graphCanvas.axes.plot(dflist[0][:][dflist[0].attrs['scanax1']],dflist[0][:]['detector'],'o-')
                    self.graphCanvas.axes.grid(axis='both', which='major')
                    self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                    self.graphCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
                    self.graphCanvas.axes.set_ylabel("Count")
                    self.graphCanvas.draw()
                    
                    motorlist=dflist[0].columns.tolist()
                    self.comboXAxis.clear()
                    self.comboXAxis.addItems(motorlist)
                    self.comboXAxis.addItems(motorlist)
                    curIndex=self.comboXAxis.findText(dflist[0].attrs['scanax1'])
                    self.comboXAxis.setCurrentIndex(curIndex)
                    
                    ##########GraphTab            
        
      

    def instr_changed(self):
        if self.radioTaipan.isChecked():
            self.bTaipan = True
            self.exp_path_selection()
            
        else:
            self.bTaipan=False
            self.exp_path_selection()
            


        

    def scan_number_changed(self):
  
        currentscan = self.spinboxScanNum.value()
        currentfilename=''

        if self.radioTaipan.isChecked():
            #currentscan = self.spinboxScanNum.value()
            
            for filename in self.filelist:
                if  filename[-4:]=='.dat' and filename[0:10]=="TAIPAN_exp":
                    #print("debugging")
                    #print(filename[filename.find('scan')+4:-4])
                    if int(filename[filename.find('scan')+4:-4])==currentscan:   #taipan file name: TAIPAN_exp1305_scan106112.dat
                        currentfilename=filename            
            
            if currentfilename == '':
                errorbox=QMessageBox()
                errorbox.setText("There is no datafile #{} in this folder.".format(currentscan))
                errorbox.exec_()
                return
            else:
                dflist = self.taipan.taipan_scanlist_to_dflist(path=self.dataPath, scanlist=[currentscan])
                self.curData=dflist[0].copy()
                self.curScanInf=dflist[0].copy()
                if not self.checkOverplot.isChecked():
                    self.browseCanvas.axes.clear()
                dataX=dflist[0][:][dflist[0].attrs['scanax1']].to_numpy()
                dataY=dflist[0][:]['detector'].to_numpy()
                self.browseCanvas.axes.plot(dataX,dataY,'o-')
                self.browseCanvas.axes.grid(axis='both', which='major')
                self.browseCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                self.browseCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
                self.browseCanvas.axes.set_ylabel("Count")
                
                if self.checkFit.isChecked():
                    if len(dataX) > 4:
                        cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G')
                        self.browseCanvas.axes.plot(cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
    
                        #self.txtFitParam.setPlainText(u"Int: {0:6.2f} \u00B1 {1:6.2f}\nx0: {2:4.4f} \u00B1 {3:4.4f}\nW: {4:4.4f} \u00B1 {5:4.4f}".format(cur_fitpar['A'][0],cur_fitpar['A_err'][0],cur_fitpar['x0'][0],cur_fitpar['x0_err'][0],cur_fitpar['w'][0],cur_fitpar['w_err'][0]))
                        if cur_fitpar['A_err'][0] == None or cur_fitpar['x0_err'][0] == None or cur_fitpar['w_err'][0] == None:
                            self.txtFitParam.setPlainText(u"Fitted Parameters:\nInt: {0} \u00B1 {1}\nx0: {2} \u00B1 {3}\nW: {4} \u00B1 {5}".format(cur_fitpar['A'][0],cur_fitpar['A_err'][0],cur_fitpar['x0'][0],cur_fitpar['x0_err'][0],cur_fitpar['w'][0],cur_fitpar['w_err'][0]))
                        else: 
                            self.txtFitParam.setPlainText(u"Fitted Parameters:\nInt: {0:6.2f} \u00B1 {1:6.2f}\nx0: {2:4.4f} \u00B1 {3:4.4f}\nW: {4:4.4f} \u00B1 {5:4.4f}".format(cur_fitpar['A'][0],cur_fitpar['A_err'][0],cur_fitpar['x0'][0],cur_fitpar['x0_err'][0],cur_fitpar['w'][0],cur_fitpar['w_err'][0]))
                    else:
                        self.txtFitParam.setPlainText("Not enough data to fit.")
                else:
                    self.txtFitParam.setPlainText("")
                if self.chkMinMax.isChecked():
                    #minY, maxY=self.
                    minmaxYstr=self.yMinMax.text()
                    minmaxValues=minmaxYstr.split(":")
                    #print(minmaxValues)
                    if len(minmaxValues)>1:
                        if strisfloat(minmaxValues[0]) and strisfloat(minmaxValues[1]):
                            self.browseCanvas.axes.set_ylim(float(minmaxValues[0]),float(minmaxValues[1]))
                        #self.contourCanvas.draw_idle()
                        #print("set y lim done")
                    #else:
                        #print("The input in minmax text box is wrong!")
    
                self.browseCanvas.draw()
                #minY, maxY=self.browseCanvas.axes.get_ylim()
                #self.yMinMax.setText("{:.2f}:{:.2f}".format(minY, maxY))               

                with open(self.dataPath + currentfilename, "r") as f:
                    datafiletxt = f.read()
                self.txtDataFile.setPlainText(datafiletxt)
                self.txtHKLW.setPlainText(dflist[0].to_string())

        elif self.radioSika.isChecked():
            
            for filename in self.filelist:
               if int(filename[-10:-4])==currentscan:
                   currentfilename=filename
            
            if currentfilename == '':
                errorbox=QMessageBox()
                errorbox.setText("There is no datafile #{} in this folder.".format(currentscan))
                errorbox.exec_()
                return
            else:

                dflist = self.sika.sika_scanlist_to_dflist(path=self.dataPath, scanlist=[currentscan])
                if dflist is None:
                    del self.scanlist[0]
                    del self.filelist[0]
                    errorbox=QMessageBox()
                    errorbox.setText("There is no data in this data file.")
                    errorbox.exec_()
                    
                else:
                    #if dflist
                    self.curData=dflist[0].copy()
                    self.curScanInf=dflist[0].attrs.copy()
                    if not self.checkOverplot.isChecked():
                        self.browseCanvas.axes.clear()
                    #print(dflist[0][:][infolist[0]['scanax']])   #
                    dataX=dflist[0][:][dflist[0].attrs['scanax1']].to_numpy()
                    dataY=dflist[0][:]['detector'].to_numpy()
                    self.browseCanvas.axes.plot(dataX,dataY,'o-')
                    self.browseCanvas.axes.grid(axis='both', which='major')
                    self.browseCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                    self.browseCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
                    self.browseCanvas.axes.set_ylabel("Count")
                    if self.checkFit.isChecked():
                        if len(dataX) > 4:
                            cur_fitpar, cur_fitdat = fit_peak(dataX, dataY, func='G')
                            self.browseCanvas.axes.plot(cur_fitdat["X"], cur_fitdat["Y_fit"], '-')
    
                            if cur_fitpar['A_err'][0] == None or cur_fitpar['x0_err'][0] == None or cur_fitpar['w_err'][0] == None:
                                self.txtFitParam.setPlainText(u"Fitted Parameters:\nInt: {} \u00B1 {}\nx0: {} \u00B1 {}\nW: {} \u00B1 {}".format(cur_fitpar['A'][0],cur_fitpar['A_err'][0],cur_fitpar['x0'][0],cur_fitpar['x0_err'][0],cur_fitpar['w'][0],cur_fitpar['w_err'][0]))
                            else: 
                                self.txtFitParam.setPlainText(u"Fitted Parameters:\nInt: {0:6.2f} \u00B1 {1:6.2f}\nx0: {2:4.4f} \u00B1 {3:4.4f}\nW: {4:4.4f} \u00B1 {5:4.4f}".format(cur_fitpar['A'][0],cur_fitpar['A_err'][0],cur_fitpar['x0'][0],cur_fitpar['x0_err'][0],cur_fitpar['w'][0],cur_fitpar['w_err'][0]))
                        else:
                            self.txtFitParam.setPlainText("Not enough data to fit.")
                    else:
                        self.txtFitParam.setPlainText("")
                    if self.chkMinMax.isChecked():
                        #minY, maxY=self.
                        minmaxYstr=self.yMinMax.text()
                        minmaxValues=minmaxYstr.split(":")
                        #print(minmaxValues)
                        if len(minmaxValues)>1:
                            if strisfloat(minmaxValues[0]) and strisfloat(minmaxValues[1]):
                                self.browseCanvas.axes.set_ylim(float(minmaxValues[0]),float(minmaxValues[1]))
                    self.browseCanvas.draw()

                    with open(self.dataPath + currentfilename, "r") as f:
                        datafiletxt = f.read()
                
                    self.txtDataFile.setPlainText(datafiletxt)
                    self.txtHKLW.setPlainText(dflist[0].to_string())
        
        return



    def graph_number_changed(self):
        #self.expPath = self.editExpPath.text()
        self.dataPath=self.expPath+'/Datafiles/'

        if self.radioTaipan.isChecked():
            currentscan = self.spinBoxGraph.value()
            if self.checkGraphComb.isChecked():
                tempscannostr=self.combScanNo.text()
                addlist=list(map(int, tempscannostr.split('+')))
                addlist.append(currentscan)
                print(addlist)
                df_comb = self.taipan.tas_datacombine(path=self.dataPath, scanlist=addlist)
                #print(dflist)
                dflist=[df_comb]

            else:
                dflist = self.taipan.taipan_scanlist_to_dflist(path=self.dataPath, scanlist=[currentscan])
                    
            self.curData=dflist[0].copy()
            self.curScanInf=dflist[0].attrs.copy()
            if not self.checkOverGraph.isChecked():
                self.graphCanvas.axes.clear()
            dataX=dflist[0][:][dflist[0].attrs['scanax1']].to_numpy()
            dataY=dflist[0][:]['detector'].to_numpy()
            self.graphCanvas.axes.plot(dataX,dataY,'o-')
            self.graphCanvas.axes.grid(axis='both', which='major')
            self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
            self.graphCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
            self.graphCanvas.axes.set_ylabel("Count")
            
            if self.checkGraphComp.isChecked():
                compScanNoStr=self.compScanNo.text()
                complist=parse_number_string(compScanNoStr)
                compdflist=self.taipan.taipan_scanlist_to_dflist(path=self.dataPath, scanlist=[complist])
                for eachScan in compdflist:
                    compDataX=eachScan[eachScan.attrs['scanax1']].to_numpy()
                    compDataY=eachScan['detector'].to_numpy()
                    self.graphCanvas.axes.plot(compDataX,compDataY,'--')
            ###########there is a problem here. cannot use number+number for the compare data.     
                
            if self.chkMinMaxG.isChecked():
                #minY, maxY=self.
                minmaxYGstr = self.yMinMaxGraph.text()
                minmaxGvalues      = minmaxYGstr.split(":")
                print(minmaxGvalues)
                if len(minmaxGvalues)>1:
                    if strisfloat(minmaxGvalues[0]) and strisfloat(minmaxGvalues[1]):
                        self.graphCanvas.axes.set_ylim(float(minmaxGvalues[0]),float(minmaxGvalues[1]))

            self.graphCanvas.draw()
            #print(infolist[0]['scanax'])
            curIndex=self.comboXAxis.findText(dflist[0].attrs['scanax1'])
            self.comboXAxis.setCurrentIndex(curIndex)
            

        elif self.radioSika.isChecked():

            currentscan = self.spinBoxGraph.value()

            dflist = self.sika.sika_scanlist_to_dflist(path=self.dataPath, scanlist=[currentscan])
            
            if dflist is None:
                del self.scanlist[0]
                del self.filelist[0]
                errorbox=QMessageBox()
                errorbox.setText("There is no data in this data file.")
                errorbox.exec_()
                return
                
            else:
                self.curData=dflist[0].copy()
                self.curScanInf=dflist[0].attrs.copy()
                if not self.checkOverGraph.isChecked():
                    self.graphCanvas.axes.clear()
                dataX = dflist[0][:][dflist[0].attrs['scanax1']].to_numpy()
                dataY = dflist[0][:]['detector'].to_numpy()
                self.graphCanvas.axes.plot(dataX,dataY,'o-')
                self.graphCanvas.axes.grid(axis='both', which='major')
                self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                self.graphCanvas.axes.set_xlabel(dflist[0].attrs['scanax1'])
                self.graphCanvas.axes.set_ylabel("Count")
                if self.chkMinMaxG.isChecked():
                    #minY, maxY=self.
                    minmaxYGstr = self.yMinMaxGraph.text()
                    minmaxGvalues      = minmaxYGstr.split(":")
                    #print(minmaxValues)
                    if len(minmaxGvalues)>1:
                        if strisfloat(minmaxGvalues[0]) and strisfloat(minmaxGvalues[1]):
                            self.graphCanvas.axes.set_ylim(float(minmaxGvalues[0]),float(minmaxGvalues[1]))    
    
                self.graphCanvas.draw()
                print(dflist[0].attrs['scanax1'])
                curIndex=self.comboXAxis.findText(dflist[0].attrs['scanax1'])
                self.comboXAxis.setCurrentIndex(curIndex)
                
                return
        





    def tabChanged(self, index):

        if index == 1:
            #self.groupPeakA.setEnabled(False)
            #self.groupPeakB.setEnabled(False)
            #self.groupPeakC.setEnabled(False)
            if self.spinNoPeaks.value() == 1:
                self.groupPeakA.setChecked(True)
                self.groupPeakB.setChecked(False)
                self.groupPeakC.setChecked(False)
            if self.spinNoPeaks.value() == 2:

                self.groupPeakA.setChecked(True)
                self.groupPeakB.setChecked(True)
                self.groupPeakC.setChecked(False)
            if self.spinNoPeaks.value() == 3:

                self.groupPeakA.setChecked(True)
                self.groupPeakB.setChecked(True)
                self.groupPeakC.setChecked(True)
            
    def btnSaveScanList_clicked(self):
        if self.radioTaipan.isChecked(): 
            path, ok = QFileDialog.getSaveFileName(self, 'Save CSV', os.getenv('HOME'), 'CSV(*.csv)')
            if ok:
                self.taipanlog.to_csv(path)
        elif self.radioSika.isChecked():
            path, ok = QFileDialog.getSaveFileName(self, 'Save CSV', os.getenv('HOME'), 'CSV(*.csv)')
            if ok:
                self.sikalog.to_csv(path)

        return
                        
    def peakmum_changed(self):  
        if self.spinNoPeaks.value() == 1:
            self.groupPeakA.setChecked(True)
            self.groupPeakB.setChecked(False)
            self.groupPeakC.setChecked(False) 
        elif self.spinNoPeaks.value() == 2:
            self.groupPeakA.setChecked(True)
            self.groupPeakB.setChecked(True)
            self.groupPeakC.setChecked(False)
        elif self.spinNoPeaks.value() == 3:
            self.groupPeakA.setChecked(True)
            self.groupPeakB.setChecked(True)
            self.groupPeakC.setChecked(True)     
            
            

    def btnGraphPlot_clicked(self):
        plotXaxis=self.comboXAxis.currentText()
        
        
        if self.radioTaipan.isChecked():
            currentscan = self.spinBoxGraph.value()
            
            dflist = self.taipan.taipan_scanlist_to_dflist(path=self.dataPath, scanlist=[currentscan])
            self.curData=dflist[0].copy()
            self.curScanInf=dflist[0].attrs.copy()
            if not self.checkOverGraph.isChecked():
                self.graphCanvas.axes.clear()
            
            dataX=self.curData[plotXaxis].to_numpy()
            dataY=self.curData['detector'].to_numpy()
            self.graphCanvas.axes.plot(dataX,dataY,'o-')
            self.graphCanvas.axes.grid(axis='both', which='major')
            self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
            self.graphCanvas.axes.set_xlabel(plotXaxis)
            self.graphCanvas.axes.set_ylabel("Count")


            self.graphCanvas.draw()
            
            
        elif self.radioSika.isChecked():

            currentscan = self.spinBoxGraph.value()

            dflist = self.sika.sika_scanlist_to_dflist(path=self.dataPath, scanlist=[currentscan])
            
            if dflist is None:
                del self.scanlist[0]
                del self.filelist[0]
                errorbox=QMessageBox()
                errorbox.setText("There is no data in this data file.")
                errorbox.exec_()
                return
                
            else:
                self.curData=dflist[0].copy()
                self.curScanInf=dflist[0].attrs.copy()
                if not self.checkOverGraph.isChecked():
                    self.graphCanvas.axes.clear()
                dataX=self.curData[plotXaxis].to_numpy()
                dataY=self.curData['detector'].to_numpy()
                self.graphCanvas.axes.plot(dataX,dataY,'o-')
                self.graphCanvas.axes.grid(axis='both', which='major')
                self.graphCanvas.axes.grid(axis='both', which='minor', linestyle='--')
                self.graphCanvas.axes.set_xlabel(plotXaxis)
                self.graphCanvas.axes.set_ylabel("Count")
    
    
                self.graphCanvas.draw()
                
                
        return
                
                    
            
    def btnGraphFit_clicked(self):
        pars = Parameters()

        pars.add('GL_1', value = 1, vary=False)
        pars.add('A_1',  value = self.ampValueA.value(),   vary=not(self.bFixA1.isChecked()))
        pars.add('x0_1', value = self.x0ValueA.value(),    vary=not(self.bFixx01.isChecked()))
        pars.add('w_1',  value = self.widthValueA.value(), vary=not(self.bFixw1.isChecked()))
        
        pars.add('GL_2', value = 1, vary=False)
        pars.add('A_2',  value = self.ampValueB.value(),   vary=not(self.bFixA2.isChecked()))
        pars.add('x0_2', value = self.x0ValueB.value(),    vary=not(self.bFixx02.isChecked()))
        pars.add('w_2',  value = self.widthValueB.value(), vary=not(self.bFixw2.isChecked()))
        
        pars.add('GL_3', value = 1, vary=False)
        pars.add('A_3',  value = self.ampValueC.value(),   vary=not(self.bFixA3.isChecked()))
        pars.add('x0_3', value = self.x0ValueC.value(),    vary=not(self.bFixx02.isChecked()))
        pars.add('w_3',  value = self.widthValueC.value(), vary=not(self.bFixw2.isChecked()))
        
        pars.add('bg',  value = self.bgValue.value(), vary=not(self.bFixbg.isChecked()))
        pars.add('slope',  value = self.slopeValue.value(), vary=not(self.bFixslope.isChecked()))

        
        if self.groupPeakA.isChecked():
            if self.peakShapeA.currentText == 'Gaussian':
                pars['GL_1'].value = 1
            if self.peakShapeA.currentText == 'Lorentzian':
                pars['GL_1'].value = 2
        else:
            print('error')
            pars['GL_1'].value = 0

                

        if self.groupPeakB.isChecked():
            if self.peakShapeB.currentText == 'Gaussian':
                pars['GL_2'].value = 1
            if self.peakShapeB.currentText == 'Lorentzian':
                pars['GL_2'].value = 2
        else:
            pars['GL_2'].value = 0
            pars['A_2'].vary =  False
            pars['x0_2'].vary = False
            pars['w_2'].vary = False
            

        if self.groupPeakC.isChecked():
            if self.peakShapeC.currentText == 'Gaussian':
                pars['GL_3'].value = 1
            if self.peakShapeC.currentText == 'Lorentzian':
                pars['GL_3'].value = 2
        else:
            pars['GL_3'].value = 0
            pars['A_3'].vary =  False
            pars['x0_3'].vary = False
            pars['w_3'].vary = False  

        dataX=self.curData[:][ self.curScanInf['scanax1']].to_numpy()
        if  self.radioTaipan.isChecked():
            dataY=self.curData[:]['detector'].to_numpy()
        elif self.radioSika.isChecked():
            dataY=self.curData[:]['detector'].to_numpy()
        out   = minimize(three_peaks_on_slope_residual, pars, args=(dataX,),   kws={'data': dataY})
        fittedY    =  three_peaks_on_slope_residual(out.params, dataX)
        
        if self.fitcurve ==None:
            self.fitcurve,=self.graphCanvas.axes.plot(dataX,fittedY,'-')
        else:
            self.fitcurve.remove()
            self.fitcurve,=self.graphCanvas.axes.plot(dataX,fittedY,'-')
        self.graphCanvas.draw()
        print(fit_report(out))
        self.ampValueA.setValue(out.params['A_1'])
        self.x0ValueA.setValue(out.params['x0_1'])
        self.widthValueA.setValue(out.params['w_1'])
        
        self.ampValueB.setValue(out.params['A_2'])
        self.x0ValueB.setValue(out.params['x0_2'])
        self.widthValueB.setValue(out.params['w_2'])
        
        
        self.ampValueC.setValue(out.params['A_3'])
        self.x0ValueC.setValue(out.params['x0_3'])
        self.widthValueC.setValue(out.params['w_3'])
        
        self.bgValue.setValue(out.params['bg'])
        self.slopeValue.setValue(out.params['slope'])
        
        outstr="Fitted Parameters:\n"
        for param in out.params:
            print(param)
            print(out.params[param])
            #if out.params['GL_1'].value ==1:
            #outstr=outstr+param +': '+ str(out.params[param].value)+'\n'
        outstr=outstr + "bg: {0:8.4f} \u00B1 {1:8.4f}\n".format(out.params['bg'].value,         out.params['bg'].stderr)
        outstr=outstr + "slope: {0:8.4f} \u00B1 {1:8.4f}\n\n".format(out.params['slope'].value, out.params['slope'].stderr)
        if out.params['GL_1'].value ==1:
            outstr=outstr + "A_1: {0:8.4f} \u00B1 {1:8.4f}\n".format(out.params['A_1'].value,    out.params['A_1'].stderr)
            outstr=outstr + "x0_1: {0:8.4f} \u00B1 {1:8.4f}\n".format(out.params['x0_1'].value,  out.params['x0_1'].stderr)
            outstr=outstr + "w_1: {0:8.4f} \u00B1 {1:8.4f}\n\n".format(out.params['w_1'].value,  out.params['w_1'].stderr)
        if out.params['GL_2'].value ==1:
            outstr=outstr + "A_2: {0:8.4f} \u00B1 {1:8.4f}\n".format(out.params['A_2'].value,    out.params['A_2'].stderr) 
            outstr=outstr + "x0_2: {0:8.4f} \u00B1 {1:8.4f}\n".format(out.params['x0_2'].value,  out.params['x0_2'].stderr) 
            outstr=outstr + "w_2: {0:8.4f} \u00B1 {1:8.4f}\n\n".format(out.params['w_2'].value,  out.params['w_2'].stderr) 
        if out.params['GL_3'].value ==1:
            outstr=outstr + "A_3: {0:8.4f} \u00B1 {1:8.4f}\n".format(out.params['A_3'].value,    out.params['A_3'].stderr)
            outstr=outstr + "x0_3: {0:8.4f} \u00B1 {1:8.4f}\n".format(out.params['x0_3'].value,  out.params['x0_3'].stderr) 
            outstr=outstr + "w_3: {0:8.4f} \u00B1 {1:8.4f}\n\n".format(out.params['w_3'].value,  out.params['w_3'].stderr) 

    
        self.txtFitParamB.setPlainText(outstr)
                  
            
            
    def btnGraphInitial_clicked(self):
        pars = Parameters()

        pars.add('GL_1', value = 1, vary=False)
        pars.add('A_1',  value = self.ampValueA.value(),   vary=not(self.bFixA1.isChecked()))
        pars.add('x0_1', value = self.x0ValueA.value(),    vary=not(self.bFixx01.isChecked()))
        pars.add('w_1',  value = self.widthValueA.value(), vary=not(self.bFixw1.isChecked()))
        
        pars.add('GL_2', value = 1, vary=False)
        pars.add('A_2',  value = self.ampValueB.value(),   vary=not(self.bFixA2.isChecked()))
        pars.add('x0_2', value = self.x0ValueB.value(),    vary=not(self.bFixx02.isChecked()))
        pars.add('w_2',  value = self.widthValueB.value(), vary=not(self.bFixw2.isChecked()))
        
        pars.add('GL_3', value = 1, vary=False)
        pars.add('A_3',  value = self.ampValueC.value(),   vary=not(self.bFixA3.isChecked()))
        pars.add('x0_3', value = self.x0ValueC.value(),    vary=not(self.bFixx02.isChecked()))
        pars.add('w_3',  value = self.widthValueC.value(), vary=not(self.bFixw2.isChecked()))
        
        pars.add('bg',  value = self.bgValue.value(), vary=not(self.bFixbg.isChecked()))
        pars.add('slope',  value = self.slopeValue.value(), vary=not(self.bFixslope.isChecked()))

        
        if self.groupPeakA.isChecked():
            if self.peakShapeA.currentText == 'Gaussian':
                pars['GL_1'].value = 1
            if self.peakShapeA.currentText == 'Lorentzian':
                pars['GL_1'].value = 2
        else:
            print('error')
            pars['GL_1'].value = 0

                

        if self.groupPeakB.isChecked():
            if self.peakShapeB.currentText == 'Gaussian':
                pars['GL_2'].value = 1
            if self.peakShapeB.currentText == 'Lorentzian':
                pars['GL_2'].value = 2
        else:
            pars['GL_2'].value = 0
            pars['A_2'].vary =  False
            pars['x0_2'].vary = False
            pars['w_2'].vary = False
            

        if self.groupPeakC.isChecked():
            if self.peakShapeC.currentText == 'Gaussian':
                pars['GL_3'].value = 1
            if self.peakShapeC.currentText == 'Lorentzian':
                pars['GL_3'].value = 2
        else:
            pars['GL_3'].value = 0
            pars['A_3'].vary   =  False
            pars['x0_3'].vary  = False
            pars['w_3'].vary   = False  

        dataX=self.curData[:][ self.curScanInf['scanax1']].to_numpy()
        initialY=three_peaks_on_slope_residual(pars,dataX)
        if self.initialcurve ==None:
            self.initialcurve,=self.graphCanvas.axes.plot(dataX,initialY,'-')
        else:
            self.initialcurve.remove()
            self.initialcurve,=self.graphCanvas.axes.plot(dataX,initialY,'-')
        self.graphCanvas.draw()
        
        return
        
    def btnScanSim_clicked(self): 
        
        lattice_str= self.txtLatticeParam.text().split()
        vector_str1=self.txtScatterVector1.text().split()
        vector_str2=self.txtScatterVector2.text().split()
        vector_u=np.array([float(vector_str1[0]), float(vector_str1[1]), float(vector_str1[2])])
        vector_v=np.array([float(vector_str2[0]), float(vector_str2[1]), float(vector_str2[2])])
        
        #print(vector_u)
        #print(vector_v)
        ef_sim=float(self.txtFinalEnergy.text())
        drive_cmd= self.txtDriveCmd.text()
        scan_cmd=self.txtCmdToSim.text()
        
        if len(lattice_str) == 6:
            aa = float(lattice_str[0])
            bb =float(lattice_str[1])
            cc =float(lattice_str[2])
            alpha1=float(lattice_str[3])
            beta1=float(lattice_str[4])
            gamma1=float(lattice_str[5])
            
            exp=self.taipan.taipan_expconfig(ef=ef_sim)
            exp.sample.a=aa
            exp.sample.b=bb
            exp.sample.c=cc
            exp.sample.alpha=alpha1
            exp.sample.beta=beta1
            exp.sample.gamma=gamma1
            exp.mono.tau      = self.taipanMono.currentText() # 'PG(002)'
            collim = self.taipanCollimation.text().split()
            if len(collim) == 4:
                if strisint(collim[0]) and  strisint(collim[1]) and strisint(collim[2]) and strisint(collim[3]):
                    exp.hcol = [collim[0], collim[1], collim[2],120]
            
            exp.sample.u       = vector_u  #np.array([1, 0, 0])
            exp.sample.v       = vector_v  #np.array([0, 0, 1])
            
            if scan_cmd[0:5] == 'mscan':
                output=self.taipan.taipan_scansim(cmdline=scan_cmd, exp=exp)
            elif scan_cmd[0:7] == 'runscan':
                output=self.taipan.taipan_scansim(cmdline=drive_cmd+'\n'+scan_cmd, exp=exp)
            self.txtScanSimOutput.setPlainText('')
            #print(output.columns.values)
            self.txtScanSimOutput.setPlainText(output.to_string(float_format=lambda x: "{:.4f}".format(x)))
            #output.to_string()
        else:
            print("error")
            
        return
    
    
    def btnBatchSim_clicked(self): 
        
        batchFile, _  = QFileDialog.getOpenFileName(self, "Open File", "C:\\", "taipan scan batch files (*.txt)")
        
        validation_output,valid_cmd_list=self.taipan.taipan_batch_validate(batchFile)

        
        
        lattice_str= self.txtLatticeParam.text().split()
        vector_str1=self.txtScatterVector1.text().split()
        vector_str2=self.txtScatterVector2.text().split()
        vector_u=np.array([float(vector_str1[0]), float(vector_str1[1]), float(vector_str1[2])])
        vector_v=np.array([float(vector_str2[0]), float(vector_str2[1]), float(vector_str2[2])])
        
        #print(vector_u)
        #print(vector_v)
        ef_sim=float(self.txtFinalEnergy.text())
        #drive_cmd= self.txtDriveCmd.text()
        #scan_cmd=self.txtCmdToSim.text()
        sim_out_str = ""
        temp_index=99
        #output=None
        total_no_point=0
        
        if len(lattice_str) == 6:
            aa = float(lattice_str[0])
            bb =float(lattice_str[1])
            cc =float(lattice_str[2])
            alpha1=float(lattice_str[3])
            beta1=float(lattice_str[4])
            gamma1=float(lattice_str[5])
            
            exp=self.taipan.taipan_expconfig(ef=ef_sim)
            exp.sample.a=aa
            exp.sample.b=bb
            exp.sample.c=cc
            exp.sample.alpha=alpha1
            exp.sample.beta=beta1
            exp.sample.gamma=gamma1
            exp.mono.tau      = self.taipanMono.currentText() # 'PG(002)'
            collim = self.taipanCollimation.text().split()
            if len(collim) == 4:
                if strisint(collim[0]) and  strisint(collim[1]) and strisint(collim[2]) and strisint(collim[3]):
                    exp.hcol = [collim[0], collim[1], collim[2],120]
            
            exp.sample.u       = vector_u  #np.array([1, 0, 0])
            exp.sample.v       = vector_v  #np.array([0, 0, 1])
            
            for index, scan_line in enumerate(valid_cmd_list):
                if scan_line[0:5] == 'drive':
                    temp_index=index
                elif scan_line[0:5] == 'mscan':
                    tempstr=scan_line
                    #print(scan_line)
                    total_no_point= total_no_point + int(tempstr.split()[-3])
                    output=self.taipan.taipan_scansim(cmdline=scan_line, exp=exp)
                    sim_out_str=sim_out_str +"\n\nCommand:\n" + scan_line+ "\n\n" + output.to_string(float_format=lambda x: "{:.4f}".format(x))
            
                elif scan_line[0:7] == 'runscan':
                    tempstr=scan_line
                    #print(scan_line)
                    total_no_point= total_no_point + int(tempstr.split()[-3])
                    output=self.taipan.taipan_scansim(cmdline=valid_cmd_list[temp_index]+'\n'+scan_line, exp=exp)
                    sim_out_str=sim_out_str +"\n\nCommand:\n" + valid_cmd_list[temp_index]+'\n'+scan_line + "\n\n" + output.to_string(float_format=lambda x: "{:.4f}".format(x))
            
            self.txtScanSimOutput.setPlainText('')
            #print(output.columns.values)
            self.txtScanSimOutput.setPlainText(validation_output+"\n\nHere are the simulated scans:\n"+sim_out_str+"\n\nTotal Number of Data Points: {}".format(total_no_point))
            #output.to_string()
        else:
            print("error")
            
        return
    
    def btnSikaScanSim_clicked(self): 
        
        lattice_str= self.txtLatticeOnSika.text().split()
        vector_str1=self.txtSikaVector1.text().split()
        vector_str2=self.txtSikaVector2.text().split()
        vector_u=np.array([float(vector_str1[0]), float(vector_str1[1]), float(vector_str1[2])])
        vector_v=np.array([float(vector_str2[0]), float(vector_str2[1]), float(vector_str2[2])])
        
        #print(vector_u)
        #print(vector_v)
        ef_sim=float(self.txtSikaEf.text())
        #drive_cmd= self.txtDriveCmd.text()
        scan_cmd=self.txtSikaScanToSim.text()
        
        if len(lattice_str) == 6:
            aa = float(lattice_str[0])
            bb =float(lattice_str[1])
            cc =float(lattice_str[2])
            alpha1=float(lattice_str[3])
            beta1=float(lattice_str[4])
            gamma1=float(lattice_str[5])
            
            exp=self.sika.sika_expconfig(ef=ef_sim)
            exp.sample.a=aa
            exp.sample.b=bb
            exp.sample.c=cc
            exp.sample.alpha=alpha1
            exp.sample.beta=beta1
            exp.sample.gamma=gamma1
            #exp.mono.tau      = self.taipanMono.currentText() # 'PG(002)'
            collim = self.sikaCollimation.text().split()
            if len(collim) == 4:
                if strisint(collim[0]) and  strisint(collim[1]) and strisint(collim[2]) and strisint(collim[3]):
                    exp.hcol = [collim[0], collim[1], collim[2],120]
            
            exp.sample.u       = vector_u  #np.array([1, 0, 0])
            exp.sample.v       = vector_v  #np.array([0, 0, 1])
            
            
            output=self.sika.sika_scansim(command=scan_cmd, exp=exp)
            
            self.txtSikaScanSimOutput.setPlainText('')   #clean the old history
            #print(output.columns.values)
            self.txtSikaScanSimOutput.setPlainText(output.to_string(float_format=lambda x: "{:.4f}".format(x)))
            #output.to_string()
        else:
            print("error")
            
        return
    
    def btnSikaBatchSim_clicked(self): 
        print("button clicked")
        
        batchFile, _  = QFileDialog.getOpenFileName(self, "Open File", "C:\\", "taipan scan batch files (*.txt)")
        
        validation_output,valid_cmd_list=self.sika.sika_batchfile_validate(batchFile)

        
        
        lattice_str = self.txtLatticeOnSika.text().split()
        vector_str1 = self.txtSikaVector1.text().split()
        vector_str2 = self.txtSikaVector2.text().split()
        vector_u = np.array([float(vector_str1[0]), float(vector_str1[1]), float(vector_str1[2])])
        vector_v = np.array([float(vector_str2[0]), float(vector_str2[1]), float(vector_str2[2])])
        
        #print(vector_u)
        #print(vector_v)
        ef_sim = float(self.txtSikaEf.text())
        #drive_cmd= self.txtDriveCmd.text()
        #scan_cmd=self.txtCmdToSim.text()
        sim_out_str = ""
        #temp_index  = 99
        #output=None
        total_no_point = 0
        
        if len(lattice_str) == 6:
            aa = float(lattice_str[0])
            bb =float(lattice_str[1])
            cc =float(lattice_str[2])
            alpha1=float(lattice_str[3])
            beta1=float(lattice_str[4])
            gamma1=float(lattice_str[5])
            
            exp=self.sika.sika_expconfig(ef=ef_sim)
            exp.sample.a=aa
            exp.sample.b=bb
            exp.sample.c=cc
            exp.sample.alpha=alpha1
            exp.sample.beta=beta1
            exp.sample.gamma=gamma1
            
            collim = self.sikaCollimation.text().split()
            if len(collim) == 4:
                if strisint(collim[0]) and  strisint(collim[1]) and strisint(collim[2]) and strisint(collim[3]):
                    exp.hcol = [collim[0], collim[1], collim[2],120]
            
            exp.sample.u       = vector_u  #np.array([1, 0, 0])
            exp.sample.v       = vector_v  #np.array([0, 0, 1])
            
            for index, scan_line in enumerate(valid_cmd_list):

                if scan_line[0:4] == 'scan':
                    #tempstr=scan_line
                    #print(scan_line)
                    #total_no_point= total_no_point + int(tempstr.split()[-3])
                    print(scan_line)
                    output=self.sika.sika_scansim(command=scan_line, exp=exp)
                    if output is not None:
                        sim_out_str=sim_out_str +"\n\nCommand:\n" + scan_line+ "\n\n" + output.to_string(float_format=lambda x: "{:.4f}".format(x))
            

            
            self.txtSikaScanSimOutput.setPlainText('')
            #print(output.columns.values)
            self.txtSikaScanSimOutput.setPlainText(validation_output+"\n\nHere are the simulated scans:\n"+sim_out_str+"\n\nTotal Number of Data Points: {}".format(total_no_point))
            #output.to_string()
        else:
            print("error")
            
        return
    
    
    def btnSaveContour_clicked(self):
        
        if self.contourCanvas.fig is not None:
            path, ok = QFileDialog.getSaveFileName(self, 'Save Contour Figure', os.getenv('HOME'), 'JPG(*.jpg)')
            if ok:
                self.contourCanvas.fig.savefig(path)
                
        else:
            print("error")
        
        return

    '''    
    def yMinMax_return(self):
        
        minmaxYstr=self.yMinMax.text()
        minmaxValues=minmaxYstr.split(":")
        print(minmaxValues)
        if len(minmaxValues)>1:
            self.contourCanvas.axes.set_ylim(float(minmaxValues[0]),float(minmaxValues[1]))
            #self.contourCanvas.draw_idle()
            print("set y lim done")
        else:
            print("The input in minmax text box is wrong!")

        return
    '''
    
    
    
    def btnPlotContour_clicked(self):
        
        #print(self.combScanNoList.text())
        if self.dataPath == '':
            print('Warning: no data path was choosen.')
            return
        
        scannoString=self.combScanNoList.text()
        
        tempstr="0123456789 +,"
        for eachchar in scannoString:
            if tempstr.find(eachchar)<0:
                errorbox=QMessageBox()
                errorbox.setText("There are some typo in the scan number list! Please check and correct it.")
                errorbox.exec_()
                return
            
        scannoStrList=[substr.strip() for substr in scannoString.split(',')]
        
        scannoList = list([])
        
        for substr in scannoStrList:
            if substr.find('+')>0:
                sublist=list(map(int, substr.split('+')))
                scannoList.append(sublist)
            else:
                scannoList.append(int(substr))
        #print(scannoList)
        
        
        
        motorlist = ['qh', 'qk', 'ql', 'en']
        labellist =  ["QH [rlu]", "QK [rlu]", "QL [rlu]", "E [meV]" ]

        motormapping = dict(zip(motorlist, labellist))
        low           = float(self.contourYmin.text() )
        high          = float(self.contourYmax.text() )
        vminmax       = [int(self.contourImin.text()), int(self.contourImax.text())]
        
        scanaxis1     = self.contourXaxis.text()          #this is the axis change from file to file
        scanaxis1     = scanaxis1.strip()
        scanaxis2     = self.contourYaxis.text()           #this is the scan axis in one data file
        scanaxis2     = scanaxis2.strip()
        plottitle     =  motormapping[scanaxis1] + '-' + motormapping[scanaxis2] + " contour map"

        motorlist = ['qh', 'qk', 'ql', 'en', 'detector', 'monitor']

        dflist=self.taipan.taipan_batch_reduction(self.dataPath, scannoList, motorlist=motorlist)

        all_emin = min(df['en'].min() for df in dflist)
        all_emax = max(df['en'].max() for df in dflist)

        if scanaxis1 in motorlist and scanaxis2 in motorlist:

            if self.radioTaipan.isChecked():
                self.contourCanvas.axes.clear()
                qq,ee,intensity,self.cs=self.taipan.tas_tidy_contour(dflist, vminmax,
                                         x_col=scanaxis1, y_col=scanaxis2, scan_range=[all_emin, all_emax], xlabel=motormapping[scanaxis1],
                                         ylabel=motormapping[scanaxis2], title=plottitle, ax=self.contourCanvas.axes)
    
                self.contourCanvas.draw()

            elif self.radioSika.isChecked():
                self.contourCanvas.axes.clear()
                qq,ee,intensity,self.cs=self.sika.tas_tidy_contour(scanlist=scannoList, vminmax=vminmax,
                                         x_col=scanaxis1, y_col=scanaxis2, scan_range=[all_emin, all_emax], xlabel=motormapping[scanaxis1],
                                         ylabel=motormapping[scanaxis2], title=plottitle, ax=self.contourCanvas.axes)
    
                self.contourCanvas.draw()
            
        else:
            print("Error: Axis names are wrong. Please give correct axis names: qh, qk, ql, en!")
        
        return
                               



def main():     

    # Must be set before QApplication is created
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Optional: avoid rounding of scaling factors (Qt ≥ 5.14)
    """
        if hasattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy"):
        QtCore.QCoreApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )   
    """

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__" :
    main()


