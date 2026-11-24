import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm

class reservoir_Groningen():
    def __init__(self ):
        pass
    
    def load_EQcatalog(self, catalog_file = "./data/KNMI_CAT_1991-12to2023-01_InsideReservoir.csv"):
        df = pd.read_csv(catalog_file)
        df['RDX'] = 1e-3*df['RDX'] # km
        df['RDY'] = 1e-3*df['RDY'] # km
        return df

    def load_outline(self, pressure_file = './data/ReservoirOutline.npy'):
        OUTLINE = 1e-3 * np.load(pressure_file, allow_pickle=True)
        self.OUTLINE = OUTLINE
        return OUTLINE
    
    def load_thickness(self, thickness_file = './data/ReservoirThickness.npy'): 
        X, Y = np.mgrid[221.250:269.750:0.500, 566.000:615.500:0.500] 
        THICKNESS = np.load(thickness_file, allow_pickle=True)
        self.X = X
        self.Y = Y
        self.THICKNESS = THICKNESS
        return X, Y, THICKNESS
    
    def load_pressure_mesh(self, pressure_file = './data/pressures_list.npy', mesh_file = './data/MESH.npy'):
        PRESSURE = np.load(pressure_file, allow_pickle= True )
        DPRESSURE = PRESSURE - PRESSURE[0,:]
        MESH = np.load(mesh_file, allow_pickle= True )
        
        self.PRESSURE = PRESSURE
        self.DPRESSURE = DPRESSURE
        self.MESH = MESH

        return PRESSURE, DPRESSURE, MESH
    
    def interpolate_grid_DPRESSURE(self):
        time_count = self.DPRESSURE.shape[0]
        mesh_X_count = self.X.shape[0]
        mesh_Y_count = self.X.shape[1]
        
        DPRESSURE_grid_all = np.empty((time_count,  mesh_X_count, mesh_Y_count))


        for i in tqdm(range(time_count), desc = "interpolating scatter PRESSURE into grid", unit = "time step"):
            DPRESSURE_grid_all[i] = griddata(
                points = self.MESH,
                values = self.DPRESSURE[i, :],
                xi = (self.X, self.Y),
                method = 'linear'
            )

        return DPRESSURE_grid_all



class catalog_KNMI:
    def __init__(self, date, magnitude, RDX, RDY):
        self.date = date
        self.magnitude = magnitude
        self.RDX = RDX
        self.RDY = RDY
        
    def fit_coordinates(self, X, Y):
        X_idx = np.argmin(np.abs(X[:, 0] - self.RDX))
        Y_idx = np.argmin(np.abs(Y[0, :] - self.RDY))
        self.X_idx = X_idx
        self.Y_idx = Y_idx

    def get_dpressure(self, DPRESSURE_grid_all):
        DPRESSURE_time = 1956 + 1/12 + np.arange(0,1000,1)/12

        time_idx = np.argmin(np.abs(DPRESSURE_time - self.date))
        if time_idx < 0 or time_idx >= DPRESSURE_grid_all.shape[0]:
            raise ValueError("Time index out of bounds")
        
        dpressure = DPRESSURE_grid_all[time_idx, self.X_idx, self.Y_idx]
        self.dpressure = dpressure
        return dpressure 
    
    def get_dpressure_rate(self, DPRESSURE_grid_all):
        DPRESSURE_time = 1956 + 1/12 + np.arange(0,1000,1)/12

        time_idx = np.argmin(np.abs(DPRESSURE_time - self.date))
        if time_idx <= 0 or time_idx >= DPRESSURE_grid_all.shape[0]-1:
            raise ValueError("Time index out of bounds for rate calculation")

        dpressure_rate = np.gradient(DPRESSURE_grid_all[:, self.X_idx, self.Y_idx])[time_idx] * 12  # Convert to MPa per year
        self.dpressure_rate = dpressure_rate
        return dpressure_rate
    
    def get_cm(self, CM_dict):
        if 'InSARGPS' in CM_dict:
            self.Cm_InSARGPS = CM_dict['InSARGPS'][self.X_idx, self.Y_idx]
        else:
            self.Cm_InSARGPS = None
        
        if 'RAD' in CM_dict:
            self.Cm_RAD = CM_dict['RAD'][self.X_idx, self.Y_idx]
        else:
            self.Cm_RAD = None
        
        if 'OL' in CM_dict:
            self.Cm_OL = CM_dict['OL'][self.X_idx, self.Y_idx]
        else:
            self.Cm_OL = None
        
        return (self.Cm_InSARGPS, self.Cm_RAD, self.Cm_OL)
    
    def get_thickness(self, THICKNESS):
        self.thickness = THICKNESS[self.X_idx, self.Y_idx]
        return self.thickness

    def calculate_MPH3(self, X, Y, DPRESSURE_grid_all, CM_dict, THICKNESS, Cm_TYPE = 'RAD'):
        self.fit_coordinates(X, Y)
        self.get_dpressure(DPRESSURE_grid_all)
        self.get_cm(CM_dict)
        self.get_thickness(THICKNESS)
        if Cm_TYPE == 'InSARGPS':
            self.Cm = self.Cm_InSARGPS
        elif Cm_TYPE == 'RAD':
            self.Cm = self.Cm_RAD
        elif Cm_TYPE == 'OL':
            self.Cm = self.Cm_OL
        elif Cm_TYPE == 'average':
            self.Cm = (self.Cm_InSARGPS + self.Cm_RAD + self.Cm_OL) / 3
        else:
            raise TypeError("Invalid Cm_TYPE. Choose from 'InSARGPS', 'RAD', or 'OL'.")
        
        MPH3 = -self.dpressure * self.thickness**3 / self.Cm
        self.MPH3 = MPH3

        return MPH3

        