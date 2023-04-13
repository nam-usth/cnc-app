import numpy as np
from numpy.polynomial import hermite as H
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

def API_hermite(points, filename):
    error_id = 0
    
    points.columns = ['x', 'y', 'z']
    
    x = np.array(points['z'].values)
    y = np.array(points['x'].values)
    z = np.array(points['y'].values)
    
    # First 250 points
    x_f250 = x[:250]
    
    total_rows = len(points.index)
    count_last = total_rows - 250
    
    # Last count_last points
    x_l50 = x[-count_last:]
    y_l50 = y[-count_last:]
    z_l50 = z[-count_last:]
    
    x = x[::10]
    y = y[::10]
    z = z[::10]
    
    x_lm = x[::5]
    y_lm = y[::5]
    z_lm = z[::5]
    
    # x_samples = x[::10]
    # y_samples = y[::10]
    # z_samples = z[::10]
    x_samples = x
    y_samples = y
    z_samples = z

    x_ori = x
    print(len(x), len(y), len(z))
    
    #Duplicate points with index 0, 50, 100 so that they pull the line closer to these points
    offset = 0
    for i in range(len(x_ori)):
        if i%5 == 0:
            ex = x[i]
            ey = y[i]
            ez = z[i]
            j = i + offset
            x = np.insert(x,[j, j, j, j, j, j, j, j, j, j],[ex, ex, ex, ex, ex, ex, ex, ex, ex, ex])
            y = np.insert(y,[j, j, j, j, j, j, j, j, j, j],[ey, ey, ey, ey, ey, ey, ey, ey, ey, ey])
            z = np.insert(z,[j, j, j, j, j, j, j, j, j, j],[ez, ez, ez, ez, ez, ez, ez, ez, ez, ez])
    
            offset = offset + 10
            print(i, j)
            #print(ele)
    print(len(x), len(y), len(z))
    
    try:
        #Fit to compute co-efficients (c_y, c_z) of Hermite series
        c_y, stats = H.hermfit(x,y,8,full=True)
        c_z, stats = H.hermfit(x,z,8,full=True)
    except:
        error_id = 1005    
        return error_id
    
    try:
        #Now given x_eval value, we can compute the corresponding y and z values
        #x_eval = np.linspace(min(x),max(x),250)
        x_eval = x_f250
        y_eval = H.hermval(x_eval,c_y) # X-axis
        z_eval = H.hermval(x_eval,c_z) # Y-axis
    except:
        error_id = 1006
        return error_id

    data = {"x": np.concatenate((y_eval, y_l50), axis=0), "y": np.concatenate((z_eval, z_l50), axis=0), "z": np.concatenate((x_eval, x_l50), axis=0)}
    
    outdir = './storage'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    x = time.time()

    new_df = pd.DataFrame(data)
    
    new_df.to_csv(os.path.join(outdir, filename + '_smooth.csv'), index=False, header=False)

    return error_id

# %% Main function

if __name__ == "__main__":
    '''
    csv_path = "D:/Working/KhoanCNC/Nam_work/experiment/edge_01.csv"
    points = pd.read_csv(csv_path)
    API_hermite(points, "edge_01")
    '''
    
    pass