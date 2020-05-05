# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:19:35 2019

@author: Silvia
"""

import numpy as np

# --- Construction of weighted matrix of connections --- #


def weight_matrix(gdf, Distance_3D, paycheck):

    # Altitude distance in meters
    weight = gdf['Weight'].values
    N = gdf['X'].size
    weight_columns = np.repeat(weight[:, np.newaxis], N, 1)
    weight_rows = np.repeat(weight[np.newaxis, :], N, 0)
    Tot_weight = (weight_columns+weight_rows)/2

    # 3D distance
    weight_matrix = (Distance_3D*Tot_weight)*paycheck/1000
    
    return weight_matrix
