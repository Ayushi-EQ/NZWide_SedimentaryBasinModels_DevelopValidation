# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 09:06:50 2025

@author: ati47
"""

import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler
import os 
from pathlib import Path
# Read the two profiles
def read_profile(file):
    df = pd.read_csv(file, delim_whitespace=True, skiprows=1,
                     names=['Elevation (km)', 'Vp (km/s)', 'Vs (km/s)', 'Rho (g/cm^3)'])
    # Convert depth from km to m (positive downwards)
    df = df.dropna(subset=['Vp (km/s)']) 
    df['Elevation (m)'] = df['Elevation (km)'] * 1000
    return df

out_dir = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Journal papers\Paper 2\Figures\2. Figure Outputs")
profile_Generic  = read_profile('Generic_Paper2Figure.txt')
profile_Nelson  = read_profile('Nelson_Paper2Figure.txt')

# Plot step-like Vs profile
fig, ax = plt.subplots(figsize=(6,8),constrained_layout=True)


ax.step(profile_Generic['Vs (km/s)']*1000, profile_Generic['Elevation (m)'], where='post', color='r', label='Generic', linewidth=2)
ax.step(profile_Nelson['Vs (km/s)']*1000, profile_Nelson['Elevation (m)'], where='post', color='b', label='Nelson/Westport', linewidth=2)

ax.set_xlabel(r'$V_s$ (m/s)',fontsize=16)
ax.set_ylabel('Elevation (m)',fontsize=16)
# ax.set_ylim(0, min(profile1['Elevation (m)'].max(), profile2['Elevation (m)'].max()))
ax.set_ylim(0,3000)
ax.set_xlim(0,3000)
ax.invert_yaxis()  # Depth increases downwards
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax.legend(fontsize=16)
plt.savefig(os.path.join(out_dir,"GenericVs.Nelson.pdf"))
