import warnings
warnings.filterwarnings("ignore")
###############################################################################
# Code to process GPROF files, GVMRMS files, and ERA5 files together in order
# to create and store tables by pexl with rainfall and atmospheric information
#
# Author: Andres Monsalve - PhD student at University of Texas at El Paso
# Creation Date: 06/21/2022 - mm/dd/yyyy
# Last Update: 07/18/2022 - mm/dd/yyyy
###############################################################################

# to stop printing warnings and info
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
'''0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed'''

#import important libraries
import numpy as np
import pandas as pd
import pygrib
import seaborn as sns
import h5py
import scipy as sp
import xarray as xr
from datetime import datetime
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import rasterio
import gzip
import shutil
from pixel_position_correction import rotation_angle 
import joblib
import swifter
import re
import argparse

from scipy.spatial import cKDTree

#for forward stepwise regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler

#for the gaussian window to make the weighted average
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt

#module with GeoSenSE development
import  functions as FG

#Paths to read from
data_base_path = '/home/geosense/data/' #modify path according to local machine
ERA5_PL_base_path = '/home/geosense/data/ERA5/PL/' #modify path according to local machine
ERA5_SL_base_path = '/home/geosense/data/ERA5/SL/' #modify path according to local machine
figures_base_path = '/home/geosense/OneDrive/Research/figures/variable_ranking/' #modify path according to local machine

#Base path to write on
outputdf_base_path = '/home/geosense/data/OWN_created/' #modify path according to local machine

################################ ARGS READING #################################


parser = argparse.ArgumentParser(prog = 'create_df', description='This program create the dataframes matching the GMI retrievals, GV-MRMS and ERA5, among others sources of data pixelwise, in order to work as input in ML analysis')

parser.add_argument('--region', '-r', default = '',  type=str, nargs='+', help='Name of a region from the polygons_input table')
parser.add_argument('--date_range', '-rf', default = None, type=str, nargs = 2, help='Date range to obtain the dataframe (possible from oct-2019 to oct-2020) in format YYYYMMDD (year month day)')
parser.add_argument('--threshold_rain', '-tr', default = 1,  type=float, help='threshold rain to filter out the data when rainfall either in MRMS or GPROF datasets doesnt fullfil the threshold (1 by default)')
parser.add_argument('--threshold_bias', '-tb', default = 1,  type=float, help='threshold bias to filter out the data when the bias between the two datasets doesnt fullfil the threshold (1 by default)')

args = parser.parse_args()

################################ POLYGONS READING #################################

#input desired region according to names given in the text file
arg_region = args.region[0]

#check if folder of region already exists, if it doesn't, create it
if not os.path.exists(outputdf_base_path + arg_region):
    os.makedirs(outputdf_base_path + arg_region)
outputdf_base_path = outputdf_base_path + arg_region + '/'

if args.date_range == None:
    initial_date = None
    final_date = None
    print('\n-full dates available selected by default')
else:
    initial_date = args.date_range[0]
    final_date = args.date_range[1]
    print('\n-dates selected: ' + initial_date + ' to ' + final_date)
thresh_rain = args.threshold_rain
thresh_bias = args.threshold_bias

# we define the column types and get the names and values from the polygon_input text file
# we are just using the coordinates of the poligons, not actually the polygon objects
dtypes = {'region': 'str', 'state': 'str', 'lat_min': 'float' \
    , 'lat_max': 'float', 'lon_min': 'float', 'lon_max': 'float', 'processing_type': 'float'}
polygons_input = pd.read_csv('polygons_input.txt', index_col=0, header=0, delimiter=' ', dtype=dtypes)
polygon_in_use = polygons_input.loc[polygons_input.region == arg_region,:].to_dict(orient = 'records')[0]
print('\n-Region used: ', polygon_in_use['region'], polygon_in_use['state'])

################################ FZ READING #################################

FZ = joblib.load(data_base_path + 'OWN_created/freezing_level_ERA5_hourly_CONUS_PL')
lat_FZ = joblib.load(data_base_path + 'OWN_created/lat_freezing_level_ERA5_CONUS')
lon_FZ = joblib.load(data_base_path + 'OWN_created/lon_freezing_level_ERA5_CONUS')
time_FZ = joblib.load(data_base_path + 'OWN_created/times_freezing_level_ERA5_CONUS')
print('\n-Freezing level reading, shape:', FZ.shape)


################################ GPROF READING #################################

#to iterate over all available GPROF files (recursive searching as they are in independent folders per day)
if (initial_date != None) & (final_date != None):
    available_dates = pd.date_range(start=initial_date, end=final_date, freq='D')
else: #all the available date range, a full hydrological year in the northern hemisphere
    available_dates = pd.date_range(start='20191001', end='20201031', freq='D')

files_GPROF = []
for av in available_dates: #get the files of GPROFV6 (when updated V7, change path)
    file_date = datetime.strftime(av, '%Y%m%d')
    files_GPROF = files_GPROF + sorted(glob(data_base_path + 'GPROF_V7/{}/{}/*.HDF5'.format(file_date[2:6], file_date[2:])))

print('\n-GPROF files reading from 20{} to 20{}'.format(files_GPROF[0].split('/')[6], files_GPROF[-1].split('/')[6]))

files_inside_polygon = [] #to keep track of the files that contain the polygon
df_rain = pd.DataFrame(columns=['time','lon','lat','GPROF_rain','rot_angle'])
print('\n-Reading of GPROF files starting:', end='')

#check if there is already a df_rain file, if yes, load it
filename_dfrain0 = outputdf_base_path + 'df_' + polygon_in_use['region'] + '_GPROF_V7.0.csv'
if os.path.isfile(filename_dfrain0):
    #to read the dataframe  saved
    print('\n-df_rain0 file found, reading it')
    df_rain = pd.read_csv(filename_dfrain0)
    df_rain['time'] = pd.DatetimeIndex(df_rain['time'])
else:
    print('\n-No df_rain0 file found, creating one')
    for it1, file in enumerate(files_GPROF):
        print(it1, end='|')
        data = h5py.File(file, "r")
        lat = data["S1"]["Latitude"][:,:]
        lon = data["S1"]["Longitude"][:,:]
        #after first reading latitude and longitude of the file, check if inside the polygon
        lon_flat = np.ravel(lon)
        lat_flat = np.ravel(lat)
        pos_lon_flat = np.logical_and(lon_flat > polygon_in_use['lon_min'], lon_flat < polygon_in_use['lon_max'])
        pos_lat_flat = np.logical_and(lat_flat > polygon_in_use['lat_min'], lat_flat < polygon_in_use['lat_max'])
        pos_flat = pos_lon_flat * pos_lat_flat
        if sum(pos_flat) < 2 : #'To check if any pixel inside the polygon for that file'
            continue
        else:
            files_inside_polygon.append(file)
        #after checking that pur region is inside the file
        rain = data["S1"]["surfacePrecipitation"][:,:]
        rain[rain == -9999] = np.nan
        scan_times = pd.DatetimeIndex([pd.to_datetime(str(data['S1']['ScanTime/Year'][i])+'-'
                                                    +str(data['S1']['ScanTime/Month'][i])+'-'
                                                    +str(data['S1']['ScanTime/DayOfMonth'][i])+' '
                                                    +str(data['S1']['ScanTime/Hour'][i])+':'
                                                    +str(data['S1']['ScanTime/Minute'][i])+':'
                                                    +str(data['S1']['ScanTime/Second'][i]), format = '%Y-%m-%d %H:%M:%S') 
                                                    for i in range(len(data['S1']['ScanTime/Year']))])
        data.close()
        #create false array to match index
        times_bidim = np.reshape(np.repeat(scan_times,repeats = lon.shape[1]),(lon.shape[0],lon.shape[1]))
        #flatten arrays to vectorize operations
        times_flat = np.ravel(times_bidim)
        rain_flat = np.ravel(rain)
        # find the angle for every pixel with north
        pixinfo = rotation_angle(file) #(function developed by COLOSTATE)
        pixinfo_flat = np.ravel(pixinfo[:,:,4]) #fourth position in third axis is the angle
        #create a new column with the rotation angle of every pair of coordinates
        # - creating table with found positions
        temp_df = pd.DataFrame({'time': pd.to_datetime(times_flat[pos_flat]),'lon':lon_flat[pos_flat],\
                                    'lat':lat_flat[pos_flat],'GPROF_rain':rain_flat[pos_flat],\
                                    'rot_angle':pixinfo_flat[pos_flat]})
        df_rain = pd.concat([df_rain,temp_df],axis=0)
    df_rain.reset_index(inplace=True, drop=True)
    #save the dataframe
    df_rain.to_csv(filename_dfrain0,index=False)
    print('\n-first df stored on {}'.format(outputdf_base_path))

######################## Height range correction heights################
heights_delta = np.arange(-4000,5000,500)


################################ FZ PROCESSING #################################

# Now got to find the closest FREEZING LEVEL time to each GPROF time
# Get unique times from GPROF to match on FZ
dates_for_FZ = pd.DatetimeIndex(df_rain['time'].unique())
## FZ time position for each cell in scan times from GPROF
# FZ_time_GPROF = [np.argmin(np.abs(time_FZ - scan_times[x])) for x in range(len(scan_times))]
FZ_time_GPROF = time_FZ.get_indexer(dates_for_FZ, method='nearest') 

#assigning times from FZ(ERA5) to every row at the GPROF pixels
df_rain['f_pos_FZ'] = np.nan
df_rain['times_FZ'] = np.nan
df_temporal = pd.DataFrame({'dates_for_FZ': dates_for_FZ, 'files_pos': FZ_time_GPROF, 'times_FZ': time_FZ[FZ_time_GPROF]})
for i, row in df_temporal.iterrows():
    #for the rows where df_rain['time'] is equal to row.dates_for_FZ, equal it to row.files_pos
    df_rain.loc[df_rain['time'] == row.dates_for_FZ, 'f_pos_FZ'] = row.files_pos   
    df_rain.loc[df_rain['time'] == row.dates_for_FZ, 'times_FZ'] = row.times_FZ

print('\n-Processing of Freezing level for parallax correction')

#differences between latitude groups (GPROF vs FZ) simultaneously
temp = df_rain['lat'].values[:,None]  - lat_FZ
idx_lat = np.abs(temp).argmin(axis=1)
df_rain.loc[:, 'lat_FZ'] = np.array([lat_FZ[i] for i in idx_lat])
#differences between latitude groups (GPROF vs FZ) simultaneously
temp = df_rain['lon'].values[:,None]  - lon_FZ
idx_lon = np.abs(temp).argmin(axis=1)
df_rain.loc[:, 'lon_FZ'] = np.array([lon_FZ[i] for i in idx_lon])
#create FZ column
df_rain['FZ'] = np.nan

## For every unique time in times_FZ, select according pixels from FZ
for i, pos in enumerate(df_rain['f_pos_FZ'].unique()):
    positions_temp = df_rain['f_pos_FZ'] == pos
    idx_lat = np.abs(df_rain['lat'].values[positions_temp,None]  - lat_FZ).argmin(axis=1)
    idx_lon = np.abs(df_rain['lon'].values[positions_temp,None]  - lon_FZ).argmin(axis=1)
    #assign FZ value of the closest pixel
    df_rain.loc[positions_temp, 'FZ'] = np.array([FZ[int(pos),i,j] for i,j in zip(idx_lat, idx_lon)])

# create a new columns for every height delta applyed to the FZ
for i, delt in enumerate(heights_delta):
    key = 'FZ_'+str(delt)
    df_rain[key] = df_rain['FZ'] + delt

################################ Parallax Correction #################################
print('\n-Parallax correction')

alpha = 65 #angle in degrees that the satellite use when pointing to the ground (change in case is different)
# find the distance by using the freezing level---- dist = FZ/tan(90-alpha)

#check if there is already a df_rain2 file, if yes, load it
filename_dfrain1 = outputdf_base_path + 'df_' + polygon_in_use['region'] + '_GPROF_V7.1.csv'

if os.path.isfile(filename_dfrain1):
    #to read the dataframe  saved
    print('\n-df_rain1 file found, reading it')

    df_rain = pd.read_csv(filename_dfrain1, index_col=0)
    df_rain['time'] = pd.DatetimeIndex(df_rain['time'])
else:
    print('\n-No df_rain1 file found, creating one')

    temp_vars_parallax = [] #list to fill with the  anmes of the new FZ corrections (in order to drop them later)

    for i, delt in enumerate(heights_delta):

        print(i, end=' ')
        key = 'FZ_'+str(delt)
        temp_vars_parallax.append(key) # the idea is to remove all this after the correction

        df_rain['dist_corr_' + key] = df_rain[key]/np.tan(np.deg2rad(90-alpha))
        
        #make it work in parallel for the full dataframe
        df_rain['lat_corr_' + key] = df_rain.apply(lambda row: np.array(FG.getEndpoint(row['lat'],row['lon'],row['rot_angle'], row['dist_corr_' + key]/1000))[0], axis=1)
        df_rain['lon_corr_' + key] = df_rain.apply(lambda row: np.array(FG.getEndpoint(row['lat'],row['lon'],row['rot_angle'], row['dist_corr_' + key]/1000))[1], axis=1)


    ####################################### GV-MRMS #################################

    print('\n-MRMS matching to every GPROF pixel')

    #to iterate over all available MRMS files (recursive searching as they are in independent folders per day)
    dates_for_MRMS = pd.DatetimeIndex(df_rain['time'].unique())
    ### Show files available for MRMS for the specified uniques dates
    files_MRMS_pre = []
    for file_date in dates_for_MRMS.strftime('%Y%m').unique(): #get the files of GPROFV6 (when updated V7, change path)
        temp = sorted(glob(data_base_path + 'MRMS_CLIP/level2/GPM/{}/{}/PRECIPRATE.GC*.gz'.format(file_date[:4], file_date[4:6])))
        files_MRMS_pre = files_MRMS_pre + temp
    # files_MRMS_pre haves all the possible MRMS files that we will need (and more)
    # create array of dates for MRMS ziped files
    times_MRMS = pd.to_datetime([i.split('.')[2]+' '+i.split('.')[3] for i in files_MRMS_pre])
    # Match Gprof times with available MRMS files (FROM THE CLIPPED ONES, NOT THE FULL DATASET)
    pos_files_on_dates = times_MRMS.get_indexer(dates_for_MRMS, method='nearest') 

    # Then select only the MRMS files that are going to be used
    files_MRMS = np.array(files_MRMS_pre)[np.unique(pos_files_on_dates)]

    # assigning the MRMS file position to every GPROFpixel
    df_rain['f_pos_MRMS'] = np.nan
    df_rain['times_MRMS'] = np.nan
    df_temporal = pd.DataFrame({'dates_for_MRMS': dates_for_MRMS, 'files_pos': list(map(int, pos_files_on_dates)), 'times_MRMS': times_MRMS[pos_files_on_dates]})
    for i, row in df_temporal.iterrows():
        #for the rows where df_rain['time'] is equal to row.dates_for_MRMS, equal it to row.files_pos
        df_rain.loc[df_rain['time'] == row.dates_for_MRMS, 'f_pos_MRMS'] = row.files_pos   
        df_rain.loc[df_rain['time'] == row.dates_for_MRMS, 'times_MRMS'] = row.times_MRMS

    #stablish of the gaussian window to the MRMS data
    window = 8
    gaussian = np.array(FG.get_gaussian([0, 0], [0.4,0.4], window)) #those are the gaussan window characteristics, not sure how to stablish the size, but this works!

    #create the empty columns for the different corrected pixels.
    df_rain['MRMS_rain_u'] = np.nan #uncorrected

    temp_vars_matching = [] #list to fill with the  anmes of the new MRMS matchings (in order to drop later the ones that doesn't work)

    for i, delt in enumerate(heights_delta):
        key = 'MRMS_rain_c'+str(delt)
        df_rain[key] = np.nan #corrected
        temp_vars_matching.append(key) # the idea is to remove all this after the correction


    cont = 0
    ## For every selected MRMS file, read the data and assign it to the GPROF pixel, capturing time delta between files
    ## distance between pixels, MRMS value for the closest pixel
    for i, (pos,file) in enumerate(zip(df_rain['f_pos_MRMS'].unique(),files_MRMS)):
        if i % 5 == 0:
            print('file:', i,end='|')
        try:
            with gzip.open(file, 'rb') as f_in:
                with open(file.split('.gz')[0], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            raster_pr = rasterio.open(file.split('.gz')[0])
            
            if cont==0: #to just access lats and lons of MRMS once, as they don't change
                dtm_pre_arr = raster_pr.read(1, masked=True)
                height = dtm_pre_arr.shape[0]
                width = dtm_pre_arr.shape[1]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(raster_pr.transform, rows, cols)
                lon_MRMS = np.array(xs)[0,:]
                lat_MRMS = np.array(ys)[:,0]
                cont += 1
        except Exception as e:
            print(e)
            df_rain.loc[df_rain['f_pos_MRMS'] == pos, 'MRMS_rain'] = np.nan
            continue
        matrix_pr = raster_pr.read(1, masked=True)
        positions_temp = df_rain['f_pos_MRMS'] == pos

        #------------------find positions of MRMS pixels UNCORRECTED
        temp = df_rain['lat'].values[positions_temp,None]  - lat_MRMS
        idx_lat = np.abs(temp).argmin(axis=1)
        df_rain.loc[positions_temp, 'lat_MRMS_u'] = np.array([lat_MRMS[i] for i in idx_lat])

        temp = df_rain['lon'].values[positions_temp,None]  - lon_MRMS
        idx_lon = np.abs(temp).argmin(axis=1)
        df_rain.loc[positions_temp, 'lon_MRMS_u'] = np.array([lon_MRMS[i] for i in idx_lon])
        #assign MRMS value of the closest pixel
        df_rain.loc[positions_temp, 'MRMS_rain_u'] = np.array([
            np.nansum(gaussian*matrix_pr[int(i-window/2):int(i+window/2),int(j-window/2):int(j+window/2)])\
            /np.nansum(gaussian) for i,j in zip(idx_lat, idx_lon)])

        #------------------find positions of MRMS pixels CORRECTED
        for i, delt in enumerate(heights_delta):
            key_MRMS = 'MRMS_rain_c'+str(delt)
            key_FZ = 'FZ_'+str(delt)

            temp = df_rain['lat_corr_' + key_FZ].values[positions_temp,None]  - lat_MRMS
            idx_lat = np.abs(temp).argmin(axis=1)
            df_rain.loc[positions_temp, 'lat_MRMS_c' + key_FZ] = np.array([lat_MRMS[i] for i in idx_lat])

            temp = df_rain['lon_corr_' + key_FZ].values[positions_temp,None]  - lon_MRMS
            idx_lon = np.abs(temp).argmin(axis=1)
            df_rain.loc[positions_temp, 'lon_MRMS_c' + key_FZ] = np.array([lon_MRMS[i] for i in idx_lon])
            #assign MRMS value of the closest pixel
            try:
                df_rain.loc[positions_temp, key_MRMS] = np.array([
                np.nansum(gaussian*matrix_pr[int(i-window/2):int(i+window/2),int(j-window/2):int(j+window/2)])\
                /np.nansum(gaussian) for i,j in zip(idx_lat, idx_lon)])
            except ValueError: #except to control when the window is on the boundary, then nan
                df_rain.loc[positions_temp, key_MRMS] = np.nan

    #distance between times
    df_rain['times_MRMS'] = pd.to_datetime(df_rain['times_MRMS'])
    df_rain['time_delta'] = df_rain['times_MRMS'] - df_rain['time']
    df_rain['time_delta'] = df_rain['time_delta'].dt.total_seconds()
    #Error
    df_rain['error_u'] = df_rain['GPROF_rain'] - df_rain['MRMS_rain_u']

    for i, delt in enumerate(heights_delta):
        key_MRMS = 'MRMS_rain_c'+str(delt)
        key_error = 'error_c'+str(delt)
        df_rain[key_error] = df_rain['GPROF_rain'] - df_rain[key_MRMS]

    df_rain.to_csv(filename_dfrain1) 
    print('\n-Second version of the dataframe was created and stored on: ' + outputdf_base_path )

######################## Data Quality Control #################################
#at this point we will get rid of unwanted data (e.g. data with no rain, as it is not useful for the analysis)
# and will make some quality checks
#drop nan rows
df_rain = df_rain.dropna(axis=0)
#drop rows where the GPROF precipitation are below 0
df_rain = df_rain.loc[df_rain['GPROF_rain'] > 0]
#drop rows where the absolute value of time_delta is higher than 61 seconds (half the resolution of our coarsest grid; MRMS)
df_rain = df_rain.loc[abs(df_rain['time_delta']) < 61]
#create a pixel distance columns and then eliminate the pixels that are too far away, more than half the diagonal of the coarse grid
df_rain['pix_dist_u'] = ((df_rain['lon_MRMS_u'] - df_rain['lon'])**2 + (df_rain['lat_MRMS_u'] - df_rain['lat'])**2)**0.5

cond_u_any_rain = (df_rain['GPROF_rain']>thresh_rain) | (df_rain['MRMS_rain_u']>thresh_rain)
condicionales_corrected = {}

for i, delt in enumerate(heights_delta):
    key_error = 'error_c'+str(delt)
    cond_c_any_rain = (df_rain['GPROF_rain']>thresh_rain) | (df_rain['MRMS_rain_c'+str(delt)]>thresh_rain)
    condicionales_corrected[delt] = cond_c_any_rain 

# find the correlation and other validation metrics between the GPROF_rain and every error column in df_rain dataframe
corr_correl = []
corr_ticks = []
corr_uncorrected = df_rain['MRMS_rain_u'].loc[cond_u_any_rain].corr(df_rain['GPROF_rain'])

for i, delt in enumerate(heights_delta):
    key_MRMS = 'MRMS_rain_c'+str(delt)

    corr_correl.append(df_rain[key_MRMS].loc[condicionales_corrected[delt]].corr(df_rain['GPROF_rain']))
    if delt > 0 :
        corr_ticks.append('FL +'+str(np.round(delt/1000, 1))+' km')
    elif delt == 0:
        corr_ticks.append('FL ')

    else:
        corr_ticks.append('FL '+str(np.round(delt/1000, 1))+' km')
np.savez(outputdf_base_path + 'Correlation_'+polygon_in_use['region']+'_Parallax.npz', corr_correl=corr_correl, heights_delta=heights_delta, corr_ticks=corr_ticks, corr_uncorrected = corr_uncorrected)


############################ CATEGORICAL COLUMNS CREATION #################################

#find the maximum correlation and the height at which it occurs to create a categrical column
#and save the keys to use
key_error = 'error_c'+str(heights_delta[np.argmax(corr_correl)])
key_cond = heights_delta[np.argmax(corr_correl)]

#Selection of rows of the entire dataframe to be used in the models
df_u = df_rain.loc[cond_u_any_rain]
df_c = df_rain.loc[condicionales_corrected[key_cond]]
print('thresh_bias:', thresh_bias)
#Creation of the categorical variable based on threshold defined previously
for i, row in df_c.iterrows():
    if row[key_error] >= thresh_bias:
        df_c.at[i, 'category'] = 1
    elif row[key_error] < -thresh_bias:
        df_c.at[i, 'category'] = -1
    else:
        df_c.at[i, 'category'] = 0

####################################### ERA5 #################################

print('\n-ERA5 processing starting')
#the new df_rain to use is the df_c (this is for the corrected rows that will be used in the ML)


df_rain = df_c.copy()
###########PL VARIABLES#########################################################
# all the variables have an independent data folder, where each file contains hourly 
# data for all the pixels inside CONUS and all the pressure levels for one day
variables_PL = ['divergence', 'fraction_of_cloud_cover', 'geopotential',
            'ozone_mass_mixing_ratio', 'potential_vorticity', 'relative_humidity',
            'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 'specific_humidity',
            'specific_rain_water_content', 'specific_snow_water_content', 'temperature',
            'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
            'vorticity']
levels = np.array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,
        125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,
        550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,
        925,  950,  975, 1000])
chunk_size = 30 #number of files to read at once
'''chunk reading time for 15 files: ~2.5seg (1.16Gb)'''
print('\n----PL', end='   ')

df_rain['pos_ERA5'] = np.nan
df_rain['times_ERA5'] = np.nan
df_rain['lat_ERA5'] = np.nan
df_rain['lon_ERA5'] = np.nan
for flag, variable in enumerate(variables_PL):
    print('Reading variable: ', variable, end=' | ')
    files = sorted(glob(ERA5_PL_base_path + variable + '/*.nc'), key=FG.natural_keys)
    dates_full_ERAPL = pd.to_datetime([i.split('_')[-1].split('.')[0] for i in files], format='%Y%m%d')

    if flag < 1: #after doing this process for the first file, the flag is set to 1
        ds = xr.open_dataset(files[0])
        lat = ds.latitude.values
        lon = ds.longitude.values
        levels = ds.level.values
        
        pos_lat = np.where((lat >= polygon_in_use['lat_min']) & (lat <= polygon_in_use['lat_max']))[0]
        pos_lon = np.where((lon >= polygon_in_use['lon_min']) & (lon <= polygon_in_use['lon_max']))[0]

        lat = lat[pos_lat]
        lon = lon[pos_lon]
    columns_names = [variable+'_'+str(lev) for lev in levels] #join variable name with pressure level
    
    #create empty columns for every variable and PL
    for col in columns_names:
        df_rain[col] = np.nan
    print('Chunks:', end=' ')
    for file_groups  in FG.chunker(files, chunk_size):
        ds = xr.open_mfdataset(file_groups, combine='by_coords')
        times_ERAPL = pd.DatetimeIndex(ds.time.values)
        var_name = list(ds.data_vars.keys())[0]
        var_values = ds[var_name].values[:,:,pos_lat, :][:,:,:,pos_lon] #from [times, levels, lat, lon] select all the times and all the levels
        print(times_ERAPL[0].date(), '-', times_ERAPL[-1].date(), end=',')
        
        posiciones = (times_ERAPL[0] < df_rain['time']) & (df_rain['time'] < times_ERAPL[-1])
        dates_for_ERAPL = pd.DatetimeIndex(df_rain['time'].loc[posiciones].unique())

        pos_files_on_dates = times_ERAPL.get_indexer(dates_for_ERAPL, method='pad') 

        df_temporal = pd.DataFrame({'dates_for_ERA5': dates_for_ERAPL, \
                                    'files_pos': list(map(int, pos_files_on_dates)), \
                                    'times_ERA5': times_ERAPL[pos_files_on_dates]})
        for i, row in df_temporal.iterrows():
            df_rain.loc[df_rain['time'] == row.dates_for_ERA5, 'pos_ERA5'] = row.files_pos   
            df_rain.loc[df_rain['time'] == row.dates_for_ERA5, 'times_ERA5'] = row.times_ERA5
        
        for i, lev in enumerate(levels):
            for d, pos in enumerate(df_rain['pos_ERA5'].loc[posiciones].unique()):
                var = variable+'_'+str(lev)
                
                indexes = df_rain.loc[posiciones].loc[df_rain['pos_ERA5'].loc[posiciones] == pos].index
                #------------------find positions of ERA5 pixels 
                temp = df_rain['lat'].loc[indexes].values[:, None] - lat
                idx_lat = np.abs(temp).argmin(axis=1)
                df_rain.loc[indexes, 'lat_ERA5'] = np.array([lat[i] for i in idx_lat])

                temp = df_rain['lon'].loc[indexes].values[:, None]  - lon
                idx_lon = np.abs(temp).argmin(axis=1)
                df_rain.loc[indexes, 'lon_ERA5'] = np.array([lon[i] for i in idx_lon])
                #assign MRMS value of the closest pixel

                df_rain.loc[indexes, var] = np.array([var_values[int(pos), i, j, k] for j,k in zip(idx_lat, idx_lon)])
   
df_rain.to_csv(outputdf_base_path + 'df_' + polygon_in_use['region'] + '_GPROF_V7.2.csv') 
print('\n-Third version of the dataframe was created and stored on: ' + outputdf_base_path )


###########SL VARIABLES#########################################################
print('\n----SL', end='   ')

df_rain = pd.read_csv(outputdf_base_path + 'df_' + polygon_in_use['region'] + '_GPROF_V7.2.csv', index_col=0)
df_rain['time'] = pd.DatetimeIndex(df_rain['time'])
# all the variables have an independent data folder, where each file contains hourly 
# data for all the pixels inside CONUS for one day
variables_SL = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'convective_available_potential_energy', 'forecast_albedo', 'soil_type'] #, 'surface_pressure', 'convective_inhibition', 'precipitation_type

'''time, latitude and longitude for pressure level variables  and single levels are the same'''

for flag, variable in enumerate(variables_SL):
    print('Reading variable: ', variable, end=' | ')
    files = sorted(glob(ERA5_SL_base_path + variable + '/*.nc'), key=FG.natural_keys)
    dates_full_ERASL = pd.to_datetime([i.split('_')[-1].split('.')[0] for i in files], format='%Y%m%d')
   
    #create empty columns for every variable
    df_rain[variable] = np.nan
    ds = xr.open_mfdataset(files, combine='by_coords')
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    pos_lat = np.where((lat >= polygon_in_use['lat_min']) & (lat <= polygon_in_use['lat_max']))[0]
    pos_lon = np.where((lon >= polygon_in_use['lon_min']) & (lon <= polygon_in_use['lon_max']))[0]

    lat = lat[pos_lat]
    lon = lon[pos_lon]
    times_ERASL = pd.DatetimeIndex(ds.time.values)
    var_name = list(ds.data_vars.keys())[0]
    var_values = ds[var_name].values[:,pos_lat, :][:,:,pos_lon] #from [times, lat, lon] select all the times 
    if variable == 'soil_type': #because the soil type is a categorical variable
        var_values = np.rint(var_values)
    print(times_ERASL[0].date(), '-', times_ERASL[-1].date(), end=',')
    
    posiciones = (times_ERASL[0] < df_rain['time']) & (df_rain['time'] < times_ERASL[-1])
    dates_for_ERASL = pd.DatetimeIndex(df_rain['time'].loc[posiciones].unique())

    pos_files_on_dates = times_ERASL.get_indexer(dates_for_ERASL, method='pad') 

    for d, pos in enumerate(df_rain['pos_ERA5'].loc[posiciones].unique()):

        indexes = df_rain.loc[posiciones].loc[df_rain['pos_ERA5'].loc[posiciones] == pos].index
        #------------------find positions of ERA5 pixels 
        temp = df_rain['lat'].loc[indexes].values[:, None] - lat
        idx_lat = np.abs(temp).argmin(axis=1)
        # df_rain.loc[indexes, 'lat_ERA5'] = np.array([lat[i] for i in idx_lat])

        temp = df_rain['lon'].loc[indexes].values[:, None]  - lon
        idx_lon = np.abs(temp).argmin(axis=1)
        # df_rain.loc[indexes, 'lon_ERA5'] = np.array([lon[i] for i in idx_lon])

        df_rain.loc[indexes, variable] = np.array([var_values[int(pos), j, k] for j,k in zip(idx_lat, idx_lon)])
   
df_rain.to_csv(outputdf_base_path + 'df_' + polygon_in_use['region'] + '_GPROF_V7.3.csv') 
print('\n-Fourth version of the dataframe was created and stored on: ' + outputdf_base_path )

####################################### TOPOGRAPHY #################################

df_rain = pd.read_csv(outputdf_base_path + 'df_' + polygon_in_use['region'] + '_GPROF_V7.3.csv', index_col=0)

# ===== LECTURA DE DATOS CRUDOS
paths_data_topo = glob(data_base_path + '/NOAA/topo/*TOPO*')
topo = xr.open_dataset(paths_data_topo[0])
lon_topo = topo.x.values
lat_topo = topo.y.values
topo = topo.z.values
topo = np.where(topo < -10, np.nan, topo)

lat_topo = lat_topo[::-1]

topo = topo[::-1,:]
#clip topography to match our polygon
pos_lat = np.where((lat_topo >= polygon_in_use['lat_min']) & (lat_topo <= polygon_in_use['lat_max']))[0]
pos_lon = np.where((lon_topo >= polygon_in_use['lon_min']) & (lon_topo <= polygon_in_use['lon_max']))[0]

lat_topo = lat_topo[pos_lat]
lon_topo = lon_topo[pos_lon]
topo = topo[pos_lat, :][:, pos_lon]

df_rain['Height'] = np.nan
df_rain['lon_TOPO'] = np.nan
df_rain['lat_TOPO'] = np.nan

#-------find positions of ETOPO1 pixels 
temp = df_rain['lat'].values[:, None] - lat_topo
idx_lat = np.abs(temp).argmin(axis=1)
df_rain.loc[:, 'lat_TOPO'] = np.array([lat_topo[i] for i in idx_lat])

temp = df_rain['lon'].values[:, None]  - lon_topo
idx_lon = np.abs(temp).argmin(axis=1)
df_rain.loc[:, 'lon_TOPO'] = np.array([lon_topo[i] for i in idx_lon])

#assign ETOPO1 value of the closest pixel
df_rain.loc[:, 'Height'] = np.array([topo[j, k] for j,k in zip(idx_lat, idx_lon)])

df_rain.to_csv(outputdf_base_path + 'df_' + polygon_in_use['region'] + '_GPROF_V7.4.csv') 
print('\n-Fifth version of the dataframe was created and stored on: ' + outputdf_base_path )


print(df_rain.head(),'\n----WORKED----')





