import numpy as np
import re
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf

def find_nearest(array, value):
    '''Function to find the positiond and value of
    the closest pixel to a value
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


#formulas to find the corrected coordinates for the GPROF retrievals algorithns
def getEndpoint(lat1,lon1,bearing,d):
    '''function adapted from: https://stackoverflow.com/questions/18580414/great-circle-destination-formula-for-python
    to find the end point of a line given a starting point, bearing and distance
    inputs: lat1(initial latitude, degrees), 
            lon1 (initial longitude, degrees), 
            bearing (direction from north in clockwise direction, degrees), 
            d (distance in km)
    outputs: lat2 (final latitude, degrees),
             lon2 (final longitude, degrees)
    '''
    import math
    R = 6371                     # Average Radius of the Earth
    brng = math.radians(bearing) #convert degrees to radians
    lat1 = math.radians(lat1)    #Current lat point converted to radians
    lon1 = math.radians(lon1)    #Current long point converted to radians
    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) + math.cos(lat1)*math.sin(d/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return (lat2,lon2)

####### FUNCTIONS FOR ERA 5 PART ON ############

def atof(text):
    '''To check if string to float for every element in list'''
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''To order alphanumeric characters including special characters, e.g.'''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def find_nearest(array, value):
    '''To find in an array, the position and the element nearest to a value'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def chunker(seq, size):
    '''To split the complete list of files by chunks
     inside a for loop'''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def clean_dataset(df):
    '''To help clean the dataframes which rows is
    is being infinite or nan'''
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def get_gaussian(mu, sigma, wide):
    '''Function to create a bidimentional gaussian 
    window, used to create weighted averages of rainfall
    matrix'''
    mvn = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    x,y = tf.cast(tf.linspace(-1,1,wide),tf.float32), tf.cast(tf.linspace(-1,1,wide), tf.float32)
    # meshgrid as a list of [x,y] coordinates
    coords = tf.reshape(tf.stack(tf.meshgrid(x,y),axis=-1),(-1,2))
    gauss = mvn.prob(coords)
    return tf.reshape(gauss, (wide,wide))
    
def sum_str_list(list_str):
    #sum two string inside a list
    sum_str = ''
    for s in list_str:
        sum_str = sum_str + s
    return sum_str
