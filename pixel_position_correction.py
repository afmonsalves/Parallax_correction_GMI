import h5py
from math import asin, sin, cos
import numpy as np

def rotation_angle(path,s='S1'):
    ''' This function develops the rotation angles for the GPM GMI Instrument.
        These rotation angles are the direction in which one would move to reposition
        a given pixel closer to the sub-satellite point. These rotation angles are developed
        considering different characteristics of GMI's orbit, those being whether GPM is in an
        ascending or descending orbit branch and whether GMI is forward- or rearward-facing. This 
        function requires the input file to be of Level 1C quality.

        This function was developed using code provided by Veljko Petkovic and Ryan Gonzalez.

        This function has only been tested on GMI 'Scan1', which includes the first nine channels
        of the instrument. GMI 'Scan2' should work with this function, but care must be taken to 
        ignore 'null pixels'.

        Args: path -> the absolute path for the necessary GMI file
              s -> the scan type ('S1' or 'S2' for GMI) being used

        Returns: an array of size [nscan,npix,5], where nscan is the number of scan lines
        in the file, npix is the number of pixels per scan (221 for GMI low-frequency channels), 
        and the third dimension containing the sub-satellite point latitude and longitude, the 
        pixel latitude and longitude, and the rotation angle, respectively.
    '''
    ## Read GMI File and Identify Necessary Variables ##
    satfl = h5py.File(path,'r') #file reader
    Slats, Slons = satfl[s]['SCstatus']['SClatitude'][:], satfl[s]['SCstatus']['SClongitude'][:] # Sub-satellite Point Positions
    Plats, Plons = satfl[s]['Latitude'][:,:], satfl[s]['Longitude'][:,:] #Pixel Positions
    Sorient = satfl[s]['SCstatus']['SCorientation'] #0 if forward-facing, 180 if rearward-facing

    ## Create Output Array  ##
    pixinfo = np.zeros([Sorient.shape[0],221,5])

    ## Determine if orbit is forward- or rearward-facing ##
    if Sorient[0] == 0.: # all values in Sorient array are the same
        face = 1. #forward-facing modifier
    elif Sorient[0] == 180.:
        face = -1. #rearward-facing modifier
    else: 
        face = None #if in some case the orbit is not forward- or rearward-facing (afterwards it will be considered  as backwards)

    ## Develop Pixel-Wise Rotation Angles ##
    for ns in range(Sorient.shape[0]): #iterate by scan number
        ## Sub-Satellite Position for Scan ##
        Slat, Slon = Slats[ns], Slons[ns]

        if Slat > 65.: #logic statements constrain satellite latitude bounds to the inclination angle (65 degrees)
            Slat = 65.
        elif Slat < -65.:
            Slat = -65.
            
        if (ns == 0): #logic statements determine if satellite is ascending or descending
            branch = 1. #at beginning of orbit, satellite is always ascending
        elif (ns == Sorient.shape[0]-1):
            branch = -1. #at end of orbit, satellite is always descending
        else: #this statement determines branch by differencing the latitudes of the adjacent scans
            latdif = Slats[ns+1]-Slats[ns-1]
            if latdif > 0.:
                branch = 1. #satellite is in ascending branch of orbit (moving northward)
            else:
                branch = -1. # satellite is in descending branch of orbit (moving southward)
        #print(face, branch)

        ############################################################
        #  ROTATION ANGLE CALCULATION 1: Satellite Rotation Angle  #
        # Process: This calculation determines the angle relative  #
        # to due East in which the satellite is travelling. This   #
        # is done by first determining the slope of the satellite  #
        # ground track by scan line as the ratio of the cosine of  #
        # the inclination angle to the cosine of the absolute      #
        # value of the satellite latitude (pos_ratio). The         #
        # absolute value is necessary to prevent the inclusion of  #
        # an unnecessary negative sign. The arcsine of this ratio  #
        # gives the angle associated with this slope, which is     #
        # subtracted from 90 degrees to give the angle relative to #
        # due East. The branch modifier is also used to orient the #
        # angle with respect to the ascending and descending       #
        # branches of the orbit.                                   #
        ############################################################

        pos_ratio = cos(np.deg2rad(65.))/cos(np.deg2rad(abs(Slat)))
        if pos_ratio > 1.: #additional logic to prevent failure for approximation
            pos_ratio = 1.
        elif pos_ratio < -1.:
            pos_ratio = -1.
        track = branch*(90.-np.rad2deg(asin(pos_ratio)))

        for px in range(221): #iterate by pixel number
            ## Pixel Position ##
            Plat, Plon = Plats[ns,px], Plons[ns,px]

            ## Rotation Angle Calculations ##
            ##################################################################################
            #  ROTATION ANGLE CALCULATION 2: Pixel Displacement Angle Relative to Satellite  #
            # Process: First, multiply the pixel number by the GMI scan sector angle (140    #
            # degrees), then take the ratio of this value to one less than the total number  #
            # of pixels to find the given pixel's angular position in the scan. Using one    #
            # less than the total pixel number accounts for Python's zero-based nature.      #
            # Then, adjust this angular position by -70 degrees to get the pixel-relative    #
            # angle. The angle is negative to account for the counterclockwise scanning      #
            # strategy of GMI. For rearward-facing cases, this adjustment is                 # 
            # -70-180=-250 degrees.                                                          #
            ##################################################################################

            if face == 1.: #forward-facing angle
                theta = -70.+(140.*(px/220.))
            else: #rearward-facing angle
                theta = -250.+(140.*(px/220.))

            #########################################################################
            #  ROTATION ANGLE CALCULATION 3: Satellite and Pixel Angle Combination  #
            # Process: The pixel-relative angle (theta) is modified by the          #
            # satellite-relative angle to give that pixel's rotation angle relative #
            # to due East. Subtracting this from 90 degrees results in the pixel    #
            #rotation angle relative to due North.                                  #
            #########################################################################

            px_ang = 90.-track-theta
            #rot_ang = px_ang

            if px_ang > 360.: #these logic statements constrain angles to unit circle (0 to 360 degrees)
                rot_ang = px_ang % 360.
            elif px_ang < 0.:
                rot_ang = (360.+px_ang) % 360.
            else:
                rot_ang = px_ang

            ##########################################################
            #  ROTATION ANGLE CALCULATION 4: Directional Adjustment  #
            # Process: This step returns the direction in which to   #
            # displace a given pixel towards the satellite. This is  #
            # done by adding 180 degrees to rot_ang if it is less    #
            # than 180 degrees or by subtracting 180 degres from     #
            #rot_ang if it is greater than or equal to 180 degrees.  #
            ##########################################################

            if rot_ang < 180.:
                rot_ang += 180.
            else:
                rot_ang -= 180.

            ## Populate Output Array ##
            pixinfo[ns,px,0] = Slat #zeroth elemnt in third dimension is sub-satellite point latitude
            pixinfo[ns,px,1] = Slon #first element in third dimension is sub-satellite point longitude
            pixinfo[ns,px,2] = Plat #second element in third dimension is pixel latitude
            pixinfo[ns,px,3] = Plon #third element in third dimension is pixel longitude
            pixinfo[ns,px,4] = rot_ang #fourth element in third dimension is pixel rotation angle towards satellite
    return pixinfo
