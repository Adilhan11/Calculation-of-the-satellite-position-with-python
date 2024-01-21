import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
import math
from datetime import datetime, timedelta
from astropy.coordinates import Angle

"""
This class represents an elementary rotation matrix.

You can specity axis as 1, 2 and 3. And also angle by degrees.

The associated matrix can be obtained by the R attribute

Ex: rot = RotationMatrix(1, 30)
    R = rot.R # gives you the rotation matrix.

"""

class RotationMatrix(object):
    def __init__(self, ax=1, angle=0):
        self.ax = ax
        self.angle=angle
        self.a_rad = np.radians(angle) # for radians.
        self._construct() # construct the matrix 
    
    """
    Construct an elementary rotation matrix from axes and angle
    for x axis, ax = 1
    for y axis, ax = 2
    for z axis, ax = 3
    angle in degrees, internally it uses a_rad
    """
    def _construct(self):
        if (self.ax == 1):# x axis
            self.R = np.array([
                [      1,            0,         0          ],
                [0, np.cos(self.a_rad),  np.sin(self.a_rad)],
                [0, -np.sin(self.a_rad), np.cos(self.a_rad)],
            ])
        elif (self.ax == 2): # y axis
            self.R = np.array([
                [np.cos(self.a_rad), 0 , -np.sin(self.a_rad)],
                [      0,            1,         0          ],
                [np.sin(self.a_rad), 0 , np.cos(self.a_rad)],
            ])
        elif (self.ax == 3): #z axis
            self.R = np.array([
                [np.cos(self.a_rad),  np.sin(self.a_rad), 0],
                [-np.sin(self.a_rad), np.cos(self.a_rad), 0],
                [      0,            0,         1          ],

            ])
        else:
            raise ValueError("Axis %d is not known!" % (self.ax))


"""
Class that holds satellite state (pos and vel at a given time).
"""
class SatelliteState(object):
    def __init__(self, t, pos, vel):
        self.t = t
        self.pos = pos
        self.vel = vel
        self.GAST = t.sidereal_time("apparent",longitude=0)
        self.R_GAST = RotationMatrix(3,self.GAST.degree)
    
    """
    Return position in ECEF
    """
    def ecef(self):
        return np.dot(self.R_GAST.R,self.pos), np.dot(self.R_GAST.R,self.vel)
    
    ## convert ecef position to lat lon and height
    def lat_lon_h(self):
        pos, vel = self.ecef()
        p=EarthLocation.from_geocentric(pos[0], pos[1], pos[2], unit=u.m)
        return p.lat.deg, p.lon.deg, p.height.value
    
    """
    pretty print satellite state
    """
    def __str__(self):
        params = [str(self.t)]
        params.extend(list(self.pos))
        params.extend(list(self.vel))

        return "%s : p=[%10.4f,%10.4f,%10.4f], v=[%10.4f,%10.4f,%10.4f]" %(tuple(params))
            

        
"""
This class calculates satellite state given keplerian elements and time
"""
class KeplerianEphemeris(object):
    
    GM = 3.986004418e14 # gravitational parameter of Earth
    """
    Initialize Keplerian Orbit Elements
    a: Semi Major axis
    e: eccentricity
    I: Inclination
    RAAN: Right Ascension of the Ascending node
    w: argument of perigee
    T0: perigee passage time, astropy time object in UTC
    """
    
    def __init__(self, a, e, I, RAAN, w, T0):
        self.a = a
        self.e = e
        self.I = I
        self.RAAN = RAAN
        self.w = w
        #Rotation matrix for inclination
        self.R_I = RotationMatrix(1,-self.I)
        #Rotation matrix for Right Ascension of the Ascending Node
        self.R_RAAN = RotationMatrix(3,-self.RAAN)
        #Argument of perigee
        self.R_w = RotationMatrix(3,-self.w)
        # Perigee passage time
        self.T0 = T0
        """
        Rotation matrix from orbit to ECI
        
        R_o_to_ECI = R3_(-RAAN)R_1(-I)R_3(-w)
        """
        self.R_o_to_ECI = np.dot(self.R_RAAN.R, np.dot(self.R_I.R, self.R_w.R))
        
        #orbital period
        self.T = 2*np.pi*np.sqrt(self.a**3/KeplerianEphemeris.GM)
        #mean angular velocity
        self.n = 2*np.pi/self.T        
    
    
    def get_state(self, t):

        #mean anomaly
        M = self.n*(t - self.T0).sec
        
        #initialize Eccentric Anomaly
        E_i = M
        
        # iteration to convert mean anomaly to eccentric anomaly
        for i in range(4):
            E_ipp = M + self.e*np.sin(E_i)
            
            #Early stop if we are not improving E_ipp
            if (np.abs(E_ipp-E_i)<1e-10):
                E_i = E_ipp
                break
            ## next with newly calculated E_ipp
            E_i=E_ipp
            
        #E_i now contains the eccentric anomaly
        
        # Satellite Position in orbital plane
        pos = np.array([
            self.a*(np.cos(E_i) - self.e),
            self.a*np.sqrt(1-self.e**2)*np.sin(E_i),
            0
        ])

        # Satellite velocity in orbital plane
        vel = (self.n*self.a)/(1-self.e*np.cos(E_i))*np.array([
            -np.sin(E_i),
            np.sqrt(1-self.e**2)*np.cos(E_i),
            0
        ])

        pos = np.dot(self.R_o_to_ECI, pos)
        vel = np.dot(self.R_o_to_ECI, vel)

        return SatelliteState(t, pos, vel)
    
#Constants
GM = 3.986004418e14  # Gravitational constant * mass of Earth
a = 29599.8 * 1e3  # Semi-major axis in meters
e = 0.002  # Eccentricity
i = 60  # in degrees
w = 10  # in degrees
RAAN = Angle("3h03m40s").degree  # convert to degrees or RAAN = 3*15 + 3/60*15 + 40/(3600)*15
T0 = datetime(2023, 12, 18, 18, 0, 0)  # Perigee passage time
t = datetime(2023, 12, 18, 20, 0, 0)  # Time at which to calculate the satellite position

ke = KeplerianEphemeris(a, e, i, RAAN, w, T0)

#Calculate mean angular velocity (n)
n = ke.n

#Calculate mean anomaly (M)
delta_t = (t - T0).total_seconds()  # Time difference in seconds!
M = n * delta_t

#Iteratively calculate eccentric anomaly (E)
E = M  # Initial guess
while True:
    E_next = M + e * math.sin(E)
    if abs(E_next - E) < 1e-6:  # Convergence criterion
        E = E_next
        break
    E = E_next

print(f"Mean angular velocity (n): {n} rad/s")
print(f"Mean anomaly (M): {M} rad")
print(f"Eccentric anomaly (E): {E} rad")


# Compute ECI Coordinates

from astropy.time import Time

# Define the orbital elements
a = 29599.8 * 1e3  # convert to meters
e = 0.002
i = 60  # in degrees
RAAN = Angle("3h03m40s").degree  # convert to degrees or RAAN = 3*15 + 3/60*15 + 40/(3600)*15

w = 10  # in degrees
T0 = Time("2023-12-18T18:00:00", scale="utc")  # perigee passage time

# Define the time at which to calculate the satellite position
t = Time("2023-12-18T20:00:00", scale="utc")

# Create a KeplerianEphemeris object
ke = KeplerianEphemeris(a, e, i, RAAN, w, T0)

# Get the satellite state at the specified time
state = ke.get_state(t)

# Print the satellite state
print("ECI Coordinates in meters", state)


# Convert ECI to ECEF

import numpy as np

def R3(theta):
    """Rotation matrix for a rotation around the z-axis."""
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# Assume r_ECI_sat is the position of the satellite in ECI coordinates
# and GAST is the Greenwich Apparent Sidereal Time in radians
r_ECEF_sat = np.dot(R3(state.GAST), state.pos)
print("ECEF Coordinates", r_ECEF_sat)

# You can also call .ecef() on the SatelliteState object to get the ECEF coordinates
r_ECEF_sat, v_ECEF_sat = state.ecef()
print("ECEF Coordinates", r_ECEF_sat)


#  calculate ground track.
import astropy.units as u

import cartopy.crs as ccrs
RAAN = Angle("3h03m40s").degree  # convert to degrees or RAAN = 3*15 + 3/60*15 + 40/(3600)*15

T0 = Time('2023-12-18T18:00', format='isot', scale='utc')
coe = KeplerianEphemeris(29599.8*1000, 0.002, 60.0, RAAN, 10, T0)

lons = []
lats = []


#Loop for 3000 minutes, 

for s in range(3000):
    # time is s minutes from perigee time
    t = T0 + TimeDelta(s*60, format='sec')
    lat,lon,h = coe.get_state(t).lat_lon_h()
    lons.append(lon)
    lats.append(lat)

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1, 1, 1,projection=ccrs.PlateCarree())
ax.scatter(np.array(lons), np.array(lats))
ax.stock_img()
ax.coastlines()
ax.gridlines()
plt.show()



# Calculate azimuth and elevation of a satellite from a given user ocation
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

lat = 39.5
lon = 32.5
height = 1000
T0 = Time("2023-12-18T18:00:00", scale="utc")  # perigee passage time

coe = KeplerianEphemeris(29599.8*1000, 0.002, 60 , RAAN, 10, T0)

yourlocation = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height*u.m)


state = coe.get_state(T0)
pos, vel = state.ecef()

sat_coord = SkyCoord(x=pos[0], y=pos[1], z=pos[2], frame='icrs',representation_type='cartesian', obstime=t)


altaz = sat_coord.transform_to(AltAz(obstime=t,location=yourlocation))
print(f"Satellite Altitude at T0 = {altaz.alt.deg}, Satellite Azimuth at T0 = {altaz.az.deg}")


# You can plot the azimuth and altitude over time to see how the satellite moves.

# Initialize lists to store azimuth and altitude values
azimuths = []
altitudes = []
times = []

for mi in range(0, 120, 10):
    # Increment the time by 10 minutes
    t = T0 + TimeDelta(mi*60, format='sec')
    
    
    # Get the satellite state and position
    state = coe.get_state(t)
    pos, vel = state.ecef()
    
    # Calculate the satellite coordinates
    sat_coord = SkyCoord(x=pos[0], y=pos[1], z=pos[2], frame='icrs', representation_type='cartesian', obstime=t)
    
    # Transform to AltAz
    altaz = sat_coord.transform_to(AltAz(obstime=t, location=yourlocation))
    
    # Append the azimuth and altitude to the lists
    azimuths.append(altaz.az.deg)
    altitudes.append(altaz.alt.deg)
    times.append(t)


times_datetime = [time.datetime for time in times]

# Plot the azimuth and altitude over time
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(times_datetime, azimuths)
plt.title('Azimuth over time')
plt.xlabel('Time')
plt.ylabel('Azimuth (degrees)')
plt.xticks(rotation=45)


plt.subplot(1, 2, 2)
plt.plot(times_datetime, altitudes)
plt.title('Altitude over time')
plt.xlabel('Time')
plt.ylabel('Altitude (degrees)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


T0 = Time('2023-12-18T18:00', format='isot', scale='utc')
RAAN = Angle('3h03m40s').degree
coe = KeplerianEphemeris(29599.8*1000, 0.002, 60 , RAAN, 10, T0)


state = coe.get_state(T0)

# Calculate the velocity magnitude
v = np.linalg.norm(state.vel)


# Calculate the energies
KE = v**2/2*1e-6
PE = -KeplerianEphemeris.GM/np.sqrt(np.sum(state.pos**2))*1e-6



print ("Initial KE, PE and TE")
print ("KE = %f MJ" % (KE))
print ("PE = %f MJ" % (PE))
print ("Total Energy = %f MJ" % (KE+PE))


# Initialize lists to store energy values
KEs = []
PEs = []
total_energies = []
times = []

# Loop over 2 hours (120 minutes), incrementing by 10 minutes each time
for minute in range(0, 120, 10):
    # Increment the time by 10 minutes
    t = Time('2023-12-18T18:00', format='isot', scale='utc') + TimeDelta(minute*60, format='sec')
    
    # Get the satellite state
    state = coe.get_state(t)
    
    # Calculate the velocity magnitude
    v = np.linalg.norm(state.vel)
    
    # Calculate the energies
    KE = v**2/2*1e-6
    PE = -KeplerianEphemeris.GM/np.sqrt(np.sum(state.pos**2))*1e-6
    total_energy = KE + PE
    
    # Append the energies to the lists
    KEs.append(KE)
    PEs.append(PE)
    total_energies.append(total_energy)
    times.append(t.datetime)  # Convert to datetime for plotting


# Plot the energies over time
plt.figure(figsize=(15, 5))



# Plot KE
plt.subplot(1, 3, 1)
plt.plot(times, KEs, label='KE', color='blue')
plt.title('Kinetic Energy over time')
plt.xlabel('Time')
plt.ylabel('Energy (MJ)')
plt.legend()
plt.xticks(rotation=45)


# Plot PE
plt.subplot(1, 3, 2)
plt.plot(times, PEs, label='PE', color='green')
plt.title('Potential Energy over time')
plt.xlabel('Time')
plt.ylabel('Energy (MJ)')
plt.legend()
plt.xticks(rotation=45)


# Plot Total Energy
plt.subplot(1, 3, 3)
plt.plot(times, total_energies, label='Total Energy', color='red')
plt.title('Total Energy over time')
plt.xlabel('Time')
plt.ylabel('Energy (MJ)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Cartesian to Keplerian Calculations example

print(state.pos)
print(state.vel)



pos_normalized = state.pos/np.sqrt(np.sum(state.pos**2))
vel_normalized = state.vel/np.sqrt(np.sum(state.vel**2))


print(pos_normalized)
print(vel_normalized)


h = np.cross(state.pos, state.vel)
print ("Angular Momentum vector direction", h)



RAAN = np.degrees(np.arctan2(h[0],-h[1]))

print ("RAAN of the orbit ", RAAN, " True RRAN from orbit is ", Angle('3h03m40s').deg)
I = np.degrees(np.arctan2(  np.sqrt(h[0]**2 + h[1]**2) , h[2]))

print ("Inclination of the orbit ", I)

R1_I = RotationMatrix(1, I).R
R3_RAAN = RotationMatrix(3, RAAN).R


# poisition of the satellite in its orbital plane
p  = np.dot(R1_I, np.dot(R3_RAAN, state.pos))

print (p)


u = np.arctan2(p[1], p[0]) # Sum of argument of perigee and true anomaly

# v^2/2 - GM/r
epsilon = np.sum(state.vel**2)/2 - KeplerianEphemeris.GM/np.sqrt(np.sum(state.pos**2))


# epsilon = -GM/2a

a = -KeplerianEphemeris.GM/(2*epsilon)

print ("Semi major axis is  ", a/1000)


print ("Eccentricity is ", np.sqrt(1 - np.sum(h**2)/(KeplerianEphemeris.GM*a)))



# Cartesian to Keplerian Calculations example

print(state.pos)
print(state.vel)



pos_normalized = state.pos/np.sqrt(np.sum(state.pos**2))
vel_normalized = state.vel/np.sqrt(np.sum(state.vel**2))


print(pos_normalized)
print(vel_normalized)


h = np.cross(state.pos, state.vel)
print ("Angular Momentum vector direction", h)



RAAN = np.degrees(np.arctan2(h[0],-h[1]))

print ("RAAN of the orbit ", RAAN, " True RRAN from orbit is ", Angle('3h03m40s').deg)
I = np.degrees(np.arctan2(  np.sqrt(h[0]**2 + h[1]**2) , h[2]))

print ("Inclination of the orbit ", I)

R1_I = RotationMatrix(1, I).R
R3_RAAN = RotationMatrix(3, RAAN).R


# poisition of the satellite in its orbital plane
p  = np.dot(R1_I, np.dot(R3_RAAN, state.pos))

print (p)

u = np.arctan2(p[1], p[0]) # Sum of argument of perigee and true anomaly

# v^2/2 - GM/r
epsilon = np.sum(state.vel**2)/2 - KeplerianEphemeris.GM/np.sqrt(np.sum(state.pos**2))

# epsilon = -GM/2a

a = -KeplerianEphemeris.GM/(2*epsilon)

print ("Semi major axis is  ", a/1000)

print ("Eccentricity is ", np.sqrt(1 - np.sum(h**2)/(KeplerianEphemeris.GM*a)))