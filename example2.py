import numpy as np
import h5py
from src.randflow import GeneratorFactory, Point

class Terrain(object):
    __omega = 72.9E-6

    def __init__(self, u_ref, H_ref, z0, alpha, I10, latitude) -> None:
        self.u_ref = u_ref
        self.H_ref = H_ref
        self.z0 = z0
        self.alpha = alpha
        self.I10 = I10
        self.latitude = latitude

    def vel(self, z):
        return self.u_ref * np.power((z / self.H_ref), self.alpha)

    def __h(self):
        u_fric = self.vel(10) / (2.5 * np.log(10 / self.z0))
        f = 2 * Terrain.__omega * np.sin(np.deg2rad(self.latitude))
        return u_fric / (6 * f)

    def turblence(self, z):
        Iu = self.I10 * np.power((z / 10), -self.alpha)
        Lu = 300 * np.power((z / 300), 0.46 + 0.074 * np.log(self.z0))
        sigma_v_u = 1 - 0.22 * np.power(np.cos(np.pi / 2 * z / self.__h()), 4)
        sigma_w_u = 1 - 0.45 * np.power(np.cos(np.pi / 2 * z / self.__h()), 4)
        Iv = Iu * sigma_v_u
        Iw = Iu * sigma_w_u
        Lv = 0.5 * sigma_v_u**3 * Lu
        Lw = 0.5 * sigma_w_u**3 * Lu
        return (Iu, Iv, Iw, Lu, Lv, Lw)

# Assuming a Point object with a constructor (name, coordinates, velocity, turbulence)
class Point:
    def __init__(self, name, coordinates, velocity, turbulence):
        self.name = name
        self.coordinates = coordinates
        self.velocity = [velocity, velocity, velocity]  # Create a list with three identical components
        self.turbulence = turbulence



# Assuming Terrain and Point classes are defined as before

def generate_openfoam_data(plist, time, output_dir):
    """
    Generate turbulent inlet data for OpenFOAM from generated fluctuations.

    Args:
        plist: List of Point objects containing turbulence data.
        time: Time array for the simulation.
        output_dir: Directory to save the OpenFOAM data.
    """
    # Create the output directory if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create HDF5 file to store the data
    with h5py.File(f"{output_dir}/turbulent_inflow.h5", 'w') as f:
        # Store time data
        f.create_dataset("time", data=time)

        # For each point in the list, save the velocity and turbulence components
        for i, p in enumerate(plist):
            group = f.create_group(f"point_{i}")
            group.create_dataset("u", data=p.velocity[0])  # Velocity component u
            group.create_dataset("v", data=p.velocity[1])  # Velocity component v
            group.create_dataset("w", data=p.velocity[2])  # Velocity component w
            group.create_dataset("k", data=p.turbulence[0])  # Turbulence k
            group.create_dataset("epsilon", data=p.turbulence[1])  # Turbulence epsilon

def main():
    terr = Terrain(11.1, 0.61, 0.7, 0.22, 0.23, 23.167)

    # Define coordinates for points
    x = np.linspace(0, 2, 21)
    y = np.linspace(-1, 1, 21)
    y = np.delete(y, 10)
    z = np.linspace(0.1, 2, 20)
    z = np.delete(z, 9)
    
    coord = [(x[i], 0, 1) for i in range(21)]
    coord.extend([(0, y[i], 1) for i in range(20)])
    coord.extend([(0, 0, z[i]) for i in range(19)])

    # Create Point instances for each coordinate
    plist = [Point('p' + str(i), coord[i], terr.vel(coord[i][2]), terr.turblence(coord[i][2])) for i in range(len(coord))]

    # Time array for the simulation (just an example)
    fs = 200  # Sampling frequency (Hz)
    time = np.arange(0, 10 + 1 / fs, 1 / fs)  # Time array

    # Generate the turbulent data for OpenFOAM
    generate_openfoam_data(plist, time, "inlet_data")

if __name__ == '__main__':
    main()
