import concurrent.futures
import numpy as np
import h5py
from scipy.signal import welch
from matplotlib import pyplot as plt
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

    def turbulence(self, z):
        Iu = self.I10 * np.power((z / 10), -self.alpha)
        Lu = 300 * np.power((z / 300), 0.46 + 0.074 * np.log(self.z0))
        sigma_v_u = 1 - 0.22 * np.power(np.cos(np.pi / 2 * z / self.__h()), 4)
        sigma_w_u = 1 - 0.45 * np.power(np.cos(np.pi / 2 * z / self.__h()), 4)
        Iv = Iu * sigma_v_u
        Iw = Iu * sigma_w_u
        Lv = 0.5 * sigma_v_u**3 * Lu
        Lw = 0.5 * sigma_w_u**3 * Lu
        return (Iu, Iv, Iw, Lu, Lv, Lw)

class Scale(object):
    def __init__(self, length_scale, velocity_scale) -> None:
        self.len = length_scale
        self.vel = velocity_scale
        self.freq = velocity_scale / length_scale
        self.time = length_scale / velocity_scale
        self.forc = length_scale**2 * velocity_scale**2
        self.torq = length_scale**3 * velocity_scale**2

def main():
    terr = Terrain(11.1, 0.61, 0.7, 0.22, 0.23, 23.167)
    
    x = np.linspace(0, 2, 21)
    y = np.linspace(-1, 1, 21)
    y = np.delete(y, 10)
    z = np.linspace(0.1, 2, 20)
    z = np.delete(z, 9)
    
    coord = [(x[i], 0, 1) for i in range(21)]
    coord.extend([(0, y[i], 1) for i in range(20)])
    coord.extend([(0, 0, z[i]) for i in range(19)])
    
    plist = [Point(f'p{i}', coord[i], terr.vel(coord[i][2]), terr.turbulence(coord[i][2])) for i in range(60)]
    
    fs = 200
    time = np.arange(0, 10 + 1 / fs, 1 / fs)
    gen = GeneratorFactory.create_generator('NSRFG', {'N': 1000, 'fmax': 100, 'target_spectrum': 'vonKarman'})
    
    fluct = dict()
    spec = dict()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_results = {executor.submit(gen.create_fluct, time, p): p for p in plist}
        for future in concurrent.futures.as_completed(future_results):
            p = future_results[future]
            fluct[p], spec[p] = future.result()

    # Save generated velocity fluctuations to an HDF5 file
    h5_filename = "turbulent_inflow.h5"
    with h5py.File(h5_filename, "w") as h5f:
        for p in plist:
            group = h5f.create_group(p.name)
            group.create_dataset("u", data=fluct[p][0])
            group.create_dataset("v", data=fluct[p][1])
            group.create_dataset("w", data=fluct[p][2])
    
    print(f"Turbulent inflow data saved to {h5_filename}")

if __name__ == '__main__':
    main()
