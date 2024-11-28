import numpy as np
from theseusqfls import genplanck
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='Calibrated hyperspectral PL file')
parser.add_argument('--batch_size', type=int, default=2048, help='Number of pixels per batch array')
parser.add_argument('--max_iterations', type=int, default=10, help='Steps taken by Levenberg-Marquardt algorithm')
parser.add_argument('--step_size', type=float, default=0.75, help='Levenberg-Marquardt step size')

args = parser.parse_args()

if args.filename is None:
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()
else:
    filename = args.filename

exp_energy, exp_pl = genplanck.import_hyperspectral(filename)
exp_energy_crop, exp_pl_crop = genplanck.crop_data(exp_energy, exp_pl)
init_guesses = [1.2, 0.025, 1.5, 1.7]
points = [256, 512, 768]
guesses = np.zeros((4, len(points), len(points)))
for i in range(len(points)):
    for j in range(len(points)):
        guesses[:, i, j], __ = curve_fit(genplanck.gen_planck, exp_energy_crop, exp_pl_crop[:, i, j].squeeze(), p0=init_guesses)
guesses = list(np.mean(guesses, axis=(1, 2)))

result, info = genplanck.fit_qfls(exp_energy_crop, exp_pl_crop, guesses=guesses, batch_size=args.batch_size, max_iterations=args.max_iterations, step_size=args.step_size, verbose=False)
with h5py.File(filename, 'a') as f:
    if 'result' in list(f.keys()):
        del f['result']
    f.create_group('result')
    for key in result.keys():
        f['result'][key] = result[key]
