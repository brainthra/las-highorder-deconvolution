# from lasxrecon.specgen import *

spec_type = 'boltzmann'

try:
    configs
except NameError:
    configs = {}

# analytical params
T_min, T_max = configs.get("T", {}).get("min", 0.1), configs.get("T", {}).get("max", 0.8)

# number of samples
samples = configs.get("samples", 5_000)


T_min, T_max = T_min * Emax, T_max * Emax

print(f"T in [{T_min:.3f}, {T_max:.3f}]")

# load spectra dataset
np.random.seed(seed)
spectra_set, spectra_params = BoltzmannGenerator(TARGET_ENERGIES, FULL_ENERGIES, temp_lim=(T_min, T_max)).generate(samples, shuffle=True)

param_str = f"{spec_type}--T-{T_min:.3f}-{T_max:.3f}"


codestr = 'title_str = f"Test # {idx} | $\\\phi$: {fluxes[idx]:.2e}\\n" + f"T: {spectra_params[\'temps\'][int(len(ds)*0.75)+idx]*100/Emax:.2f}%"'