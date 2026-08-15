# from lasxrecon.specgen import *

spec_type = 'shighorder'

try:
    configs
except NameError:
    configs = {}

# analytical params
T_min, T_max = configs.get("T", {}).get("min", 0.1), configs.get("T", {}).get("max", 0.8)
mu_min, mu_max = configs.get("mu", {}).get("min", 0.150), configs.get("mu", {}).get("max", 0.750)
sg_min, sg_max = configs.get("sigma", {}).get("min", 0.200), configs.get("sigma", {}).get("max", 0.500)
alpha_min, alpha_max = configs.get("alpha", {}).get("min", -5), configs.get("alpha", {}).get("max", 5)

# weighting amplitudes
bamp_min, bamp_max = configs.get("boltz_amp", {}).get("min", 0.3), configs.get("boltz_amp", {}).get("max", 1.0)
gamp_min, gamp_max = configs.get("gauss_amp", {}).get("min", 0.3), configs.get("gauss_amp", {}).get("max", 1.0)

# sampling limits
n_gauss = configs.get("n_gauss", 2)
n_boltz = configs.get("n_boltz", 4)
max_comps = configs.get("max_comps", 10)

# number of samples
samples = configs.get("samples", 500_000)


temp_min, temp_max = T_min * Emax, T_max * Emax
mu_min, mu_max = mu_min * Emax, mu_max * Emax
sg_min, sg_max = sg_min * Emax, sg_max * Emax

print(f"T in [{temp_min:.3f}, {temp_max:.3f}]")
print(f"mu in [{mu_min:.3f}, {mu_max:.3f}]")
print(f"sg in [{sg_min:.3f}, {sg_max:.3f}]")
print(f"alpha in [{alpha_min:.3f}, {alpha_max:.3f}]")
print(f"boltz_amp in [{bamp_min:.3f}, {bamp_max:.3f}]")
print(f"gauss_amp in [{gamp_min:.3f}, {gamp_max:.3f}]") 
print(f"max n_boltz: {n_boltz}, max n_gauss: {n_gauss}, max_comps: {max_comps}")

np.random.seed(seed)
spectra_set, spectra_params = CompositeGenerator(TARGET_ENERGIES, FULL_ENERGIES, temp_lim=[temp_min, temp_max], mu_lim=(mu_min, mu_max), sg_lim=(sg_min, sg_max), alpha_lim=[alpha_min, alpha_max], boltz_amp_lim=(bamp_min, bamp_max), gauss_amp_lim=(gamp_min, gamp_max)).generate(samples, max_boltz=n_boltz, max_gauss=n_gauss, max_comps=max_comps, random_state=seed)

param_str = f"{spec_type}--T-{temp_min:.3f}-{temp_max:.3f}--mu-{mu_min:.3f}-{mu_max:.3f}--sg-{sg_min:.3f}-{sg_max:.3f}--al-{alpha_min:.3f}-{alpha_max:.3f}--bamp-{bamp_min:.3f}-{bamp_max:.3f}--gamp-{gamp_min:.3f}-{gamp_max:.3f}"


codestr = 'title_str = f"Test # {idx} | $\\\phi$: {fluxes[idx]:.2e}\\n" + f"n_gauss: {spectra_params[idx][\'n_gauss\']}, n_boltz: {spectra_params[idx][\'n_boltz\']}"'