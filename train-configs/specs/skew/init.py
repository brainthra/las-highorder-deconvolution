# from lasxrecon.specgen import *

spec_type = 'skew'

try:
    configs
except NameError:
    configs = {}

# analytical params
mu_min, mu_max = configs.get("mu", {}).get("min", 0.150), configs.get("mu", {}).get("max", 0.750)
sg_min, sg_max = configs.get("sigma", {}).get("min", 0.200), configs.get("sigma", {}).get("max", 0.500)
alpha_min, alpha_max = configs.get("alpha", {}).get("min", -5), configs.get("alpha", {}).get("max", 5)

# number of samples
samples = configs.get("samples", 100_000)


mu_min, mu_max = mu_min * Emax, mu_max * Emax
sg_min, sg_max = sg_min * Emax, sg_max * Emax

print(f"mu in [{mu_min:.3f}, {mu_max:.3f}]")
print(f"sg in [{sg_min:.3f}, {sg_max:.3f}]")
print(f"alpha in [{alpha_min:.3f}, {alpha_max:.3f}]")

# load spectra dataset
np.random.seed(seed)
spectra_set, spectra_params = SkewedGaussianGenerator(TARGET_ENERGIES, FULL_ENERGIES, mu_lim=(mu_min, mu_max), sg_lim=(sg_min, sg_max), alpha_lim=(alpha_min, alpha_max)).generate(samples, random_state=seed, shuffle=True)

param_str = f"{spec_type}--mu-{mu_min:.3f}-{mu_max:.3f}--sg-{sg_min:.3f}-{sg_max:.3f}--alpha-{alpha_min:.3f}-{alpha_max:.3f}"

codestr = 'title_str = f"Test # {idx} | $\\\phi$: {fluxes[idx]:.2e}\\n" + f"$\\\mu$: {spectra_params[\'mu\'][idx]:.3f}, $\\\sigma$: {spectra_params[\'sigma\'][idx]:.3f}, al: {spectra_params[\'alpha\'][idx]:.3f}"'