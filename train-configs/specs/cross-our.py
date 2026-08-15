T_min, T_max = 0.1, 0.8
mu_min, mu_max = 0.150, 0.750
sg_min, sg_max = 0.200, 0.500
alpha_min, alpha_max = -5, 5
n_gauss = 2
n_boltz = 4
bamp_min, bamp_max = 0.3, 1.0
gamp_min, gamp_max = 0.3, 1.0

T_min, T_max = T_min * Emax, T_max * Emax
mu_min, mu_max = mu_min * Emax, mu_max * Emax
sg_min, sg_max = sg_min * Emax, sg_max * Emax

spectra_sets, spectra_paramss = {}, {}

# load spectra dataset
if "boltzmann" in cross_types:
     test_spec_type = 'boltzmann'
     np.random.seed(seed)
     spectra_sets[test_spec_type], spectra_paramss[test_spec_type] = BoltzmannGenerator(TARGET_ENERGIES, FULL_ENERGIES, temp_lim=(T_min, T_max)).generate(5_000, shuffle=True, random_state=seed)

# load spectra dataset
if "gaussian" in cross_types:
    test_spec_type = "gaussian"
    np.random.seed(seed)
    spectra_sets[test_spec_type], spectra_paramss[test_spec_type] = GaussianGenerator(TARGET_ENERGIES, FULL_ENERGIES, mu_lim=(mu_min, mu_max), sg_lim=(sg_min, sg_max)).generate(25_000, shuffle=True, random_state=seed)

# load spectra dataset
if "skew" in cross_types:
    test_spec_type = 'skew'
    np.random.seed(seed)
    spectra_sets[test_spec_type], spectra_paramss[test_spec_type] = SkewedGaussianGenerator(TARGET_ENERGIES, FULL_ENERGIES, mu_lim=(mu_min, mu_max), sg_lim=(sg_min, sg_max), alpha_lim=(alpha_min, alpha_max)).generate(100_000, random_state=seed, shuffle=True)

if "shighorder" in cross_types:
    test_spec_type = 'shighorder'
    np.random.seed(seed)
    spectra_sets[test_spec_type], spectra_paramss[test_spec_type] = CompositeGenerator(TARGET_ENERGIES, FULL_ENERGIES, temp_lim=[T_min, T_max], mu_lim=(mu_min, mu_max), sg_lim=(sg_min, sg_max), alpha_lim=[alpha_min, alpha_max], boltz_amp_lim=(bamp_min, bamp_max), gauss_amp_lim=(gamp_min, gamp_max)).generate(500_000, max_boltz=n_boltz, max_gauss=n_gauss, max_comps=11, random_state=seed)

for test_spec_type in spectra_sets.keys():
    test_dss = {}
    test_dss[test_spec_type] = LASDataset(spectra_sets[test_spec_type], design, sample_idxs=sample_idxs)
    _, _, test_dss[test_spec_type] = test_dss[test_spec_type].split([0.6, 0.15, 0.25], shuffle=False)

    mutator = Mutator(test_digitiser, (optimal_flux/100, optimal_flux), noiserange, "linear", "linear")

    np.random.seed(seed)
    modded_ms, fluxes = mutator.prepare(test_dss[test_spec_type].measureds, (optimal_flux/100, optimal_flux), noiserange, return_flux=True)
    modded_ss = test_dss[test_spec_type].spectra * fluxes[:, np.newaxis]

    recon_infos = [reconstructor.inference(modded_ms[idx], method="opt", return_info=True, debug=False)[1] for idx in range(len(test_dss[test_spec_type]))]
    recon_preds = [info.spec_tar for info in recon_infos]
    recon_ms = [reconstructor.spectrometer(info.spec_full) for info in recon_infos]
    recon_fluxes = np.array([info.flux for info in recon_infos])
    recon_maes = mae(recon_preds, (modded_ss)[:, :target_bins])/fluxes
    recon_djss = shapeErrors(recon_preds, (modded_ss)[:, :target_bins]).target
    recon_fae = rae(recon_fluxes, fluxes)
    recon_mrs = modded_ms - np.array(recon_ms)
    recon_mrp = recon_mrs / modded_ms

    min_noise, max_noise = noiserange

    Bmin, Bmax, Bstep = test_digitiser.visible, test_digitiser.saturation, test_digitiser.step

    np.save(f'{path}/res/djs--B-{Bmin}-{Bmax}-{Bstep}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}--{test_spec_type}.npy', recon_djss)
    np.save(f'{path}/res/mae--B-{Bmin}-{Bmax}-{Bstep}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}--{test_spec_type}.npy', recon_maes)
    np.save(f'{path}/res/fae--B-{Bmin}-{Bmax}-{Bstep}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}--{test_spec_type}.npy', recon_fae)
    np.save(f'{path}/res/flx--B-{Bmin}-{Bmax}-{Bstep}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}--{test_spec_type}.npy', recon_fluxes)
    np.save(f'{path}/res/mrs--B-{Bmin}-{Bmax}-{Bstep}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}--{test_spec_type}.npy', recon_mrs)
    np.save(f'{path}/res/mrp--B-{Bmin}-{Bmax}-{Bstep}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}--{test_spec_type}.npy', recon_mrp)

