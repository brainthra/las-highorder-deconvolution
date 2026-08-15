from lasim.spectra import gaussian_peak
dummy_scint = Scintillator('C', 1.6, 1)

layer_params = [
    [0.045, 0.1, 2, 2500, 15.0, 6.0, 25_000],
    [0.1, 0.2, 5.0, 2500, 15.0, 6.0, 24_000],
    [0.2, 0.2, 5.0, 2500, 15.0, 6.0, 23_000],
    [0.3, 0.2, 5.0, 2500, 15.0, 6.0, 22_000],
    [0.4, 0.2, 5.0, 2500, 15.0, 6.0, 20_000],
    [0.5, 0.2, 5.0, 2500, 15.0, 6.0, 18_000],
    [0.6, 0.2, 5.0, 2500, 15.0, 6.0, 16_000],
    [0.7, 0.2, 5.0, 2500, 15.0, 6.0, 14_000],
    [0.8, 0.2, 5.0, 2500, 15.0, 6.0, 12_000],
    [0.9, 0.2, 5.0, 2500, 15.0, 6.0, 10_000],
]

def find_index_from_back(arr1, arr2, diff=10):
    length = min(len(arr1), len(arr2))

    for i in range(length - 1, -1, -1):
        if abs(arr1[i] - arr2[i]) <= diff:
            return i
    return -1

def setrmcm(design, layer_params):
    rm = []
    for params in layer_params:    
        sf = skewed_gaussian_peak(FULL_ENERGIES, params[0], params[1], params[2], params[3])
        sb = gaussian_peak(FULL_ENERGIES, params[4], params[5], params[6])

        idx = find_index_from_back(sf, sb, 100)
        s = np.concatenate([sf[:idx], sb[idx:]])/40_000
        for idx in range(len(s)):
            if s[idx] > FULL_ENERGIES[idx]:
                s[idx] = FULL_ENERGIES[idx]
        rm.append(s)
    design.response_matrix = np.array(rm)
    design.counts_matrix = design.response_matrix * 1.0  # Same


design:Spectrometer
# Construct Spectrometer
design = Spectrometer(FULL_ENERGIES, 
                      measure_type="counts",
                      clip_range=(0, 100),
                      build_engine="NIST")

for i in range(10):
    design.add_layer(Layer(dummy_scint, 0.1))  # Dummy layer 

design.finalise()

setrmcm(design, layer_params)

