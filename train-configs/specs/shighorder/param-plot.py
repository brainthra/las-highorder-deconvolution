n_gausss = np.array([prms['n_gauss'] for prms in spectra_params])
n_boltzs = np.array([prms['n_boltz'] for prms in spectra_params])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, ss in enumerate([train_ds, val_ds, test_ds]):
    # sns.kdeplot(x=spectra_params['mu'][ss.indices], y=spectra_params['sigma'][ss.indices], ax=ax[i], fill=True)
    # create n_gauss x n_boltz histogram
    n_gauss = n_gausss[ss.indices]
    n_boltz = n_boltzs[ss.indices]
    hist, xedges, yedges = np.histogram2d(n_gauss, n_boltz, bins=(np.arange(-0.5, n_gauss.max()+1.5, 1), np.arange(-0.5, n_boltz.max()+1.5, 1)))
    ax[i].imshow(hist.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='Blues')
    # set text with counts in each bin
    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            if hist[x, y] > 0:
                ax[i].text(x, y, f"{int(hist[x, y])}\n {int(hist[x, y])/hist.sum():.2%}", color='black', ha='center', va='center', fontsize=8)
    ax[i].set_title(["Train Set", "Validation Set", "Test Set"][i])
    ax[i].set_xlabel("n_gauss")
    ax[i].set_ylabel("n_boltz")
    ax[i].set_xticks(np.arange(0, n_gausss.max()+1, 1))
    ax[i].set_yticks(np.arange(0, n_boltzs.max()+1, 1))
    ax[i].grid(False)

fig.tight_layout()

n_g = np.array([prms['n_gauss'] for prms in spectra_params])[test_ds.indices]
n_b = np.array([prms['n_boltz'] for prms in spectra_params])[test_ds.indices]
# save spectra_params[indices]
# np.save("res/test_n_gauss.npy", n_g)
# np.save("res/test_n_boltz.npy", n_b)

# np.save("res/test_spectra_params.npy", np.array(spectra_params)[test_ds.indices])