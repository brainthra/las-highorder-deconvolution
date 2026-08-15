fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, ss in enumerate([train_ds, val_ds, test_ds]):
    # sns.kdeplot(x=spectra_params['mu'][ss.indices], y=spectra_params['sigma'][ss.indices], ax=ax[i], fill=True)
    # ax[i].scatter(spectra_params['mu'][ss.indices], spectra_params['sigma'][ss.indices], alpha=0.5)
    ax[i].hist2d(spectra_params['mu'][ss.indices], spectra_params['sigma'][ss.indices], bins=50, cmap='viridis', norm=LogNorm())
    ax[i].set_title(["Train Set", "Validation Set", "Test Set"][i])
    ax[i].set_xlabel("mu")
    ax[i].set_ylabel("sigma")

fig.tight_layout()