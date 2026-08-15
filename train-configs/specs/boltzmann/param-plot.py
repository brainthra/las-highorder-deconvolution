fig, ax = plt.subplots(3, 1, figsize=(15, 5))
for i, ss in enumerate([train_ds, val_ds, test_ds]):
    sns.histplot(spectra_params['temps'][ss.indices], bins=100, kde=True, ax=ax[i])
    ax[i].set_title(["Train Set", "Validation Set", "Test Set"][i])
    ax[i].set_xlabel("T")

fig.tight_layout()