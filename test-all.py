import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_utils import Resolution, NoiseLevel, FluxLevel, Condition


Emin = 0.001
Emax = 1.0
target_bins = 100
ext_factor = 5.0

model_type = "mlp"
pred_bins_target = 20
pred_tar_scale = "lin"
pred_bins_ext = 10
pred_ext_scale = "log"
ensembles = 10

MAX_WORKERS = 1


resolutions = {
    "int": Resolution(10, 65535, 1, "np.int64"),
    "float": Resolution(0.0001, 65535.0, 0.0001, "float"),
}

spec_types = ["shighorder", "skew", "gaussian", "boltzmann"]

noise_levels = {
    "clean": NoiseLevel(0.0, 0.0),
    "noised": NoiseLevel(0.0, 1.0),
}

flux_levels = {
    "span": FluxLevel(0.01, 1.0),
    "high": FluxLevel(1.0, 1.0),
}

seeds = [0, 1, 10, 33, 42]

conditions = {
    "ideal": Condition(
        resolutions["float"],
        noise_levels["clean"],
        flux_levels["high"],
    ),
    "A": Condition(
        resolutions["int"],
        noise_levels["clean"],
        flux_levels["span"],
    ),
    "B": Condition(
        resolutions["int"],
        noise_levels["noised"],
        flux_levels["span"],
    ),
}

jobs = []
# design = "existing"
# for condition_name, condition in conditions.items():
#     for spec_type in spec_types:
#         for seed in seeds:

#             command = [
#                 "python",
#                 "seed-test-script.py",
#                 "--Bmin", str(condition.resolution.Bmin),
#                 "--Bmax", str(condition.resolution.Bmax),
#                 "--Bstep", str(condition.resolution.Bstep),
#                 "--Btype", condition.resolution.Btype,
#                 "--Emin", str(Emin),
#                 "--Emax", str(Emax),
#                 "--target-bins", str(target_bins),
#                 "--ext-factor", str(ext_factor),
#                 "--design", design,
#                 "--spec-type", spec_type,
#                 "--noise-min", str(condition.noise_level.noise_min),
#                 "--noise-max", str(condition.noise_level.noise_max),
#                 "--flux-min", str(condition.flux_level.flux_min),
#                 "--flux-max", str(condition.flux_level.flux_max),
#                 "--model-type", model_type,
#                 "--pred-bins-target", str(pred_bins_target),
#                 "--pred-tar-scale", pred_tar_scale,
#                 "--pred-bins-ext", str(pred_bins_ext),
#                 "--pred-ext-scale", pred_ext_scale,
#                 "--ensembles", str(ensembles),
#                 "--seed", str(seed),
#             ]

#             jobs.append({
#                 "design": design,
#                 "condition": condition_name,
#                 "spec_type": spec_type,
#                 "seed": seed,
#                 "command": command,
#             })

design = "idealised"
for spec_type in ["shighorder"]:
    for seed in seeds:

        command = [
            "python",
            "seed-test-script.py",
            "--Bmin", str(resolutions["int"].Bmin),
            "--Bmax", str(resolutions["int"].Bmax),
            "--Bstep", str(resolutions["int"].Bstep),
            "--Btype", resolutions["int"].Btype,
            "--Emin", str(Emin),
            "--Emax", str(Emax),
            "--target-bins", str(target_bins),
            "--ext-factor", str(ext_factor),
            "--design", design,
            "--spec-type", spec_type,
            "--noise-min", str(noise_levels["noised"].noise_min),
            "--noise-max", str(noise_levels["noised"].noise_max),
            "--flux-min", str(flux_levels["span"].flux_min),
            "--flux-max", str(flux_levels["span"].flux_max),
            "--model-type", model_type,
            "--pred-bins-target", str(pred_bins_target),
            "--pred-tar-scale", pred_tar_scale,
            "--pred-bins-ext", str(pred_bins_ext),
            "--pred-ext-scale", pred_ext_scale,
            "--ensembles", str(ensembles),
            "--seed", str(seed),
        ]

        jobs.append({
            "design": design,
            "condition": "idealised",
            "spec_type": spec_type,
            "seed": seed,
            "command": command,
        })


print(f"Loaded {len(jobs)} jobs")


def run_job(job):
    print(
        f"Starting: "
        f"condition={job['condition']} "
        f"design={job['design']} "
        f"spec_type={job['spec_type']} "
        f"seed={job['seed']}"
    )

    subprocess.run(
        job["command"],
        check=True,
    )

    return job


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = [
        executor.submit(run_job, job)
        for job in jobs
    ]

    for future in as_completed(futures):
        try:
            job = future.result()

            print(
                f"Finished: "
                f"condition={job['condition']} "
                f"design={job['design']} "
                f"spec_type={job['spec_type']} "
                f"seed={job['seed']}"
            )

        except Exception as exc:
            print(f"Job failed: {exc}")