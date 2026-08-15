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

MAX_WORKERS = 2


seeds = [0, 1, 10, 33, 42]

target_compss = [
    "boltz_boltz",
    "gauss_gauss",
    "boltz_gauss",
]

jobs = []


for target_comps in target_compss:
    for seed in seeds:
        command = [
            "python",
            "seed-bench-our.py",
            "--target_comps", target_comps,
            "--seed", str(seed),
        ]

        jobs.append({
            "target_comps": target_comps,
            "seed": seed,
            "command": command,
        })


print(f"Loaded {len(jobs)} jobs")


def run_job(job):
    print(
        f"Starting: "
        f"target_comps={job['target_comps']} "
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
                f"target_comps={job['target_comps']} "
                f"seed={job['seed']}"
            )

        except Exception as exc:
            print(f"Job failed: {exc}")