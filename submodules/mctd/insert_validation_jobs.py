import os
import copy
import json
from datetime import datetime

seeds = list(range(1))
task_ids = [1]

wandb_entity = "r-pad"
wandb_project = "mctd"

basic_configs = [
    ##############################################################
    # Diffuser
    ##############################################################
    {
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.group": "PushBlock",
        "+name": "Diffuser-unguided",
        "experiment": "exp_planning",
        "algorithm": "df_planning",
        "algorithm.mctd": False,
        "dataset": "pushblock_offline",
        "dataset.episode_len": 48,
        "experiment.tasks": ["validation"],
        "experiment.validation.batch_size": 1,
        "load": "562j5rbm",
    }
]

configs = []
for config in basic_configs:
    # group = copy.deepcopy(config["wandb.group"])
    name = copy.deepcopy(config["+name"])
    for task_id in task_ids:
        for seed in seeds:
            config = copy.deepcopy(config)
            config["experiment.validation.seed"] = seed
            config["algorithm.task_id"] = task_id
            config["+name"] = f"{name}_TaskID{task_id}_Seed{seed}"
            configs.append(config)

# Check there is the jobs folder
jobs_folder = "jobs"
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

# Write the jobs with each config, which name is current time (Too quickly to be overwritten)
for config in configs:
    with open(
        f"{jobs_folder}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.json", "w"
    ) as f:
        json.dump(config, f, indent=4)

print(f"Generated validation {len(configs)} jobs")
