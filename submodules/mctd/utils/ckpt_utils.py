from pathlib import Path
import wandb


def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    # If the model checkpoint is already downloaded, return the path.
    if (download_dir / run_path / "model.ckpt").exists():
        return download_dir / run_path / "model.ckpt"
    else:
        api = wandb.Api()
        run = api.run(run_path)
        # Find the latest saved model checkpoint.
        latest = None
        for artifact in run.logged_artifacts():
            if artifact.type != "model" or artifact.state != "COMMITTED":
                continue

            if latest is None or version_to_int(artifact) > version_to_int(latest):
                latest = artifact

        # Download the checkpoint.
        download_dir.mkdir(exist_ok=True, parents=True)
        root = download_dir / run_path
        latest.download(root=root)
        return root / "model.ckpt"