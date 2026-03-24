import json
import shutil
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_env_manager = "conda" if shutil.which("conda") else "local"

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: requires the model artifact to be tagged "prod" in W&B before running
    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')  # Adding version_base for Python 3.13 compatibility
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            cfg = config["etl"]["download"]
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                env_manager=_env_manager,
                parameters={
                    "sample": cfg["sample"],
                    "artifact_name": cfg["output_artifact"],
                    "artifact_type": cfg["output_type"],
                    "artifact_description": cfg["output_desc"],
                },
            )

        if "basic_cleaning" in active_steps:
            cfg = config["etl"]["basic_cleaning"]
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                env_manager=_env_manager,
                parameters={
                    "dirty_artifact": cfg["input_artifact"],
                    "clean_artifact": cfg["output_artifact"],
                    "artifact_type": cfg["output_type"],
                    "artifact_description": "Basic cleaned dataset",
                    "filters": json.dumps([list(f) for f in cfg["filters"]]),
                },
            )

        if "data_check" in active_steps:
            cfg = config["etl"]["data_check"]
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                env_manager=_env_manager,
                parameters={
                    "csv": cfg["input_artifact"],
                    "ref": cfg["ref_artifact"],
                    "kl_threshold": cfg["kl_threshold"],
                    "min_price": cfg["min_price"],
                    "max_price": cfg["max_price"],
                },
            )

        if "data_split" in active_steps:
            cfg = config["data_split"]
            mod = config["modeling"]
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager=_env_manager,
                parameters={
                    "input": cfg["input_artifact"],
                    "test_size": mod["test_size"],
                    "random_seed": mod["random_seed"],
                    "stratify_by": mod["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            mod = config["modeling"]
            run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                env_manager=_env_manager,
                parameters={
                    "trainval_artifact": mod["input_artifact"],
                    "val_size": mod["val_size"],
                    "random_seed": mod["random_seed"],
                    "stratify_by": mod["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": mod["max_tfidf_features"],
                    "output_artifact": mod["output_artifact"],
                },
            )
            # Fetch r2 from the latest W&B training run in case we want to run Optuna
            wb_runs = wandb.Api().runs(
                config['main']['project_name'],
                filters={"jobType": "train_random_forest"},
                order="-created_at",
            )
            r2 = float(wb_runs[0].summary["r2"])

        if "test_regression_model" in active_steps:
            cfg = config["test_regression_model"]
            try:
                wandb.Api().artifact(f"{config['main']['project_name']}/{cfg['input_model']}")
            except wandb.errors.CommError:
                logger.warning(
                    f"Skipping test_regression_model: artifact '{cfg['input_model']}' not found in W&B. "
                    "Promote a model export to 'prod' first."
                )
            else:
                _ = mlflow.run(
                    f"{config['main']['components_repository']}/test_regression_model",
                    "main",
                    env_manager=_env_manager,
                    parameters={
                        "mlflow_model": cfg["input_model"],
                        "test_dataset": cfg["input_artifact"],
                    },
                )

    return r2 if "train_random_forest" in active_steps else None


if __name__ == "__main__":
    go()
