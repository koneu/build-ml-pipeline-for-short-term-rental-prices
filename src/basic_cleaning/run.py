#!/usr/bin/env python
"""
Performs basic cleanup tasks
"""
import argparse
import json
import logging
import os
import tempfile

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Fetching artifact {args.dirty_artifact}")
    artifact_path = run.use_artifact(args.dirty_artifact).file()
    df = pd.read_csv(artifact_path)

    filters = json.loads(args.filters)
    for column, low, high in filters:
        before = len(df)
        mask = pd.Series(True, index=df.index)
        if low is not None:
            mask &= df[column] >= low
        if high is not None:
            mask &= df[column] <= high
        df = df[mask]
        logger.info(f"Filtered {column} [{low}, {high}]: {before - len(df)} rows dropped")

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["last_review"] = pd.to_datetime(df["last_review"])  # unreviewed entries become NaT

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, args.clean_artifact)
        df.to_csv(output_path, index=False)

        artifact = wandb.Artifact(
            name=args.clean_artifact,
            type=args.artifact_type,
            description=args.artifact_description,
        )
        artifact.add_file(output_path)
        run.log_artifact(artifact)
        artifact.wait()

    logger.info(f"Saved {len(df)} rows to {args.clean_artifact}")
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument("--dirty_artifact", type=str, required=True,
                        help="W&B artifact name for the raw input data")
    parser.add_argument("--clean_artifact", type=str, required=True,
                        help="Output artifact name")
    parser.add_argument("--artifact_type", type=str, required=True,
                        help="W&B artifact type for the output")
    parser.add_argument("--artifact_description", type=str, required=True,
                        help="Description of the output artifact")
    parser.add_argument("--filters", type=str, default="[]",
                        help='JSON array of [column, min, max] triples, e.g. \'[["price",10,350]]\'')

    args = parser.parse_args()
    go(args)