import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from trainer import MetaTrainer, Trainer
from utils.config_omega import cfg
from utils.err_handler import init_err_handler

###############################################################################
# Customized Utils: Logging, Debug and Exception Handling (Fail Notification)
###############################################################################
from utils.logger import logger

init_err_handler()


###############################################################################
# Customized Functions
###############################################################################
def print_config_changes(prev_cfg, new_cfg, source, rank):
    changes = []
    for k, v in new_cfg.items():
        if v != prev_cfg.get(k):
            changes.append(f"  {k}: {prev_cfg.get(k)} -> {v}")

    if changes and rank == 0:
        print(f"Options updated by {source}:")
        for change in changes:
            print(change)


###############################################################################
# Main
###############################################################################
def main(cfg, device):
    # Logger init after device set.
    logger.init(cfg.output_dir)

    logger.section("Configuration")
    formatted_cfg = OmegaConf.to_yaml(cfg, resolve=True)
    print(formatted_cfg)

    logger.section("Seed Setting")
    if cfg.seed is not None:
        seed = cfg.seed
        logger.info(f"Setting fixed seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        logger.warning("No fixed seed set")

    if cfg.deterministic:
        logger.info("Setting CUDNN to deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.warning("Setting CUDNN to non-deterministic mode")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    logger.section("Trainer Initialization")
    if not cfg.use_meta:
        trainer = Trainer(cfg, device)
    else:
        trainer = MetaTrainer(cfg, device)

    trainer.initialize()

    if cfg.model_dir is not None:
        logger.info(f"Loading model from {cfg.model_dir}")
        trainer.load_model(cfg.model_dir)

    if cfg.zero_shot:
        logger.info("Running in zero-shot mode")
        trainer.test()
    elif cfg.test_train:
        logger.info("Running test on training data")
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[: cfg.output_dir.index("_test_train_True")]
            logger.info(f"Model directory: {cfg.model_dir}")
        trainer.load_model(cfg.model_dir)
        trainer.test("train")
    elif cfg.test_only:
        logger.info("Running test only")
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[: cfg.output_dir.index("_test_only_True")]
            logger.info(f"Model directory: {cfg.model_dir}")
        trainer.load_model(cfg.model_dir)
        trainer.test()
    else:
        logger.info("Starting training")
        trainer.train()

    logger.section("Completion")
    logger.info("Main process completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="data config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model config file",
    )
    parser.add_argument(
        "--tuner",
        type=str,
        default="",
        help="tuner config file",
    )
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        help="modify config options using the command-line (KEY=VALUE pairs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for saving results",
    )
    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(f"Training on {world_size} GPU(s)")
        print("Loading configuration files")

    # Convert base config to OmegaConf
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Load and merge configs
    config_files = [
        (f"./configs/data/{args.dataset}.yaml", "Data config"),
        (f"./configs/model/{args.model}.yaml", "Model config"),
    ]
    if args.tuner:
        config_files.append((f"./configs/tuner/{args.tuner}.yaml", "Tuner config"))

    for config_path, config_name in config_files:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        prev_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
        if local_rank == 0:
            print_config_changes(prev_cfg, cfg, config_name, local_rank)

    # Handle command line options
    if args.opts:
        prev_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cli_conf = OmegaConf.from_dotlist(args.opts)
        cfg = OmegaConf.merge(cfg, cli_conf)
        if local_rank == 0:
            print_config_changes(prev_cfg, cfg, "opts", local_rank)

    # Set output directory
    components = [args.dataset, args.model]
    if args.tuner:
        components.append(args.tuner)

    if args.opts:
        opt_components = []
        for opt in args.opts:
            key, value = opt.split("=")
            if value.lower() == "true":
                opt_components.append(key)
            elif value.lower() == "false":
                continue
            else:
                clean_key = key.replace(".", "_")
                clean_value = (
                    value.strip("[]").replace(",", "-").replace(".", "_").strip()
                )
                opt_components.append(f"{clean_key}_{clean_value}")

        if opt_components:
            components.append("_".join(opt_components))

    base_name = "/".join(components)
    cfg.output_dir = os.path.join(args.output_dir, base_name)

    if local_rank == 0:
        print(f"Using output dir: {cfg.output_dir}")
        os.makedirs(cfg.output_dir, exist_ok=True)

    dist.barrier()
    if local_rank == 0:
        print(">>>>> All Distributed GPUs initialized <<<<<")

    main(cfg, device)
