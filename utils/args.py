import argparse
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--model_name", type=str, default="EnergyMamba")

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")

    parser.add_argument("--quantile_alpha", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="")

    return parser


def get_log_path(args):
    return os.path.join("", args.model_name, args.dataset, "")


def get_data_path():
    return ""


def print_args(logger, args):
    for k, v in vars(args).items():
        logger.info(f"{k:20s}: {v}")
