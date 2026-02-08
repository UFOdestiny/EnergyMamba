import os
import sys
import numpy as np
import torch

torch.set_num_threads(8)

from base.engine import BaseEngine
from model.energy_mamba import EnergyMamba
from model.ascqr import ASCQR
from utils.args import get_public_config, get_log_path, print_args
from utils.dataloader import load_dataset, load_adj, get_dataset_info
from utils.graph_algo import compute_gcn_norm
from utils.log import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--unet_depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--ascqr", action="store_true", default=False)
    parser.add_argument("--ascqr_gamma", type=float, default=0.005)
    parser.add_argument("--ascqr_window", type=int, default=100)
    args = parser.parse_args()

    log_dir = get_log_path(args)
    logger = get_logger(log_dir, __name__)
    print_args(logger, args)
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)

    dataloader, scaler = load_dataset(data_path, args, logger)

    adj = load_adj(adj_path)
    A_hat = torch.tensor(compute_gcn_norm(adj), dtype=torch.float32)

    model = EnergyMamba(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_state=args.d_state,
        depth=args.unet_depth,
        dropout=args.dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = BaseEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        A_hat=A_hat,
        loss_fn="Quantile",
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=0,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        normalize=args.normalize,
        alpha=args.quantile_alpha,
        metric_list=["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "COV", "IS"],
        args=args,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate("test", args.model_path)

    if args.ascqr:
        logger.info("Running AS-CQR calibration...")
        engine.load_model(log_dir)
        cal_lo, cal_mid, cal_up, cal_y = engine.get_predictions("val")
        test_lo, test_mid, test_up, test_y = engine.get_predictions("test")

        calibrator = ASCQR(
            alpha=args.quantile_alpha,
            gamma=args.ascqr_gamma,
            window_size=args.ascqr_window,
        )
        adj_lo, adj_up = calibrator.run(cal_lo, cal_up, cal_y, test_lo, test_up, test_y)

        from base.metrics import masked_coverage, masked_mpiw, masked_IS
        cov = masked_coverage(adj_lo, adj_up, test_y)
        mpiw = masked_mpiw(adj_lo, adj_up)
        is_score = masked_IS(adj_lo, adj_up, test_y, alpha=args.quantile_alpha)
        logger.info(f"AS-CQR COV: {cov:.3f}, MPIW: {mpiw:.3f}, IS: {is_score:.3f}")


if __name__ == "__main__":
    main()
