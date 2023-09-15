import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet
import pandas as pd
from sklearn.metrics import r2_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def rmspe(y_true, y_pred):
    """
    Calculate Root Mean Squared Percentage Error
    """
    # Move tensors to CPU and convert them to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Avoid division by zero and convert zeros to nan
    y_true = np.where(y_true == 0, np.nan, y_true)

    rmspe = np.sqrt(np.nanmean(np.square((y_true - y_pred) / y_true)))
    return rmspe


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="data/METR-LA", help="data path")

parser.add_argument(
    "--adj_data", type=str, default="data/sensor_graph/adj_mx.pkl", help="adj data path"
)
parser.add_argument(
    "--gcn_true",
    type=str_to_bool,
    default=True,
    help="whether to add graph convolution layer",
)
parser.add_argument(
    "--buildA_true",
    type=str_to_bool,
    default=True,
    help="whether to construct adaptive adjacency matrix",
)
parser.add_argument(
    "--load_static_feature",
    type=str_to_bool,
    default=False,
    help="whether to load static feature",
)
parser.add_argument(
    "--cl", type=str_to_bool, default=True, help="whether to do curriculum learning"
)

parser.add_argument("--gcn_depth", type=int, default=4, help="graph convolution depth")
parser.add_argument(
    "--num_nodes", type=int, default=31, help="number of nodes/variables"
)
parser.add_argument(
    "--dropout", type=float, default=0.1744673287068504, help="dropout rate"
)
parser.add_argument("--subgraph_size", type=int, default=23, help="k")
parser.add_argument("--node_dim", type=int, default=39, help="dim of nodes")
parser.add_argument(
    "--dilation_exponential", type=int, default=1, help="dilation exponential"
)

parser.add_argument(
    "--conv_channels", type=int, default=56, help="convolution channels"
)
parser.add_argument(
    "--residual_channels", type=int, default=42, help="residual channels"
)
parser.add_argument("--skip_channels", type=int, default=90, help="skip channels")
parser.add_argument("--end_channels", type=int, default=213, help="end channels")


parser.add_argument("--in_dim", type=int, default=1, help="inputs dimension")
parser.add_argument("--seq_in_len", type=int, default=7, help="input sequence length")
parser.add_argument("--seq_out_len", type=int, default=1, help="output sequence length")

parser.add_argument("--layers", type=int, default=9, help="number of layers")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.007457655616415955, help="learning rate"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0002748826394620175,
    help="weight decay rate",
)
parser.add_argument("--clip", type=int, default=5, help="clip")
parser.add_argument("--step_size1", type=int, default=2500, help="step_size")
parser.add_argument("--step_size2", type=int, default=100, help="step_size")


parser.add_argument("--epochs", type=int, default=300, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--save", type=str, default="./save/", help="save path")
parser.add_argument("--expid", type=int, default=1, help="experiment id")

parser.add_argument("--propalpha", type=float, default=0.05, help="prop alpha")
parser.add_argument("--tanhalpha", type=float, default=3, help="adj alpha")

parser.add_argument(
    "--num_split", type=int, default=1, help="number of splits for graphs"
)

parser.add_argument("--runs", type=int, default=1, help="number of runs")


args = parser.parse_args()
torch.set_num_threads(3)


space = {
    "learning_rate": hp.loguniform(
        "learning_rate", -5, 0
    ),  # Log-uniform distribution between e^-5 and 1
    "dropout": hp.uniform(
        "dropout", 0.1, 0.5
    ),  # Uniform distribution between 0.1 and 0.5
    "batch_size": hp.choice(
        "batch_size", [16, 32, 64]
    ),  # Choice among the specified values
    "gcn_depth": hp.choice(
        "gcn_depth", [1, 2, 3, 4, 5]
    ),  # Choice among the specified values
    "layers": hp.quniform(
        "layers", 1, 10, 1
    ),  # Uniform distribution of integer values between 1 and 10
    "weight_decay": hp.loguniform(
        "weight_decay", -10, -4
    ),  # Log-uniform distribution between e^-10 and e^-4
    "subgraph_size": hp.quniform(
        "subgraph_size", 10, 30, 1
    ),  # Uniform distribution of integer values between 10 and 30
    "node_dim": hp.quniform(
        "node_dim", 20, 60, 1
    ),  # Uniform distribution of integer values between 20 and 60
    "conv_channels": hp.quniform(
        "conv_channels", 16, 64, 1
    ),  # Uniform distribution of integer values between 16 and 64
    "residual_channels": hp.quniform(
        "residual_channels", 16, 64, 1
    ),  # Uniform distribution of integer values between 16 and 64
    "skip_channels": hp.quniform(
        "skip_channels", 32, 128, 1
    ),  # Uniform distribution of integer values between 32 and 128
    "end_channels": hp.quniform(
        "end_channels", 64, 256, 1
    ),  # Uniform distribution of integer values between 64 and 256
}


def main(runid, hyperparams):
    parser.set_defaults(
        learning_rate=hyperparams["learning_rate"],
        dropout=hyperparams["dropout"],
        batch_size=int(hyperparams["batch_size"]),  # Ensure this is an integer
        gcn_depth=int(hyperparams["gcn_depth"]),  # Ensure this is an integer
        layers=int(hyperparams["layers"]),  # Ensure this is an integer
        weight_decay=hyperparams["weight_decay"],
        subgraph_size=int(hyperparams["subgraph_size"]),  # Ensure this is an integer
        node_dim=int(hyperparams["node_dim"]),  # Ensure this is an integer
        conv_channels=int(hyperparams["conv_channels"]),  # Ensure this is an integer
        residual_channels=int(
            hyperparams["residual_channels"]
        ),  # Ensure this is an integer
        skip_channels=int(hyperparams["skip_channels"]),  # Ensure this is an integer
        end_channels=int(hyperparams["end_channels"]),  # Ensure this is an integer
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    dataloader = load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    predefined_A = pd.read_csv("data/sensor_graph/distance.csv", header=None)

    # Assuming 'matrix' is your 2D NumPy array
    mean = np.mean(predefined_A)
    std = np.std(predefined_A)

    # Z-score normalization
    predefined_A = (predefined_A - mean) / std
    predefined_A = np.array([predefined_A])
    predefined_A = torch.tensor(predefined_A)

    # predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    model = gtnet(
        args.gcn_true,
        args.buildA_true,
        args.gcn_depth,
        args.num_nodes,
        device,
        predefined_A=predefined_A,
        dropout=args.dropout,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        seq_length=args.seq_in_len,
        in_dim=args.in_dim,
        out_dim=args.seq_out_len,
        layers=args.layers,
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        layer_norm_affline=True,
    )

    print(args)
    print("The recpetive field size is", model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print("Number of model parameters is", nParams)

    engine = Trainer(
        model,
        args.learning_rate,
        args.weight_decay,
        args.clip,
        args.step_size1,
        args.seq_out_len,
        scaler,
        device,
        args.cl,
    )

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        # dataloader["train_loader"].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes / args.num_split)
            for j in range(args.num_split):
                if j != args.num_split - 1:
                    id = perm[j * num_sub : (j + 1) * num_sub]
                else:
                    id = perm[j * num_sub :]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train(tx, ty[:, 0, :, :], id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}"
                print(
                    log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),
                    flush=True,
                )
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch"
        print(
            log.format(
                i,
                mtrain_loss,
                mtrain_mape,
                mtrain_rmse,
                mvalid_loss,
                mvalid_mape,
                mvalid_rmse,
                (t2 - t1),
            ),
            flush=True,
        )

        if mvalid_loss < minl:
            torch.save(
                engine.model.state_dict(),
                args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth",
            )
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
    )

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # valid data
    outputs = []
    realy = torch.Tensor(dataloader["y_val"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(dim=1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    pred = scaler.inverse_transform(yhat)
    vmae, vmape, vrmse = metric(pred, realy)

    # test data
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(dim=1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    rmspe_list = []
    r2_list = []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        rmspe_value = rmspe(real, pred)  # Calculate RMSPE
        r2_value = r2_score(
            real.cpu().numpy(), pred.cpu().numpy()
        )  # Calculate R-squared

        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test RMSPE: {:.4f}, Test R2: {:.4f}"
        print(
            log.format(i + 1, metrics[0], metrics[1], metrics[2], rmspe_value, r2_value)
        )

        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
        rmspe_list.append(rmspe_value)  # Store RMSPE values in a list
        rmspe_array = np.array(rmspe_list)
        r2_list.append(r2_value)  # Store R-squared values in a list
        r2_array = np.array(r2_list)
    return vmae, vmape, vrmse, mae, mape, rmse, rmspe_array, r2_array, minl


def objective(hyperparams):
    try:
        # Call your main function with the current set of hyperparameters
        vmae, vmape, vrmse, mae, mape, rmse, rmspe_array, r2_array, minl = main(
            0, hyperparams
        )
        objective_value = minl
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        objective_value = float("inf")

    # Return the objective value and some extra information
    return {"loss": objective_value, "status": STATUS_OK, "hyperparams": hyperparams}


trials = Trials()

best = fmin(
    fn=objective,  # Objective function
    space=space,  # Hyperparameter space
    algo=tpe.suggest,  # Optimization algorithm (Tree of Parzen Estimators)
    max_evals=50,  # Maximum number of evaluations
    trials=trials,  # Trials object to store the results of each evaluation
)

print("Best hyperparameters found:")
print(best)

# Find the trial with the lowest objective value
best_trial = sorted(trials, key=lambda x: x["result"]["loss"])[0]

print("Best hyperparameters:")
print(best_trial["result"]["hyperparams"])

print("Best objective value:")
print(best_trial["result"]["loss"])


if __name__ == "__main__":
    best_hyperparams = best_trial["result"]["hyperparams"]
    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    rmspe4 = []
    r_squared = []
    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3, m4, m5, _ = main(i, best_hyperparams)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        rmspe4.append(m4)
        r_squared.append(m5)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)
    rmspe4 = np.array(rmspe4)
    r_squared = np.array(r_squared)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)
    armspe = np.mean(rmspe4, 0)
    arsquared = np.mean(r_squared, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)
    srmspe = np.std(rmspe4, 0)
    sr_squared = np.std(r_squared, 0)

    print("\n\nResults for 10 runs\n\n")

    # valid data
    print("valid\tMAE\tRMSE\tMAPE")
    log = "mean:\t{:.4f}\t{:.4f}\t{:.4f}"
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    log = "std:\t{:.4f}\t{:.4f}\t{:.4f}"
    print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print("\n\n")

    # test data
    print("test|horizon\tMAE-mean\tMAPE-mean\tRMSE-mean\tRMSPE-mean\tR2-mean")
    # Determine the number of horizons from the shape of one of your metric arrays
    # num_horizons = amae.shape[0]

    # Calculate the overall average and standard deviation for each metric
    overall_amae = np.mean(amae)
    overall_amape = np.mean(amape) * 100  # Convert to percentage
    overall_armse = np.mean(armse)
    overall_armspe = np.mean(armspe) * 100  # Convert to percentage
    overall_arsquared = np.mean(arsquared)

    # overall_smae = np.std(amae)
    # overall_smape = np.std(amape)  # Standard deviation without percentage
    # overall_srmse = np.std(rmse)
    # overall_srmspe = np.std(armspe)  # Standard deviation without percentage
    # overall_sr_squared = np.std(r_squared)

    # Print the overall averages and standard deviations
    log = "Overall\t{:.3f}\t{:.3f}%\t{:.3f}\t{:.3f}%\t{:.3f}"
    print(
        log.format(
            overall_amae,
            overall_amape,
            overall_armse,
            overall_armspe,
            overall_arsquared,
            # overall_smae,
            # overall_smape,
            # overall_srmse,
            # overall_srmspe,
            # overall_sr_squared,
        )
    )


# Three places to change when changing horizons
# - output sequence
# - generate_training_data
