'''
Nested cross validation scheme for models predictive performance evaluation
accelerated by CUDA and cuDNN. Hyperparameter optimization is performed using 
the Optuna framework and the Tree-Parzen-Estimator algorithm inside a k-fold
loop for robustness. Model validation is performed on an outer LOOCV loop in 
data not seen by the optimization procedure. Model states of each outer loop
resutls are saved on output folder.

Usage:
python train_gpu.py \
    --data <path_to_dataset> \
    --out <path_to_output_folder> \
    --optuna_storage sqlite:///<path_to_output_folder>/optuna.db \
    --study_name <test_name> \ 
    --n_trials_inner 80 \
    --k 5 \
    --epochs_inner 30 \
    --use_gpu

Arthur - arthur-miguel.github.io - 12/22/2025
'''

import os
import argparse
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut

import optuna
from optuna.samplers import NSGAIISampler, TPESampler

import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0.1, use_relu=True, use_bn=False, out_dim=2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU() if use_relu else nn.GELU())
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# def compute_metrics(true, pred):
#     se = np.sqrt((true - pred)**2)
#     rmse = np.mean(se)
#     std = np.std(se)
#     return rmse, std

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n

def validation_rmse(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            p = model(Xb).cpu().numpy()
            preds.append(p)
            trues.append(yb.numpy())

    if len(preds) == 0:
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mse_per_output = np.mean((preds - trues) ** 2, axis=0)
    rmse_per_output = np.sqrt(mse_per_output)
    mae_per_output = np.mean(np.abs(preds - trues), axis=0)
    return rmse_per_output, mae_per_output

def fold_objective(hpo_npz_path, k_folds, train_loader_args, epochs_inner, device):
    data = np.load(hpo_npz_path)
    X_outer_train = data['X']
    y_outer_train = data['y']

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 10)
        width = trial.suggest_int('width', 32, 1024)
        dropout = trial.suggest_float('dropout', 0.0, 0.4)
        use_bn = trial.suggest_categorical('use_bn', [False, True])
        use_relu = trial.suggest_categorical('use_relu', [False, True])
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        hidden = [width] * n_layers
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        rmse_f1_folds = []
        rmse_f2_folds = []

        for train_idx, val_idx in kf.split(X_outer_train):
            X_tr = X_outer_train[train_idx]
            y_tr = y_outer_train[train_idx]
            X_val = X_outer_train[val_idx]
            y_val = y_outer_train[val_idx]

            x_scaler = StandardScaler()
            X_tr = x_scaler.fit_transform(X_tr)
            X_val = x_scaler.transform(X_val)
            y_scaler = StandardScaler()
            y_tr = y_scaler.fit_transform(y_tr)
            y_val = y_scaler.transform(y_val)

            train_ds = TabularDataset(X_tr, y_tr)
            val_ds = TabularDataset(X_val, y_val)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **train_loader_args)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **train_loader_args)

            model = MLP(
                X_outer_train.shape[1], hidden,
                dropout=dropout, use_relu=use_relu,
                use_bn=use_bn, out_dim=y_outer_train.shape[1]
            ).to(device)

            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            loss_fn = nn.MSELoss()

            best_val_sum = np.inf
            best_rmse = None

            for e in range(epochs_inner):
                train_epoch(model, train_loader, opt, loss_fn, device)
                rmse_per_output, _ = validation_rmse(model, val_loader, device)
                sum_rmse = float(np.nansum(rmse_per_output))

                if sum_rmse < best_val_sum:
                    best_val_sum = sum_rmse
                    best_rmse = rmse_per_output.copy()

            rmse_f1_folds.append(best_rmse[0])
            rmse_f2_folds.append(best_rmse[1])

        avg_rmse_f1 = float(np.mean(rmse_f1_folds))
        avg_rmse_f2 = float(np.mean(rmse_f2_folds))
        return avg_rmse_f1 + avg_rmse_f2

    return objective

def nested_loocv_pipeline(args):
    set_seed(args.seed)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"\n>>> Using device: {device}")

    df = pd.read_csv(args.data)

    required = ['fexpandido', 'festrangulado']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Required column '{r}' missing in dataset")

    drop_cols = ['Unnamed: 0', 'fexpandido', 'festrangulado', 'Dpexpandido', 'Dpestrangulado']

    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
    y_all = df[['fexpandido', 'festrangulado']].values

    # joblib.dump(x_scaler, os.path.join(args.out, 'x_scaler.joblib'))
    # joblib.dump(y_scaler, os.path.join(args.out, 'y_scaler.joblib'))

    plots_dir = os.path.join(args.out, 'plots')
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    loo = LeaveOneOut()
    n_samples = X_all.shape[0]

    preds_loocv = np.zeros_like(y_all)
    trues = np.zeros_like(y_all)

    train_loader_args = {
        'num_workers': args.num_workers,
        'pin_memory': True if device.type == "cuda" else False
    }

    study_base = args.study_name if args.study_name else f"nested_shared_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    fold_count = 0
    for train_idx, test_idx in loo.split(X_all):
        fold_count += 1
        print(f"\n--- LOOCV outer fold {fold_count}/{n_samples} | test index: {test_idx[0]} ---")

        X_outer_train = X_all[train_idx]
        y_outer_train = y_all[train_idx]
        X_outer_test = X_all[test_idx]
        y_outer_test = y_all[test_idx]



        fold_dir = os.path.join(args.out, f'fold_{fold_count}')
        os.makedirs(fold_dir, exist_ok=True)
        hpo_path = os.path.join(fold_dir, 'hpo_data.npz')
        np.savez(hpo_path, X=X_outer_train, y=y_outer_train)

        study_name = f"{study_base}_fold{fold_count}"

        sampler = TPESampler(seed=args.seed)

        if args.optuna_storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=args.optuna_storage,
                direction='minimize',
                sampler=sampler,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(direction='minimize', sampler=sampler)

        objective = fold_objective(
            hpo_path, args.k, train_loader_args, args.epochs_inner, device
        )

        print(f"Starting HPO with {device}...")
        study.optimize(objective, n_trials=args.n_trials_inner, timeout=args.timeout_inner, n_jobs=1)

        if study.best_trial is None:
            raise RuntimeError('No completed trials in inner optimization')
        best_params = study.best_trial.params
        joblib.dump(best_params, os.path.join(fold_dir, 'best_params.joblib'))

        best_params.setdefault('n_layers', 2)
        best_params.setdefault('width', 64)
        best_params.setdefault('dropout', 0.1)
        best_params.setdefault('use_relu', True)
        best_params.setdefault('use_bn', False)
        best_params.setdefault('lr', 1e-3)
        best_params.setdefault('batch_size', 32)

        hidden = [best_params['width']] * best_params['n_layers']

        x_scaler = StandardScaler()
        X_outer_train = x_scaler.fit_transform(X_outer_train)
        X_outer_test = x_scaler.transform(X_outer_test)
        y_scaler = StandardScaler()
        y_outer_train = y_scaler.fit_transform(y_outer_train)
        y_outer_test = y_scaler.transform(y_outer_test)


        model = MLP(
            X_outer_train.shape[1], hidden,
            dropout=best_params['dropout'],
            use_relu=best_params['use_relu'],
            use_bn=best_params['use_bn'],
            out_dim=y_outer_train.shape[1]
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        train_ds = TabularDataset(X_outer_train, y_outer_train)
        train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

        best_val = np.inf
        patience = args.patience_outer
        counter = 0

        for epoch in range(args.epochs_outer):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            if train_loss < best_val:
                best_val = train_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f'Early stop at epoch {epoch}, train_loss {train_loss:.6f}')
                break

        model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_outer_test, dtype=torch.float32).to(device)
            pred_scaled = model(X_test_t).cpu().numpy()[0]

        preds_loocv[test_idx[0]] = pred_scaled
        trues[test_idx[0]] = y_outer_test[0]

        pred_inv = y_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
        true_inv = y_scaler.inverse_transform(y_outer_test)[0]

        rmse_fold = np.sqrt((pred_inv - true_inv) ** 2)
        mae_fold = np.abs(pred_inv - true_inv)

        print(f"Fold {fold_count} metrics (unscaled):")
        print(f"  RMSE - fexpandido: {rmse_fold[0]:.6f}, festrangulado: {rmse_fold[1]:.6f}")
        print(f"  MAE  - fexpandido: {mae_fold[0]:.6f}, festrangulado: {mae_fold[1]:.6f}")

        torch.save(model.state_dict(), os.path.join(fold_dir, 'model_state.pt'))

        if args.debug and fold_count >= args.debug_max_folds:
            break

    preds_inv = y_scaler.inverse_transform(preds_loocv)
    trues_inv = y_scaler.inverse_transform(trues)

    mse_per_target = np.mean((preds_inv - trues_inv) ** 2, axis=0)
    rmse_per_target = np.sqrt(mse_per_target)

    print('\n=== LOOCV aggregated results ===')
    print('RMSE fexpandido:', rmse_per_target[0])
    print('RMSE festrangulado:', rmse_per_target[1])
    print('Sum RMSE:', float(np.sum(rmse_per_target)))

    df_results = df.copy()
    df_results['pred_fexpandido'] = preds_inv[:, 0]
    df_results['pred_festrangulado'] = preds_inv[:, 1]
    df_results.to_csv(os.path.join(args.out, 'loocv_predictions.csv'), index=False)

    print('Saved LOOCV predictions to', os.path.join(args.out, 'loocv_predictions.csv'))
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nested CV with GPU/CPU Optuna')

    parser.add_argument('--data', type=str, default='./data/data.txt')
    parser.add_argument('--out', type=str, default='./out/test')

    parser.add_argument('--n_trials_inner', type=int, default=40)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs_inner', type=int, default=30)
    parser.add_argument('--epochs_outer', type=int, default=500)
    parser.add_argument('--patience_outer', type=int, default=30)
    parser.add_argument('--timeout_inner', type=int, default=None)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--optuna_storage', type=str, default=None)
    parser.add_argument('--study_name', type=str, default='nested_shared')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_max_folds', type=int, default=4)
    parser.add_argument('--n_jobs', type=int, default=24)

    parser.add_argument('--use_gpu', action='store_true',
                        help='Enable GPU training if CUDA is available')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.optuna_storage and args.optuna_storage.startswith('sqlite:///'):
        import sqlalchemy as sa
        engine = sa.create_engine(args.optuna_storage)
        with engine.connect() as conn:
            conn.execute(sa.text("PRAGMA journal_mode=WAL"))
            conn.execute(sa.text("PRAGMA busy_timeout=15000"))

    nested_loocv_pipeline(args)
