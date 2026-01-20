'''
Aggregates results from hyperparameter optimization and performs predictive 
performance estimations on an extrapolated experimental envelope. Final model
state is provided in a .pt and .onnx file.

Arthur - arthur-miguel.github.io - 12/22/2025
'''

from collections import Counter, defaultdict
from sklearn.model_selection import LeaveOneGroupOut
import glob
import math
from train_gpu import *

set_seed(42)

def aggregate_best_params(folds_dir, prefer_mode_for_all=False):
    paths = glob.glob(os.path.join(folds_dir, 'fold_*', 'best_params.joblib'))
    if len(paths) == 0:
        raise RuntimeError(f'No best_params.joblib found in {folds_dir} (expected inside fold_* folders).')

    all_params = []
    for p in paths:
        try:
            params = joblib.load(p)
            all_params.append(params)
        except Exception as e:
            print(f"Warning: unable to load {p}: {e}")

    if len(all_params) == 0:
        raise RuntimeError('No readable best_params found.')

    vals = defaultdict(list)
    for d in all_params:
        for k, v in d.items():
            vals[k].append(v)

    final = {}
    for k, vlist in vals.items():
        types = set(type(x) for x in vlist if x is not None)

        if prefer_mode_for_all or any(t in (bool, str) for t in types) or all(isinstance(x, int) for x in vlist):
            c = Counter(vlist)
            most_common = c.most_common()
            maxcount = most_common[0][1]
            candidates = [val for val, cnt in most_common if cnt == maxcount]
            final[k] = sorted(candidates)[0]
        else:
            try:
                arr = np.array(vlist, dtype=float)
                final[k] = float(np.median(arr))
            except Exception:
                c = Counter(vlist)
                final[k] = c.most_common(1)[0][0]

    final.setdefault('n_layers', 2)
    final.setdefault('width', 64)
    final.setdefault('dropout', 0.1)
    final.setdefault('use_relu', True)
    final.setdefault('use_bn', False)
    final.setdefault('lr', 1e-3)
    final.setdefault('batch_size', 32)

    final['n_layers'] = int(round(final['n_layers']))
    final['width'] = int(round(final['width']))
    final['batch_size'] = int(round(final['batch_size']))
    final['use_relu'] = bool(final['use_relu'])
    final['use_bn'] = bool(final['use_bn'])
    final['dropout'] = float(final['dropout'])
    final['lr'] = float(final['lr'])

    return final

def make_re_groups(df, re_col='Re_medio', n_groups=5, labels=None):
    if re_col not in df.columns:
        raise ValueError(f"Column '{re_col}' not in dataframe.")
    try:
        groups = pd.cut(df[re_col], bins=n_groups, labels=False, include_lowest=True)
    except Exception:
        groups = pd.qcut(df[re_col], q=n_groups, labels=False, duplicates='drop')
    groups = groups.astype(int).to_numpy()
    return groups

def run_logocv_with_groups(df, X_all, y_all, groups, best_params, args, device, out_dir):
    logo = LeaveOneGroupOut()
    n_samples = X_all.shape[0]
    preds = np.zeros_like(y_all)
    trues = np.zeros_like(y_all)
    group_ids = np.unique(groups)
    results = []

    os.makedirs(out_dir, exist_ok=True)

    hidden = [best_params['width']] * best_params['n_layers']

    fold_idx = 0
    for train_idx, test_idx in logo.split(X_all, y_all, groups):
        fold_idx += 1
        group_left_out = int(groups[test_idx][0])
        print(f"\n--- LOGOCV fold {fold_idx}/{len(group_ids)} | leaving out Re group {group_left_out} (test size {len(test_idx)}) ---")

        X_tr = X_all[train_idx]
        y_tr = y_all[train_idx]
        X_te = X_all[test_idx]
        y_te = y_all[test_idx]

        x_scaler = StandardScaler()
        X_tr_s = x_scaler.fit_transform(X_tr)
        X_te_s = x_scaler.transform(X_te)
        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(y_tr)
        y_te_s = y_scaler.transform(y_te)

        train_ds = TabularDataset(X_tr_s, y_tr_s)
        train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True, num_workers=0,
                                  pin_memory=(device == "cuda"))

        model = MLP(X_tr_s.shape[1], hidden,
                    dropout=best_params['dropout'],
                    use_relu=best_params['use_relu'],
                    use_bn=best_params['use_bn'],
                    out_dim=y_tr_s.shape[1]).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        best_val = np.inf
        counter = 0
        patience = 30
        best_state = None

        for epoch in range(1000):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            if train_loss < best_val:
                best_val = train_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f'  Early stop epoch {epoch} (train_loss {train_loss:.6f})')
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_te_s, dtype=torch.float32).to(device)
            pred_scaled = model(X_te_t).cpu().numpy()

        pred_inv = y_scaler.inverse_transform(pred_scaled)
        trues_inv = y_scaler.inverse_transform(y_te_s)

        preds[test_idx] = pred_inv
        trues[test_idx] = trues_inv

        mse_group = np.mean((pred_inv - trues_inv) ** 2, axis=0)
        rmse_group = np.sqrt(mse_group)
        mae_group = np.mean(np.abs(pred_inv - trues_inv), axis=0)
        mape_group = np.mean(np.abs((trues_inv - pred_inv) / (trues_inv + 1e-16)), axis=0)

        print(f" Group {group_left_out} metrics (unscaled) - RMSE: {rmse_group}, MAE: {mae_group}")

        # save model state and scalers for this fold
        fold_dir = os.path.join(out_dir, f'logocv_group_{group_left_out}')
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, 'model_state.pt'))
        joblib.dump(x_scaler, os.path.join(fold_dir, 'x_scaler.joblib'))
        joblib.dump(y_scaler, os.path.join(fold_dir, 'y_scaler.joblib'))
        joblib.dump(best_params, os.path.join(fold_dir, 'best_params_used.joblib'))

        results.append({
            'group': int(group_left_out),
            'n_test': int(len(test_idx)),
            'rmse_fexpandido': float(rmse_group[0]),
            'rmse_festrangulado': float(rmse_group[1]),
            'mae_fexpandido': float(mae_group[0]),
            'mae_festrangulado': float(mae_group[1]),
            'mape_fexpandido': float(mape_group[0]),
            'mape_festrangulado': float(mape_group[1]),
            'train_loss_at_best': float(best_val)
        })

    preds_all = preds
    trues_all = trues
    df_res = pd.DataFrame(results)

    print("\n=== LOGOCV aggregated group-wise metrics ===")
    for target in ["fexpandido", "festrangulado"]:
        rmse_col = f"rmse_{target}"
        mae_col = f"mae_{target}"
        mape_col = f"mape_{target}"

        rmse_mean = df_res[rmse_col].mean()
        rmse_std  = df_res[rmse_col].std()

        mae_mean = df_res[mae_col].mean()
        mae_std  = df_res[mae_col].std()

        mape_mean = df_res[mape_col].mean()
        mape_std  = df_res[mape_col].std()

        print(f"Target: {target}")
        print(f"\tRMSE\t{rmse_mean:.5f} ± {rmse_std:.5f}")
        print(f"\tMAE\t{mae_mean:.5f} ± {mae_std:.5f}")
        print(f"\tMAPE\t{mape_mean:.5f} ± {mape_std:.5f}")


    out_preds_df = df.copy()
    out_preds_df['pred_fexpandido'] = preds_all[:, 0]
    out_preds_df['pred_festrangulado'] = preds_all[:, 1]
    out_preds_df.to_csv(os.path.join(out_dir, 'logocv_group_predictions.csv'), index=False)
    joblib.dump(results, os.path.join(out_dir, 'logocv_group_results.joblib'))

    return preds_all, trues_all, results

def train_final_on_full(df, X_all, y_all, best_params, args, device, out_dir):
    out_path = os.path.join(out_dir, 'final_model_overall_params.pt')

    hidden = [best_params['width']] * best_params['n_layers']

    x_scaler = StandardScaler()
    X_s = x_scaler.fit_transform(X_all)
    y_scaler = StandardScaler()
    y_s = y_scaler.fit_transform(y_all)

    train_ds = TabularDataset(X_s, y_s)
    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))

    model = MLP(X_s.shape[1], hidden,
                dropout=best_params['dropout'],
                use_relu=best_params['use_relu'],
                use_bn=best_params['use_bn'],
                out_dim=y_s.shape[1]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    counter = 0
    patience = 30
    best_state = None

    for epoch in range(1000):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        if train_loss < best_val:
            best_val = train_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f'Final model early stop at epoch {epoch}, train_loss {train_loss:.6f}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
        pred_scaled = model(X_t).cpu().numpy()

    preds = y_scaler.inverse_transform(pred_scaled)

    preds_all = preds
    trues_all = y_all
    mse_per_target = np.mean((preds_all - trues_all) ** 2, axis=0)
    rmse_per_target = np.sqrt(mse_per_target)
    mae_per_target = np.mean(np.abs(preds_all - trues_all), axis=0)
    mape_per_target = np.mean(np.abs((trues_all - preds_all) / (trues_all + 1e-16)), axis=0)
    
    print('\n=== Final Model Results ===')
    print("Target: fexpandido")
    print(f"\tRMSE\t{rmse_per_target[0]:.5f}")
    print(f"\tMAE\t{mae_per_target[0]:.5f}")
    print(f"\tMAPE\t{mape_per_target[0]:.5f}")
    print("Target: festrangulado")
    print(f"\tRMSE\t{rmse_per_target[1]:.5f}")
    print(f"\tMAE\t{mae_per_target[1]:.5f}")
    print(f"\tMAPE\t{mape_per_target[1]:.5f}")


    out_preds_df = df.copy()
    out_preds_df['pred_fexpandido'] = preds_all[:, 0]
    out_preds_df['pred_festrangulado'] = preds_all[:, 1]
    out_preds_df.to_csv(os.path.join(out_dir, 'final_model_predictions.csv'), index=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    joblib.dump(x_scaler, out_path.replace('.pt', '_x_scaler.joblib'))
    joblib.dump(y_scaler, out_path.replace('.pt', '_y_scaler.joblib'))
    print(f"Saved final model to {out_path} and scalers.")

    return model, x_scaler, y_scaler

def run_logocv(args, device):
    folds_dir = args.out
    df = pd.read_csv(args.data)

    drop_cols = ['Unnamed: 0', 'fexpandido', 'festrangulado', 'Dpexpandido', 'Dpestrangulado']
    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
    y_all = df[['fexpandido', 'festrangulado']].values

    print("Aggregating best_params from each fold...")
    best_params_overall = aggregate_best_params(folds_dir)
    print("Aggregated params:", best_params_overall)
    joblib.dump(best_params_overall, os.path.join(folds_dir, 'best_params_overall.joblib'))

    n_re_groups = 6
    groups = make_re_groups(df, re_col='Re_medio', n_groups=n_re_groups)

    logocv_out = os.path.join(folds_dir, 'logocv_re_groups')
    _, _, _ = run_logocv_with_groups(df, X_all, y_all, groups, best_params_overall, args, device, logocv_out)

    train_final_on_full(df, X_all, y_all, best_params_overall, args, device, folds_dir)

    print("Finalize: done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nested CV with GPU/CPU Optuna')

    parser.add_argument('--data', type=str, default='./data/data.txt')
    parser.add_argument('--out', type=str, default='./out/test')

    args = parser.parse_args()

    run_logocv(args, "cuda")
