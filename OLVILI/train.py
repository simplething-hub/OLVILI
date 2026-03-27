import torch
import torch.nn as nn
import torch.nn.functional as F
from dataload import load_data
from model import GraphContrastiveAEClassifier
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
def save_results_txt(
    save_path,
    ACC,
    F1,
    AUC,
    Sensitivity,
    Specificity,
    alpha,
    beta,
    latent_dim
):
    with open(save_path, 'a', encoding='utf-8') as f:
        line = (
            f"ACC={ACC:.6f}, "
            f"F1={F1:.6f}, "
            f"AUC={AUC:.6f}, "
            f"Sensitivity={Sensitivity:.6f}, "
            f"Specificity={Specificity:.6f}, "
            f"alpha={alpha}, "
            f"beta={beta}, "
            f"latent_dim={latent_dim}\n"
        )
        f.write(line)
def evaluate_binary_metrics(logits, labels, threshold=0.5):
    """
    logits: Tensor or ndarray, shape (N,)
    labels: Tensor or ndarray, shape (N,)
    """

    # ---------- logits → numpy ----------
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    else:
        probs = 1 / (1 + np.exp(-logits))  # sigmoid

    # ---------- labels → numpy ----------
    if isinstance(labels, torch.Tensor):
        y_true = labels.detach().cpu().numpy()
    else:
        y_true = labels

    y_pred = (probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return acc, f1, auc, sensitivity, specificity

def get_predictions(logits):
    """
    logits: torch.Tensor (N,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return probs.cpu().numpy(), preds.cpu().numpy()
def extract_labeled_adj(adj, idx_labeled, device):
    """
    adj: scipy sparse (N, N)
    idx_labeled: list of indices
    return: torch.FloatTensor (M, M)
    """
    adj_l = adj.tocsr()[idx_labeled, :][:, idx_labeled]
    adj_l = adj_l.toarray()
    adj_l = torch.tensor(adj_l, dtype=torch.float32, device=device)
    return adj_l
def graph_contrastive_loss(z, adj, temperature=0.5, eps=1e-8):
    """
    z: (M, d) normalized embeddings
    adj: (M, M) adjacency matrix (0/1 or weighted)
    """
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature
    sim.fill_diagonal_(-1e9)

    exp_sim = torch.exp(sim)

    pos = (exp_sim * adj).sum(dim=1)
    denom = exp_sim.sum(dim=1)

    loss = -torch.log((pos + eps) / (denom + eps))

    valid = adj.sum(dim=1) > 0
    return loss[valid].mean()

recon_loss_fn = nn.MSELoss()




def train(args, device):
    features, y, idx_labeled, idx_unlabeled, view, fea_num, num, adj = load_data(args, device)
    features = torch.tensor(features, dtype=torch.float32, device=device)
    adj_l = extract_labeled_adj(adj, idx_labeled, device)
    model = GraphContrastiveAEClassifier(input_dim=fea_num, latent_dim=args.nhid).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t1 = time.time()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    with (tqdm(total=args.epoch) as pbar):
        for ep in range(args.epoch):
            pbar.set_description('Training:')
            # z, x_recon, logits = model(torch.tensor(features, device=device))
            # recon_loss = F.mse_loss(x_recon, features)
            # z_l = z[idx_labeled]
            z, x_recon, logits = model(features[idx_labeled])
            recon_loss = F.mse_loss(x_recon, features[idx_labeled])
            z_l = z

            contrast_loss = graph_contrastive_loss(z_l, adj_l)

            # logits_l = logits[idx_labeled]
            logits_l = logits

            labels_l = torch.tensor(y[idx_labeled], dtype=torch.float32, device=device)
            cls_loss = bce_loss_fn(logits_l, labels_l)
            loss = args.beta * recon_loss + args.alpha * contrast_loss + cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # f_loss.write(str(loss_train.item()) + '\n')
            pbar.update(1)
    model.eval()
    with torch.no_grad():
        z, x_recon, logits = model(features)

        logits_test = logits[idx_unlabeled]
        y_test = y[idx_unlabeled]
        metrics_test = evaluate_binary_metrics(logits_test, y_test)
        print("Test")
        print(metrics_test)
        ACC, F1, AUC, Sensitivity, Specificity = metrics_test
    save_results_txt(
        "./results.txt",
        ACC,
        F1,
        AUC,
        Sensitivity,
        Specificity,
        args.alpha,
        args.beta,
        args.nhid
    )

