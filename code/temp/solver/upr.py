import numpy as np
import torch
import torch.nn.functional as F

from temp.solver.utils import fix_var, repair, set_starts, unfix_var, solve_inst

EVIDENCE_FUNCS = {
    "softplus": (lambda y: F.softplus(y)),
    "relu": (lambda y: F.relu(y)),
    "exp": (lambda y: torch.exp(torch.clamp(y, -10, 10))),
}


def get_predictions(logits, binary_mask):
    probs = torch.softmax(logits, axis=1)
    preds = probs[:, 1]
    probs = _to_numpy(probs)
    preds = _to_numpy(preds).squeeze()
    preds[binary_mask] = preds[binary_mask].round()
    return probs, preds


def get_uncertainty(logits, evidence_func_name: str = "softplus"):
    evidence = EVIDENCE_FUNCS[evidence_func_name](logits)
    alpha = evidence + 1
    uncertainty = logits.shape[1] / torch.sum(alpha, dim=1, keepdim=True)
    return uncertainty


def get_threshold(uncertainty: torch.Tensor, r_min: float = 0.4, r_max: float = 0.55):
    q = (r_min + r_max) / 2
    threshold = torch.quantile(uncertainty, q)
    r = (uncertainty <= threshold).float().mean()

    if r > r_max:
        threshold = torch.quantile(uncertainty, r_max)
        return threshold

    if r < r_min:
        threshold = torch.quantile(uncertainty, r_min)
        return threshold

    return threshold


def get_confident_idx(indices, uncertainty, threshold):
    confident_mask = uncertainty <= threshold
    confident_idx = indices[confident_mask]
    return confident_idx.sort()[0]


def solve(inst, prediction, uncertainty, indices, max_iter):
    
    threshold = get_threshold(uncertainty)
    min_q = sum(uncertainty <= threshold) / len(uncertainty)
    max_q = min(min_q * 1.5, 1.0)
    dq = (max_q - min_q) / (max_iter - 1)

    threshold = np.quantile(uncertainty, max_q)
    conf_idxs = get_confident_idx(indices, uncertainty, threshold)
    conf_vals = prediction[conf_idxs]
    bounds = fix_var(inst, conf_idxs, conf_vals)
    print(max_q, threshold)
    
    fixed = set(conf_idxs.tolist())
    freed = set(repair(inst, fixed, bounds))
    print(len(freed))
    for i in range(1, max_iter):
        sol = solve_inst(inst)
        q = max_q - dq * i
        threshold = np.quantile(uncertainty, q)
        conf_idxs = get_confident_idx(indices, uncertainty, threshold)
        to_unfix = list(fixed - set(conf_idxs.tolist()))
        to_unfix = [i for i in to_unfix if i not in freed]
        print(q, threshold, len(to_unfix))
        print("^"*78)
        unfix_var(inst, to_unfix, [bounds[i] for i in to_unfix])
        starts = {i: sol[i] for i in to_unfix}
        starts.update({i: sol[i] for i in freed})
        set_starts(inst, starts)
    unfix_var(inst, bounds.keys(), bounds.values())
    sol = solve_inst(inst)
    return sol


def _to_numpy(tensor_obj):
    return tensor_obj.cpu().detach().numpy()
