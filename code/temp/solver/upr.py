import numpy as np
import torch
import torch.nn.functional as F

from .utils import fix_var, repair, set_starts, unfix_var

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
    confident_idx = list(indices[confident_mask])
    return sorted(confident_idx)


def solve(inst):
    vs = inst.getVars()
    inst.optimize()
    return inst.getAttr("X", vs)


def solve(inst, prediction, uncertainty, indices, max_iter):

    threshold = get_threshold(uncertainty)
    conf_idxs = get_confident_idx(indices, uncertainty, prediction, threshold)
    conf_vals = prediction[conf_idxs]
    bounds = fix_var(inst, conf_idxs, conf_vals)

    min_q = sum(uncertainty <= threshold) / len(uncertainty)
    max_q = 1.0
    dq = (max_q - min_q) / (max_iter - 1)

    fixed = set(conf_idxs)
    freed = set(repair(inst, fixed, bounds))
    for i in range(1, max_iter):
        sol = solve(inst)
        q = max_q - dq * i
        threshold = np.quantile(uncertainty, q)
        conf_idxs = get_confident_idx(indices, uncertainty, prediction, threshold)
        to_unfix = list(fixed - set(conf_idxs))
        to_unfix = [i for i in to_unfix if i not in freed]
        unfix_var(inst, to_unfix, bounds)
        starts = {i: sol[i] for i in to_unfix}
        starts.update({i: sol[i] for i in freed})
        set_starts(inst, starts)
    return sol


def _to_numpy(tensor_obj):
    return tensor_obj.cpu().detach().numpy()
