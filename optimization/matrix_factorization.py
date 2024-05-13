import torch
import tensorly as tl
import numpy as np

from scipy.optimize import minimize_scalar

def empirical_variational_bayes_matrix_factorization(Y, sigma2=None, H=None):
    Y = torch.tensor(Y, dtype=torch.float32)
    L, M = Y.shape
    if H is None:
        H = L

    alpha = L / M
    tau_upper_bound = 2.5129 * torch.sqrt(torch.tensor(alpha))
    _, s, _ = torch.svd(Y)
    #U = U[:, :H]
    s = s[:H]
    #V = V.t()[:H, :]

    residual = 0.
    if H < L:
        residual = torch.sum(torch.sum(Y ** 2) - torch.sum(s ** 2))

    if sigma2 is None:
        x_upper_bound = (1 + tau_upper_bound) * (1 + alpha / tau_upper_bound)
        eH_upper_bound = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
        upper_bound = (torch.sum(s**2) + residual) / (L * M)
        lower_bound = np.max([s[eH_upper_bound + 1]**2 / (M * x_upper_bound), torch.mean(s[eH_upper_bound + 1:]**2) / M])

        sigma2_opt = minimize_scalar(calculate_sigma2_loss, args=(L, M, s, residual, alpha, tau_upper_bound),
                                     bounds=(lower_bound.item(), upper_bound.item()), method='bounded')
        sigma2 = sigma2_opt.x

    threshold = torch.sqrt(M * sigma2 * (1 + tau_upper_bound) * (1 + alpha / tau_upper_bound))
    pos = torch.sum(s > threshold)
    if pos == 0:
        return torch.empty(0)

    d = torch.mul(s[:pos] / 2, \
    1 - (L + M) * sigma2 / s[:pos]**2 + torch.sqrt( \
    (1 - ((L + M) * sigma2) / s[:pos]**2)**2 - \
    (4 * L * M * sigma2**2) / s[:pos]**4))

    return torch.diag(d)

def calculate_sigma2_loss(sigma2, L, M, s, residual, alpha, tau_upper_bound):
    H = len(s)
    x_upper_bound = (1 + tau_upper_bound) * (1 + alpha / tau_upper_bound)
    x = s**2 / (M * sigma2)

    z1 = x[x > x_upper_bound]
    z2 = x[x <= x_upper_bound]
    tau_z1 = compute_tau(z1, alpha)

    term1 = torch.sum(z2 - torch.log(z2))
    term2 = torch.sum(z1 - tau_z1)
    term3 = torch.sum(torch.log((tau_z1 + 1) / z1))
    term4 = alpha * torch.sum(torch.log(tau_z1 / alpha + 1))

    obj = term1 + term2 + term3 + term4 + residual / (M * sigma2) + (L - H) * np.log(sigma2)
    return obj.item()

def compute_tau(x, alpha):
   return 0.5 * (x - (1 + alpha) + torch.sqrt((x - (1 + alpha))**2 - 4 * alpha))

def tucker_rank(layer):

  weight = layer.weight.data.cpu().numpy()

  mode_3_unfold = tl.base.unfold(weight, mode=0)
  mode_4_unfold = tl.base.unfold(weight, mode=1)


  rank_mode_3 = empirical_variational_bayes_matrix_factorization(mode_3_unfold)
  rank_mode_4 = empirical_variational_bayes_matrix_factorization(mode_4_unfold)


  estimated_rank_1 = rank_mode_3.shape[0]
  estimated_rank_2 = rank_mode_4.shape[1] if rank_mode_4.size != 0 else 0


  del mode_3_unfold, mode_4_unfold, rank_mode_3, rank_mode_4

  rounded_rank_1 = int(np.ceil(estimated_rank_1 / 16) * 16)
  rounded_rank_2 = int(np.ceil(estimated_rank_2 / 16) * 16)

  return [rounded_rank_1, rounded_rank_2]

def cp_rank(conv_layer):

  weight = conv_layer.weight.data.cpu().numpy()

  mode_3_unfold = tl.base.unfold(weight, mode=0)
  mode_4_unfold = tl.base.unfold(weight, mode=1)

  rank_mode_3 = empirical_variational_bayes_matrix_factorization(mode_3_unfold)
  rank_mode_4 = empirical_variational_bayes_matrix_factorization(mode_4_unfold)

  max_rank = max(rank_mode_3.shape[0], rank_mode_4.shape[0])
  rounded_rank = int(np.ceil(max_rank / 16) * 16)

  return rounded_rank
