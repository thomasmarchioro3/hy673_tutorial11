import torch

def get_schedules(beta_1, beta_T, n_T):
    """
    Linear scheduler. 
    Useful to pre-compute all the parameters (even fractions, square roots, etc).
    """

    beta_t = (beta_T - beta_1) * torch.arange(0, n_T + 1, dtype=torch.float32) / n_T + beta_1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrt_abar = torch.sqrt(alphabar_t)
    one_over_sqrt_a = 1 / torch.sqrt(alpha_t)

    sqrt_inv_abar = torch.sqrt(1 - alphabar_t)
    inv_abar_over_sqrt_inv_abar = (1 - alpha_t) / sqrt_inv_abar

    return {
        "alpha": alpha_t,  # \alpha_t
        "one_over_sqrt_a": one_over_sqrt_a,  # 1/\sqrt{\alpha_t}
        "sqrt_beta": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar": alphabar_t,  # \bar{\alpha_t}
        "sqrt_abar": sqrt_abar,  # \sqrt{\bar{\alpha_t}}
        "sqrt_inv_abar": sqrt_inv_abar,  # \sqrt{1-\bar{\alpha_t}}
        "inv_alpha_over_sqrt_inv_abar": inv_abar_over_sqrt_inv_abar,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
