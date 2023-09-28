import torch
import numpy as np
import torch.nn.functional as F


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if len(s.shape) == 0:
        return s * ts
    
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        if isinstance(x0, list):
            # n = np.random.choice(list(range(1, self.N + 1)), (len(x0[0]),))
            n = np.array([1000, 1000,1000, 1000])
            eps = [torch.randn_like(tensor) for tensor in x0]
            xn = [stp(self.cum_alphas[n] ** 0.5, tensor) + stp(self.cum_betas[n] ** 0.5, _eps) for tensor, _eps in zip(x0, eps)]
            return torch.tensor(n), eps, xn
        else:
            n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
            eps = torch.randn_like(x0)
            xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
            return torch.tensor(n), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'





def LSimple_T2I(img, clip_img, text, data_type, nnet, schedule, device, config, mask=None):
    r"""
    文到图loss
    """
    n, eps, xn = schedule.sample([img, clip_img])  # n in {1, ..., 1000}
    target, clip_img_eps = eps # img_eps, clip_img_eps, target = eps
    img_eps, clip_img_eps = eps
    img_n, clip_img_n = xn
    n = n.to(device)
    clip_img_n=clip_img_n.to(torch.float32)
    t_text=torch.zeros_like(n, device=device)
    data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type
    torch.save(img_n, 'girl2_img.pt')
    exit()
    img_out, clip_img_out, text_out = nnet(img_n, clip_img_n, text, t_img=n, t_text=t_text, data_type=data_type)
    
    img_out, img_out_prior = torch.chunk(img_out, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)
    
    mask = torch.chunk(mask, 2, dim=0)[0]
    # Compute instance loss
    aloss = F.mse_loss(img_out.float(), target.float(), reduction="none")
    aloss = ((aloss*mask).sum([1,2,3]) / mask.sum([1,2,3])).mean()

    # Compute prior loss
    prior_loss = F.mse_loss(img_out_prior.float(), target_prior.float(), reduction="mean")
    loss_img_clip =  F.mse_loss(clip_img_out.float(), clip_img_eps.float(), reduction="mean")
    text_out = torch.nn.functional.softplus(text_out).mean()
    # lora_img_out = torch.nn.functional.softplus(lora_img_out).mean()
    bloss =  1.2*aloss+  prior_loss +  loss_img_clip + 0. * text_out #+ 0. *lora_img_out
  
    return  bloss




def get_by_bool(lst, bool_tensor):
    bool_list = bool_tensor.tolist()
    return [elem for idx, elem in enumerate(lst) if bool_list[idx]]
