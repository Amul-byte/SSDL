import torch
import torch.nn as nn
from tqdm import tqdm

class Scheduler:
    def __init__(self, sched_type, T, step, device):
        self.device = device
        t_vals = torch.arange(1, T + 1, step).to(torch.int)

        if sched_type == "cosine":
            def f(t):
                s = 0.008
                return torch.clamp(torch.cos(((t / T + s) / (1 + s)) * (torch.pi / 2)) ** 2 /
                                   torch.cos(torch.tensor((s / (1 + s)) * (torch.pi / 2))) ** 2,
                                   1e-10, 0.999)

            self.a_bar_t = f(t_vals)
            self.a_bar_t1 = f((t_vals - step).clamp(0, torch.inf))
            self.beta_t = 1 - (self.a_bar_t / self.a_bar_t1)
            self.beta_t = torch.clamp(self.beta_t, 1e-10, 0.999)
            self.a_t = 1 - self.beta_t
        else:  # Linear
            self.beta_t = torch.linspace(1e-4, 0.02, T)
            self.beta_t = self.beta_t[::step]
            self.a_t = 1 - self.beta_t
            self.a_bar_t = torch.stack([torch.prod(self.a_t[:i]) for i in range(1, (T // step) + 1)])
            self.a_bar_t1 = torch.stack([torch.prod(self.a_t[:i]) for i in range(1, (T // step) + 1)])

        self.sqrt_a_t = torch.sqrt(self.a_t)
        self.sqrt_a_bar_t = torch.sqrt(self.a_bar_t)
        self.sqrt_1_minus_a_bar_t = torch.sqrt(1 - self.a_bar_t)
        self.sqrt_a_bar_t1 = torch.sqrt(self.a_bar_t1)
        self.beta_tilde_t = ((1 - self.a_bar_t1) / (1 - self.a_bar_t)) * self.beta_t

        self.to_device()

    def to_device(self):
        self.beta_t = self.beta_t.to(self.device)
        self.a_t = self.a_t.to(self.device)
        self.a_bar_t = self.a_bar_t.to(self.device)
        self.a_bar_t1 = self.a_bar_t1.to(self.device)
        self.sqrt_a_t = self.sqrt_a_t.to(self.device)
        self.sqrt_a_bar_t = self.sqrt_a_bar_t.to(self.device)
        self.sqrt_1_minus_a_bar_t = self.sqrt_1_minus_a_bar_t.to(self.device)
        self.sqrt_a_bar_t1 = self.sqrt_a_bar_t1.to(self.device)
        self.beta_tilde_t = self.beta_tilde_t.to(self.device)

        self.beta_t = self.beta_t.unsqueeze(-1).unsqueeze(-1)
        self.a_t = self.a_t.unsqueeze(-1).unsqueeze(-1)
        self.a_bar_t = self.a_bar_t.unsqueeze(-1).unsqueeze(-1)
        self.a_bar_t1 = self.a_bar_t1.unsqueeze(-1).unsqueeze(-1)
        self.sqrt_a_t = self.sqrt_a_t.unsqueeze(-1).unsqueeze(-1)
        self.sqrt_a_bar_t = self.sqrt_a_bar_t.unsqueeze(-1).unsqueeze(-1)
        self.sqrt_1_minus_a_bar_t = self.sqrt_1_minus_a_bar_t.unsqueeze(-1).unsqueeze(-1)
        self.sqrt_a_bar_t1 = self.sqrt_a_bar_t1.unsqueeze(-1).unsqueeze(-1)
        self.beta_tilde_t = self.beta_tilde_t.unsqueeze(-1).unsqueeze(-1)

    def sample_a_t(self, t):
        return self.a_t[t - 1]

    def sample_beta_t(self, t):
        return self.beta_t[t - 1]

    def sample_a_bar_t(self, t):
        return self.a_bar_t[t - 1]

    def sample_a_bar_t1(self, t):
        return self.a_bar_t1[t - 1]

    def sample_sqrt_a_t(self, t):
        return self.sqrt_a_t[t - 1]

    def sample_sqrt_a_bar_t(self, t):
        return self.sqrt_a_bar_t[t - 1]

    def sample_sqrt_1_minus_a_bar_t(self, t):
        return self.sqrt_1_minus_a_bar_t[t - 1]

    def sample_sqrt_a_bar_t1(self, t):
        return self.sqrt_a_bar_t1[t - 1]

    def sample_beta_tilde_t(self, t):
        return self.beta_tilde_t[t - 1]

class DiffusionProcess:
    def __init__(self, scheduler, device='cpu', ddim_scale=0.5):
        """
        Initialize the DiffusionProcess class with a scheduler.

        Args:
            scheduler (DDIM_Scheduler): Scheduler for the beta noise term.
            device (str): Device to run the diffusion process on ('cpu' or 'cuda').
            ddim_scale (float): Scale between DDIM (0) and DDPM (1) sampling.
        """
        self.scheduler = scheduler
        self.device = device
        self.ddim_scale = ddim_scale

    def add_noise(self, x0, t):
        """
        Add noise to the clean input data.

        Args:
            x0 (torch.Tensor): Original input data.
            t (torch.Tensor): Timesteps.

        Returns:
            torch.Tensor: Noisy data.
            torch.Tensor: Noise added.
        """
        noise = torch.randn_like(x0)
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t).to(self.device)
        sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(self.device)

        xt = sqrt_a_bar_t * x0 + sqrt_1_minus_a_bar_t * noise
        return xt, noise

    def denoise(self, xt, context, t, label, model, predict_noise=True):
        """
        Denoise the input data using the model.

        Args:
            xt (torch.Tensor): Noisy data.
            context (torch.Tensor): Context or conditioning input.
            t (torch.Tensor): Timestep.
            model (torch.nn.Module): Model used for denoising.
            predict_noise (bool): Whether the model predicts noise or directly predicts the denoised data.

        Returns:
            torch.Tensor: Denoised data.
        """
        if predict_noise:
            pred_noise = model(xt, context, t)
            sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t).to(self.device)
            sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(self.device)
            x0_pred = (xt - sqrt_1_minus_a_bar_t * pred_noise) / sqrt_a_bar_t
        else:
            x0_pred = model(xt, context, t, label)
        return x0_pred

    @torch.no_grad()
    def sample(self, model, context, xt, label, steps, predict_noise=True):
        """
        Sample data using the diffusion model.

        Args:
            model (torch.nn.Module): Model used for sampling.
            context (torch.Tensor): Context or conditioning input.
            xt (torch.Tensor): Initial noisy data.
            steps (int): Number of steps.
            predict_noise (bool): Whether the model predicts noise or directly predicts the denoised data.

        Returns:
            torch.Tensor: Sampled data.
        """
        for step in tqdm(reversed(range(steps)), desc="Sampling"):
            t = torch.tensor([step], device=self.device, dtype=torch.long).expand(xt.size(0))
            x0_pred = self.denoise(xt, context, t, label, model, predict_noise=predict_noise)

            # DDIM Update: Remove noise progressively without re-adding stochastic noise
            xt = self.update_ddim(x0_pred, xt, t)
        return xt

    def update_ddim(self, x0_pred, xt, t):
        """
        Perform a deterministic DDIM update step for more deterministic sampling.

        Args:
            x0_pred (torch.Tensor): Predicted clean data.
            xt (torch.Tensor): Noisy input data.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Updated noisy data for the next step.
        """
        # DDIM logic here is deterministic. No noise is added; we just perform the reverse update.
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t).to(self.device)
        xt_next = sqrt_a_bar_t * x0_pred + torch.sqrt(1 - self.scheduler.sample_a_bar_t(t)) * xt
        return xt_next

    @torch.no_grad()
    def generate(self, model, context, shape, label, steps, predict_noise=True):
        """
        Generate samples using the DDIM sampling process.

        Args:
            model (torch.nn.Module): Model used for generation.
            context (torch.Tensor): Context or conditioning input.
            shape (tuple): Shape of the generated data.
            steps (int): Number of steps.
            predict_noise (bool): Whether the model predicts noise or directly predicts the denoised data.

        Returns:
            torch.Tensor: Generated samples.
        """
        xt = torch.randn(shape, device=self.device)
        generated_samples = self.sample(model, context, xt, label, steps, predict_noise=predict_noise)
        return generated_samples