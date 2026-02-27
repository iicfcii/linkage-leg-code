import os
import time
from abc import ABC, abstractmethod
import multiprocessing as mp
import torch
import torch.optim as optim
import numpy as np
from linkage_leg.plotter import Plotter


torch.set_float32_matmul_precision("high")


class Design(ABC):
    @staticmethod
    @abstractmethod
    def plans():
        raise NotImplementedError

    def __init__(self, plan_index, seed=None, device=None):
        self.seed = seed
        torch.manual_seed(seed)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.plan_index = plan_index
        self.params = []
        self.plotter_bbox = (-100, -100, 200, 200)

    @abstractmethod
    def _eval(self):
        raise NotImplementedError

    def eval(self):
        self.loss, self.q, self.p, self.c = self._eval()
        return self.loss

    def plot(self):
        self.plotter = Plotter(
            self.q, self.p, self.c,
            self.plotter_bbox,
            indices=torch.argsort(self.loss),
            on_plotted=self._on_plotted,
            on_design_changed=self._on_design_changed,
        )

    def _on_plotted(self, d_index, q_index):
        pass

    def _on_design_changed(self, d_index, q_index):
        pass


def optimize(
    design,
    lr=1, lr_schedule="cosine",
    n_steps=10000,
    log_interval=1000,
    id=None,
    callback=None,
):
    optimizer = optim.Adam(design.params, lr=lr)

    if os.name != "nt":
        torch.compiler.reset()
        eval = torch.compile(design.eval)
    else:
        # Not using torch.compile on Windows because it is not well supported.
        # torch.compile can significantly speed up the optimization.
        eval = design.eval

    t0 = time.time()
    for i in range(n_steps + 1):
        loss = eval()
        loss_mean = torch.mean(loss)

        if callback is not None:
            callback(i, design, optimizer)

        if i > 0 and i % log_interval == 0:
            sps = i / (time.time() - t0)
            loss_min = torch.amin(loss.detach())
            print(
                f"id: {id}, "
                f"iter: {i}, "
                f"sps: {sps:.2f}, "
                f"loss_mean: {loss_mean:.2f}, "
                f"loss_min: {loss_min:.2f}"
            )

        if i < n_steps:
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            if lr_schedule == "linear":
                optimizer.param_groups[0]["lr"] = (1 - i / n_steps) * lr
            elif lr_schedule == "cosine":
                optimizer.param_groups[0]["lr"] = (
                    np.cos(i / n_steps * np.pi) + 1
                ) / 2 * lr
            else:
                pass


def save(design, path, name="design"):
    os.makedirs(path, exist_ok=True)

    timestamp = int(time.time())
    loss_min = torch.amin(design.loss.detach()).item()
    data = {
        "timestamp": timestamp,
        "design": design,
    }

    name = f"{timestamp}_{name}_{design.plan_index}_{loss_min:.4f}.pt"
    path = os.path.join(path, name)
    torch.save(data, path)


def load(path):
    return torch.load(path, weights_only=False)["design"]


def _sweep_fun(Design, plan_index, name, optimize_kwargs):
    design = Design(plan_index)
    optimize(design, id=plan_index, **optimize_kwargs)
    save(design, "logs", name)


def sweep(Design, name="design", processes=1, optimize_kwargs={}):
    if processes > 1:
        num_plans = len(Design.plans())
        args = [
            (Design, i, name, optimize_kwargs)
            for i in range(num_plans)
        ]
        with mp.Pool(2) as p:
            p.starmap(_sweep_fun, args, chunksize=1)
    else:
        for plan_index in range(len(Design.plans())):
            _sweep_fun(Design, plan_index, name, optimize_kwargs)
