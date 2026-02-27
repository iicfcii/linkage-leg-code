import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from linkage_leg.dimensions import plot


class Plotter:
    def __init__(
        self,
        q, p, c, bbox,
        indices=None,
        on_plotted=None,
        on_design_changed=None,
    ):
        self.q = q.detach().cpu().numpy()
        self.p = {}
        for k, v in p.items():
            self.p[k] = v.detach().cpu().numpy()
        self.c = [
            _c[:4] + [_c[4].detach().cpu().numpy(), _c[5].detach().cpu().numpy()]
            for _c in c
        ]

        self.bbox = bbox
        if indices is None:
            self.indices = np.arange(self.q.shape[0])
        else:
            self.indices = indices.detach().cpu().numpy()
        self.on_plotted = on_plotted
        self.on_design_changed = on_design_changed

        self.n_designs = self.q.shape[0]
        self.n_q_combs = self.q.shape[1]

        self.d_index = 0
        self.q_index = 0
        self.label_dimensions = False

        self.fig_ctrl, self.axes_ctrl = plt.subplots(
            3, 1, num="ctrl", figsize=(6, 6),
        )
        self.fig_ctrl.subplots_adjust(
            left=0.2, right=0.8, top=0.9, bottom=0.1,
        )

        self.design_slider = Slider(
            self.axes_ctrl[1],
            "d",
            valmin=0,
            valmax=self.n_designs - 1,
            valinit=0,
            valstep=list(range(self.n_designs)),
            orientation="horizontal",
            initcolor="none",
        )
        self.design_slider.on_changed(self.on_design_slider_changed)

        self.label_slider = Slider(
            self.axes_ctrl[2],
            "l",
            valmin=0,
            valmax=1,
            valinit=0,
            valstep=[0, 1],
            orientation="horizontal",
            initcolor="none",
        )
        self.label_slider.on_changed(self.on_label_slider_changed)

        self.q_slider = Slider(
            self.axes_ctrl[0],
            "q",
            valmin=0,
            valmax=self.n_q_combs - 1,
            valinit=self.q_index,
            valstep=list(range(self.n_q_combs)),
            orientation="horizontal",
            initcolor="none",
        )
        self.q_slider.on_changed(self.on_q_slider_changed)

        self.fig, self.ax = plt.subplots(num="design", figsize=(6, 6))
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.on_design_slider_changed(self.d_index)

    def on_design_slider_changed(self, val):
        self.d_index = self.indices[val]
        self.draw()

        if self.on_design_changed is not None:
            self.on_design_changed(self.d_index, self.q_index)

    def on_label_slider_changed(self, val):
        if val == 0:
            self.label_dimensions = False
        else:
            self.label_dimensions = True
        self.draw()

    def on_q_slider_changed(self, val):
        self.q_index = val
        self.draw()

    def draw(self):
        plt.sca(self.ax)
        plt.cla()
        plot(
            self.p, self.c,
            self.d_index, self.q_index,
            label_dimensions=self.label_dimensions,
        )
        if self.on_plotted is not None:
            self.on_plotted(self.d_index, self.q_index)
        plt.xlim(self.bbox[0], self.bbox[0] + self.bbox[2])
        plt.ylim(self.bbox[1], self.bbox[1] + self.bbox[3])
        plt.draw()
