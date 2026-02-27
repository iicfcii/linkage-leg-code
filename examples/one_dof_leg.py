import sys
import torch
from torch.func import jacrev
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import linkage_leg.topology as topology
import linkage_leg.dimensions as dimensions
import linkage_leg.designer as designer


class OneDoFLegDesign(designer.Design):
    def plans():
        plans = topology.load()

        filtered_plans = []
        for plan in plans:
            g = topology.gen_graph(plan)
            n_motors = len([
                e for e in g.edges
                if g[e[0]][e[1]]["type"] == "m"
            ])
            n_links = len(g.nodes)
            n_links_to_output = nx.shortest_path_length(
                g, list(g.nodes)[0], list(g.nodes)[-1]
            )
            if (
                n_motors == 1 and
                n_links <= 6 and
                n_links_to_output >= 2
            ):
                filtered_plans.append(plan)
        return filtered_plans

    def __init__(self, plan_index, seed=0):
        super().__init__(plan_index, seed=seed)
        self.plotter_bbox = (-150, -200, 300, 300)

        self.plan = OneDoFLegDesign.plans()[self.plan_index]
        self.g = topology.gen_graph(self.plan)
        self.c_empty = dimensions.gen_constraints(self.plan)
        self.origin_key = dimensions.origin_key
        self.output_key = dimensions.get_output_key(self.c_empty)

        self.n_designs = 1000

        self.cos_max = torch.tensor(0.8).to(self.device)
        self.joint_clearance_min = torch.tensor(20.0).to(self.device)
        self.foot_clearance_min = torch.tensor(20.0).to(self.device)
        self.half_width_max = torch.tensor(75.0).to(self.device)

        self.weights = torch.tensor([
            1, 0.001, 100, 1, 1, 1,
        ]).to(self.device)

        self.p0 = {}
        for key in dimensions.get_point_keys(self.c_empty):
            p = torch.zeros([self.n_designs, 2]).to(self.device)
            p[:, 0].uniform_(-150, 150)
            p[:, 1].uniform_(-200, 100)
            p.requires_grad_(True)
            self.params.append(p)
            self.p0[key] = p

        self.q = torch.linspace(
            -np.pi, np.pi, 25 + 1,
        )[:-1].expand(self.n_designs, -1).unsqueeze(-1).to(self.device)

        self.p_output_d = torch.zeros([12, 2])
        self.p_output_d[:-1, 0] = torch.linspace(
            -50, 50, self.p_output_d.shape[0] - 1,
        )
        self.p_output_d[:, 1] = -150
        self.p_output_d[-1] = torch.tensor([0, -150 + 25])
        self.p_output_d = self.p_output_d.to(self.device)

        self.points_of_links = []
        for n in self.g.nodes:
            points = list(self.g.edges(n))
            if self.g.nodes[n]["type"] == "g":
                points.append(self.origin_key)
            if self.g.nodes[n]["type"] == "o":
                points.append(self.output_key)
            points = [tuple(sorted(list(point))) for point in points]
            self.points_of_links.append(points)

        self.joints_of_links = []
        for n in self.g.nodes:
            if self.g.nodes[n]["type"] == "g":
                continue
            joints = list(self.g.edges(n))
            joints = [tuple(sorted(list(joint))) for joint in joints]
            self.joints_of_links.append(joints)

        self.p_non_output_keys = [
            k
            for k in self.p0.keys()
            if k != self.output_key
        ]

    def _eval(self):
        c = dimensions.populate(self.p0, self.c_empty)

        p, cos_theta, cos_theta_p, cos_mu = dimensions.fk(
            self.q, self.p0, c,
        )
        if cos_theta is None:
            cos = torch.zeros(self.n_designs, 1, 3, 1, device=self.device)
        else:
            cos = torch.stack([cos_theta, cos_theta_p, cos_mu], dim=-2)

        # for each desired output point, use the smallest distance to the actual output points.
        output_error = torch.amin(
            torch.linalg.norm(
                (
                    self.p_output_d.unsqueeze(-2) -
                    p[self.output_key].unsqueeze(-3)
                ),
                dim=-1,
            ),
            dim=-1,
        )
        loss_output_error = torch.mean(output_error, dim=-1)

        centroid_link_length = []
        for points_of_link in self.points_of_links:
            _p = torch.stack(
                [self.p0[k] for k in points_of_link],
                dim=1,
            )
            # sum of distances to centroid
            centroid_link_length.append(torch.sum(
                torch.linalg.norm(
                    _p - torch.mean(_p, dim=1, keepdim=True),
                    dim=-1,
                ),
                dim=1,
            ))
        centroid_link_length = torch.stack(centroid_link_length, dim=0).T
        total_link_length = torch.sum(centroid_link_length, dim=-1)
        loss_total_link_length = total_link_length

        cos = torch.clamp(cos - self.cos_max, 0)
        cos = torch.sum(cos, dim=(-1, -2))
        loss_cos = torch.mean(cos, dim=-1)

        joint_clearance = []
        for joints_of_link in self.joints_of_links:
            _p = torch.stack(
                [self.p0[k] for k in joints_of_link],
                dim=1,
            )
            length = torch.linalg.norm(
                _p.unsqueeze(2) - _p.unsqueeze(1),
                dim=-1,
            )
            offset = torch.eye(
                length.shape[-1], device=self.device,
            ) * self.joint_clearance_min
            joint_clearance.append(torch.amin(length + offset, dim=(-1, -2)))
        joint_clearance = torch.stack(joint_clearance, dim=0).T
        joint_clearance = torch.clamp(
            -joint_clearance - -self.joint_clearance_min, 0,
        )
        loss_joint_clearance = torch.sum(joint_clearance, dim=-1)

        p_non_foot = torch.stack(
            [p[k] for k in self.p_non_output_keys],
            dim=-2,
        )
        p_output = p[self.output_key]
        y_non_foot = p_non_foot[:, :, :, 1] - p_output[:, :, [1]]
        foot_clearance = torch.clamp(-y_non_foot - -self.foot_clearance_min, 0)
        foot_clearance = torch.sum(foot_clearance, dim=-1)
        loss_foot_clearance = torch.mean(foot_clearance, dim=-1)

        x_all = torch.cat(
            [p_non_foot[:, :, :, 0], p_output[:, :, [0]]],
            dim=-1,
        )
        half_width = torch.clamp(
            torch.abs(x_all) - self.half_width_max, 0,
        )
        half_width = torch.sum(half_width, dim=-1)
        loss_half_width = torch.mean(half_width, dim=-1)

        loss_itemized = torch.stack(
            [
                loss_output_error,
                loss_total_link_length,
                loss_cos,
                loss_joint_clearance,
                loss_foot_clearance,
                loss_half_width,
            ],
            dim=1,
        )
        self.loss_weighted = self.weights * loss_itemized
        loss = torch.sum(self.weights * loss_itemized, dim=1)

        return loss, self.q, p, c

    def _on_plotted(self, d_index, q_index):
        p_output = self.p[self.output_key][d_index].detach().cpu().numpy()
        p_output_d = self.p_output_d.detach().cpu().numpy()

        plt.plot(p_output[:, 0], p_output[:, 1], '.-b', lw=1)
        plt.plot(p_output_d[:, 0], p_output_d[:, 1], '.g', lw=1)

    def _on_design_changed(self, d_index, q_index):
        print(
            f"design: {d_index}, "
            f"l: {self.loss[d_index]:.4f}, "
            f"l_fe: {self.loss_weighted[d_index][0]:.4f}, "
            f"l_tll: {self.loss_weighted[d_index][1]:.4f}, "
            f"l_cos: {self.loss_weighted[d_index][2]:.4f}, "
            f"l_jc: {self.loss_weighted[d_index][3]:.4f}, "
            f"l_fc: {self.loss_weighted[d_index][4]:.4f}, "
            f"l_hw: {self.loss_weighted[d_index][5]:.4f}, "
        )


def main():
    if sys.argv[1] == "t":
        plan_index = int(sys.argv[2])
        design = OneDoFLegDesign(plan_index)
        design.eval()
        design.plot()
        plt.show()

    if sys.argv[1] == "o":
        plan_index = int(sys.argv[2])
        design = OneDoFLegDesign(plan_index)
        designer.optimize(design, id=plan_index, n_steps=10000)
        designer.save(design, "logs", name="one_dof_leg")

    if sys.argv[1] == "s":
        designer.sweep(
            OneDoFLegDesign,
            name="one_dof_leg",
            processes=1,
            optimize_kwargs={"n_steps": 10000},
        )

    if sys.argv[1] == "p":
        path = sys.argv[2]
        design = designer.load(path)
        design.plot()
        plt.show()


if __name__ == "__main__":
    main()
