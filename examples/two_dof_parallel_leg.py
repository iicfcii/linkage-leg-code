import sys
import torch
from torch.func import jacrev
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import linkage_leg.topology as topology
import linkage_leg.dimensions as dimensions
import linkage_leg.designer as designer


class TwoDoFParallelLegDesign(designer.Design):
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
            n_ground_joints = len(g.edges(0))
            n_ground_motors = len([
                e for e in g.edges(0)
                if g[e[0]][e[1]]["type"] == "m"
            ])
            if (
                n_motors == 2 and
                n_links <= 7 and
                n_links_to_output >= 2 and
                n_ground_motors == 2 and
                n_ground_joints == 2
            ):
                filtered_plans.append(plan)
        return filtered_plans

    def __init__(self, plan_index, seed=0):
        super().__init__(plan_index, seed=seed)
        self.plotter_bbox = (-150, -200, 300, 300)

        self.plan = TwoDoFParallelLegDesign.plans()[self.plan_index]
        self.g = topology.gen_graph(self.plan)
        self.c_empty = dimensions.gen_constraints(self.plan)
        self.origin_key = dimensions.origin_key
        self.output_key = dimensions.get_output_key(self.c_empty)
        self.motor0_key, self.motor1_key = [
            e
            for e in self.g.edges
            if 0 in e and self.g[e[0]][e[1]]["type"] == "m"
        ]

        self.n_designs = 1000

        self.cos_max = torch.tensor(0.8).to(self.device)
        self.cond_min = torch.tensor(0.05).to(self.device)
        self.joint_clearance_min = torch.tensor(20.0).to(self.device)
        self.motor_clearance_min = torch.tensor(20.0).to(self.device)
        self.foot_clearance_slope = torch.tan(
            torch.tensor(1.0),
        ).to(self.device)
        self.width_max = torch.tensor(150.0).to(self.device)

        self.weights = torch.tensor([
            1, 10, 0.001, 100, 100, 1, 1, 1, 1,
        ]).to(self.device)

        self.p0 = {}
        for key in dimensions.get_point_keys(self.c_empty):
            p = torch.zeros([self.n_designs, 2]).to(self.device)
            p[:, 0].uniform_(-150, 150)
            p[:, 1].uniform_(-200, 100)
            p.requires_grad_(True)
            self.params.append(p)
            self.p0[key] = p

        grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 5) - np.pi / 2,
            torch.linspace(150.0, 50.0, 5),
            indexing="ij",
        )
        polar = torch.stack([axis.flatten() for axis in grid]).T

        self.p_output_d = torch.zeros([polar.shape[0], 2]).to(self.device)
        self.p_output_d[:, 0] = polar[:, 1] * torch.cos(polar[:, 0])
        self.p_output_d[:, 1] = polar[:, 1] * torch.sin(polar[:, 0])

        jac = torch.zeros(self.n_designs, 2, 2).to(self.device)
        jac[:, 0, :].uniform_(-1, 1)
        jac[:, 1, :].uniform_(-100, 100)
        self.jac_scale = torch.tensor(
            [100.0, 1.0],
        ).expand(2, -1).T.to(self.device)
        self.jac_scaled = jac * self.jac_scale
        self.jac_scaled.requires_grad_(True)
        self.params.append(self.jac_scaled)

        q_res = torch.zeros(self.n_designs, *polar.shape).to(self.device)
        q_res.uniform_(-0.1, 0.1)
        self.q_res_scale = 1000
        self.q_res_scaled = q_res * self.q_res_scale
        self.q_res_scaled.requires_grad_(True)
        self.params.append(self.q_res_scaled)

        self.polar = polar.to(self.device)

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

        self.p_all_keys = [k for k in self.p0.keys()]

    def _eval(self):
        c = dimensions.populate(self.p0, self.c_empty)

        jac_inv = torch.linalg.inv(
            (
                self.jac_scaled +
                torch.eye(2, device=self.device) * dimensions.eps * 100
            ) / self.jac_scale,
        )
        q = torch.matmul(
            jac_inv.unsqueeze(1), self.polar.unsqueeze(-1),
        ).squeeze(-1)
        q = q - torch.mean(q, dim=1, keepdim=True)
        q_res = self.q_res_scaled / self.q_res_scale
        q = q + q_res

        def fk(q):
            p, cos_theta, cos_theta_p, cos_mu = dimensions.fk(
                q, self.p0, c,
            )
            if cos_theta is None:
                cos = torch.zeros(self.n_designs, 1, 3, 1, device=self.device)
            else:
                cos = torch.stack([cos_theta, cos_theta_p, cos_mu], dim=-2)

            p_output = p[self.output_key]
            polar = torch.stack(
                [
                    torch.atan2(
                        p_output[:, :, 1],
                        p_output[:, :, 0] + dimensions.eps,
                    ),
                    torch.linalg.norm(p_output, dim=-1),
                ],
                dim=-1,
            )
            return torch.sum(polar, dim=(0, 1)), (p, cos)
        jac, (p, cos) = jacrev(fk, has_aux=True)(q)
        jac = jac.transpose(0, 1).transpose(1, 2)

        output_error = torch.linalg.norm(
            p[self.output_key] - self.p_output_d,
            dim=-1,
        )
        loss_output_error = torch.mean(output_error, dim=-1)

        loss_q_res = torch.mean(torch.var(q_res, dim=-2), dim=-1)

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

        jac_scaled = jac * self.jac_scale
        a = torch.matmul(jac_scaled, jac_scaled.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(
            a + torch.eye(2, device=self.device) * dimensions.eps * 100,
        )
        cond = eigvals[:, :, 0] / eigvals[:, :, 1]
        cond = torch.clamp(-cond - -self.cond_min, 0)
        loss_cond = torch.mean(cond, dim=-1)

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

        motor_clearance = torch.linalg.norm(
            self.p0[self.motor0_key] - self.p0[self.motor1_key], dim=-1
        )
        motor_clearance = torch.clamp(
            -motor_clearance - -self.motor_clearance_min, 0,
        )
        loss_motor_clearance = motor_clearance

        p_output = p[self.output_key]
        rot_y = -p_output
        rot_y = rot_y / torch.linalg.norm(rot_y, dim=-1, keepdim=True)
        rot_x = torch.stack([rot_y[:, :, 1], -rot_y[:, :, 0]], dim=-1)
        rot = torch.stack([rot_x, rot_y], dim=-2)
        p_all = torch.stack([p[k] for k in self.p_all_keys], dim=-2)
        p_all = p_all - p_output.unsqueeze(-2)
        p_all = torch.matmul(
            rot.unsqueeze(2), p_all.unsqueeze(-1),
        ).squeeze(-1)
        x_all = p_all[:, :, :, 0]
        y_all = p_all[:, :, :, 1]
        foot_clearance = torch.clamp(
            -y_all - -torch.abs(x_all) * self.foot_clearance_slope,
            0,
        )
        foot_clearance = torch.sum(foot_clearance, dim=-1)
        loss_foot_clearance = torch.mean(foot_clearance, dim=-1)

        width = torch.amax(x_all, dim=-1) - torch.amin(x_all, dim=-1)
        width = torch.clamp(width - self.width_max, 0)
        loss_width = torch.mean(width, dim=-1)

        loss_itemized = torch.stack(
            [
                loss_output_error,
                loss_q_res,
                loss_total_link_length,
                loss_cos,
                loss_cond,
                loss_joint_clearance,
                loss_motor_clearance,
                loss_foot_clearance,
                loss_width,
            ],
            dim=1,
        )
        self.loss_weighted = self.weights * loss_itemized
        loss = torch.sum(self.weights * loss_itemized, dim=1)

        return loss, q, p, c

    def _on_plotted(self, d_index, q_index):
        p_output = self.p[self.output_key][d_index].detach().cpu().numpy()
        p_output_d = self.p_output_d.detach().cpu().numpy()

        plt.plot(p_output[:, 0], p_output[:, 1], '.b', lw=1)
        plt.plot(p_output_d[:, 0], p_output_d[:, 1], '.g', lw=1)

    def _on_design_changed(self, d_index, q_index):
        print(
            f"design: {d_index}, "
            f"l: {self.loss[d_index]:.4f}, "
            f"l_fe: {self.loss_weighted[d_index][0]:.4f}, "
            f"l_qr: {self.loss_weighted[d_index][1]:.4f}, "
            f"l_tll: {self.loss_weighted[d_index][2]:.4f}, "
            f"l_cos: {self.loss_weighted[d_index][3]:.4f}, "
            f"l_cd: {self.loss_weighted[d_index][4]:.4f}, "
            f"l_jc: {self.loss_weighted[d_index][5]:.4f}, "
            f"l_mc: {self.loss_weighted[d_index][6]:.4f}, "
            f"l_fc: {self.loss_weighted[d_index][7]:.4f}, "
            f"l_w: {self.loss_weighted[d_index][8]:.4f}"
        )

        jac = self.jac_scaled[d_index] / self.jac_scale
        q = self.q[d_index]
        q_res = self.q_res_scaled[d_index] / self.q_res_scale

        jac = jac.detach().cpu().numpy()
        q_std = torch.std(q, dim=0).detach().cpu().numpy()
        q_min = torch.amin(q, dim=0).detach().cpu().numpy()
        q_max = torch.amax(q, dim=0).detach().cpu().numpy()
        q_res_std = torch.std(q_res, dim=0).detach().cpu().numpy()
        q_res_min = torch.amin(q_res, dim=0).detach().cpu().numpy()
        q_res_max = torch.amax(q_res, dim=0).detach().cpu().numpy()

        with np.printoptions(precision=4, suppress=True, floatmode="fixed"):
            print("jac: ")
            print(jac)
            print(f"q_std: {q_std}")
            print(f"q_min: {q_min}")
            print(f"q_max: {q_max}")
            print(f"q_res_std: {q_res_std}")
            print(f"q_res_min: {q_res_min}")
            print(f"q_res_max: {q_res_max}")


def main():
    if sys.argv[1] == "t":
        plan_index = int(sys.argv[2])
        design = TwoDoFParallelLegDesign(plan_index)
        design.eval()
        design.plot()
        plt.show()

    if sys.argv[1] == "o":
        plan_index = int(sys.argv[2])
        design = TwoDoFParallelLegDesign(plan_index)
        designer.optimize(design, id=plan_index, n_steps=10000)
        designer.save(design, "logs", name="two_dof_parallel_leg")

    if sys.argv[1] == "s":
        designer.sweep(
            TwoDoFParallelLegDesign,
            name="two_dof_parallel_leg",
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
