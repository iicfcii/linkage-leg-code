import os
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

eps = 1e-6
origin_key = (0, 0)


def get_output_key(c):
    for _c in c:
        for e in _c[1:]:
            if -1 in e:
                return e


def get_point_keys(c):
    return set([_e for _c in c for _e in _c[1:]])


def gen_constraints(plan):
    def add_constraint(l, lp):
        if g.nodes[lp]["type"] != "l":
            return
        e_list = list(g.edges(lp))
        e = e_list[0]
        if len(e_list) == 1:
            ep_list = g.edges(e[1])
            for ep in ep_list:
                if set(ep) != set(e):
                    break
            if 0 in ep:
                ep = origin_key
            c.append(["m", (l, lp), e, ep])
        else:
            ep = e_list[1]
            c.append(["f", (l, lp), e, ep])

    c = []  # Constraints for the topology
    g = nx.Graph()  # Graph for the topology
    g.add_node(0, type="g")
    for i, op in enumerate(plan):
        if op[0] == "m":
            l0, la = op[1:]

            add_constraint(l0, la)

            g.add_node(l0, type="l")
            g.add_edge(l0, la, type="m")

            if i == len(plan) - 1:
                e_list = g.edges(la)
                if g.nodes[la]["type"] == "l":
                    for e in e_list:
                        if set(e) != set((l0, la)):
                            break
                else:
                    e = origin_key
                c.append(["m", (-1, l0), (l0, la), e])

                g.nodes[l0]["type"] = "o"
        else:
            l0, l1, la, lb = op[1:]

            for l, lp in zip([l0, l1], [la, lb]):
                add_constraint(l, lp)
            c.append(["f", (l0, l1), (l0, la), (l1, lb)])

            g.add_node(l0, type="l")
            g.add_node(l1, type="l")
            g.add_edge(l0, l1, type="j")
            g.add_edge(l0, la, type="j")
            g.add_edge(l1, lb, type="j")

            if i == len(plan) - 1:
                c.append(["f", (-1, l1), (l0, l1), (l1, lb)])

                g.nodes[l1]["type"] = "o"

    # Make sure the edge tuples are consistent by sorting them
    for _c in c:
        for i in range(1, len(_c)):
            _c[i] = tuple(sorted(_c[i]))

    return c


def populate(p0, c):
    c = deepcopy(c)

    for _c in c:
        if _c[0] != "f":
            continue

        if sum([
            1 for _cp in c
            if _c[2] in _cp[1:] and _c[3] in _cp[1:]
        ]) > 1:
            _c[0] = "l"

    for _c in c:
        if _c[0] == "m":
            v0 = p0[_c[1]] - p0[_c[2]]
            v1 = p0[_c[2]] - p0[_c[3]]
            _c.append(torch.linalg.norm(v0, dim=1))
            _c.append(
                torch.arctan2(v0[:, 1], v0[:, 0] + eps) -
                torch.arctan2(v1[:, 1], v1[:, 0] + eps)
            )
        else:
            v0 = p0[_c[1]] - p0[_c[2]]
            v1 = p0[_c[1]] - p0[_c[3]]
            _c.append(torch.linalg.norm(v0, dim=1))
            _c.append(torch.linalg.norm(v1, dim=1))
            _c.append(v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0])
    return c


def fk(q, p0, c):
    n_q = q.shape[1]
    p = {}
    for e in p0.keys():
        if 0 in e:
            p[e] = p0[e].unsqueeze(1).tile((1, n_q, 1))
    q_index = 0
    cos_theta = []
    cos_theta_p = []
    cos_mu = []

    for _c in c:
        if _c[0] == "m":
            _p0 = p[_c[2]]
            _p1 = p[_c[3]]
            l = _c[4].unsqueeze(1).tile((1, n_q))
            v10 = _p0 - _p1
            q0 = (
                _c[5].unsqueeze(1).tile((1, n_q)) +
                torch.arctan2(v10[:, :, 1], v10[:, :, 0] + eps)
            )
            _q = q[:, :, q_index]
            p[_c[1]] = torch.stack([
                l * torch.cos(_q + q0),
                l * torch.sin(_q + q0)
            ], dim=2) + _p0
            q_index += 1
        else:
            _p0 = p[_c[2]]
            _p1 = p[_c[3]]
            v01 = _p1 - _p0

            l0 = _c[4].unsqueeze(1).tile((1, n_q))
            l1 = _c[5].unsqueeze(1).tile((1, n_q))
            l2 = torch.linalg.norm(v01, dim=2)

            _cos_theta = (
                (l2**2 + l0**2 - l1**2) / (2 * l2 * l0 + eps)
            )
            if _c[0] != "l":
                _cos_theta_p = (
                    (l2**2 + l1**2 - l0**2) / (2 * l2 * l1 + eps)
                )
                cos_theta.append(_cos_theta)
                cos_theta_p.append(_cos_theta_p)

                _cos_mu = (
                    (l0**2 + l1**2 - l2**2) / (2 * l0 * l1 + eps)
                )
                cos_mu.append(_cos_mu)
            _cos_theta_clipped = torch.clip(_cos_theta, -1 + eps, 1 - eps)
            theta = (
                torch.sign(_c[6].unsqueeze(1).tile((1, n_q))) *
                torch.arccos(_cos_theta_clipped)
            )
            rot = torch.stack([
                torch.cos(theta), -torch.sin(theta),
                torch.sin(theta), torch.cos(theta)
            ], dim=2).reshape(-1, n_q, 2, 2)
            p[_c[1]] = (
                torch.matmul(rot, v01.unsqueeze(3)).squeeze(3) /
                (l2.unsqueeze(2) + eps) * l0.unsqueeze(2)
            ) + _p0

    if len(cos_theta) > 0:
        cos_theta = torch.stack(cos_theta, dim=2)
    else:
        cos_theta = None

    if len(cos_theta_p) > 0:
        cos_theta_p = torch.stack(cos_theta_p, dim=2)
    else:
        cos_theta_p = None

    if len(cos_mu) > 0:
        cos_mu = torch.stack(cos_mu, dim=2)
    else:
        cos_mu = None

    return p, cos_theta, cos_theta_p, cos_mu


def plot(p, c, d_index, q_index, label_dimensions=False, lw=1, ms=5, mew=1):
    _p = {}
    for k, v in p.items():
        _p[k] = v[d_index][q_index]
    p = _p
    c = [
        _c[:4] + [_c[4][d_index], _c[5][d_index]]
        for _c in c
    ]

    for e in p.keys():
        if 0 in e:
            if (0, 0) == e:
                plt.plot(*p[e], "sk", markersize=ms, markeredgewidth=mew)
            else:
                plt.plot(*p[e], "^k", markersize=ms, markeredgewidth=mew)
        elif -1 not in e:
            plt.plot(*p[e], ".k", markersize=ms, markeredgewidth=mew)
        else:
            plt.plot(*p[e], "xg", markersize=ms, markeredgewidth=mew)

    for _c in c:
        if _c[0] == "m":
            lk = np.array([p[_c[1]], p[_c[2]]])
            plt.plot(
                lk[:, 0], lk[:, 1],
                "r", zorder=2.1, lw=lw,
            )
        else:
            lk0 = np.array([p[_c[1]], p[_c[2]]])
            lk1 = np.array([p[_c[1]], p[_c[3]]])
            plt.plot(lk0[:, 0], lk0[:, 1], "k", lw=lw)
            plt.plot(lk1[:, 0], lk1[:, 1], "k", lw=lw)

            if _c[0] == "l":
                pg = np.array([p[_c[1]], p[_c[2]], p[_c[3]]])
                plt.gca().add_patch(plt.Polygon(pg, color="k", alpha=0.2, lw=0))

    if label_dimensions:
        for e in p.keys():
            if 0 in e:
                p0 = p[e]
                plt.text(*p0, f"({p0[0]:.4f}, {p0[1]:.4f})", color="m")

        for _c in c:
            if _c[0] == "m":
                lk = np.array([p[_c[1]], p[_c[2]]])
                ang = np.atan2(
                    lk[0, 1] - lk[1, 1],
                    lk[0, 0] - lk[1, 0],
                )
                p0 = np.mean(lk, axis=0)
                plt.text(*p0, f"{_c[4]:.4f} @ {ang:.4f}", color="m")
            else:
                lk0 = np.array([p[_c[1]], p[_c[2]]])
                lk1 = np.array([p[_c[1]], p[_c[3]]])
                p0 = np.mean(lk0, axis=0)
                p1 = np.mean(lk1, axis=0)
                plt.text(*p0, f"{_c[4]:.4f}", color="m")
                plt.text(*p1, f"{_c[5]:.4f}", color="m")

    plt.axis("scaled")
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
