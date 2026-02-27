import os
import time
from multiprocessing import Pool, Manager, Process, Event, cpu_count
from itertools import permutations
import numpy as np
import networkx as nx
import importlib.resources
from linkage_leg import resources

min_parallel_size = 10000
remove_subgraph_isomorphic_plans_batch_size = 1000
roughly_remove_isomorphic_plans_n_batches = 10
has_isomorphic_plan_batch_size = 5000
rng = np.random.default_rng(0)


def _gen_plan(n, g):
    plan = [g.nodes[n]["op"]]
    pre_ns = list(g.predecessors(n))
    while len(pre_ns) != 0:
        assert len(pre_ns) == 1
        pre_n = pre_ns[0]
        plan.insert(0, g.nodes[pre_n]["op"])
        pre_ns = list(g.predecessors(pre_n))
    return plan


def _expand_leaf(n, g, max_n_followers):
    def out_out_bounds(n_links, n_motors):
        return n_links > n_motors + 1 + max_n_followers

    n_links = g.nodes[n]["n_links"]
    n_motors = g.nodes[n]["n_motors"]
    for op in ["m", "f"]:
        if op == "m":
            if out_out_bounds(n_links + 1, n_motors + 1):
                continue

            l0 = n_links
            for la in range(n_links):
                na = len(g.nodes)
                g.add_node(
                    na,
                    op=("m", l0, la),
                    n_links=l0 + 1,
                    n_motors=n_motors + 1
                )
                g.add_edge(n, na)
        else:
            if out_out_bounds(n_links + 2, n_motors):
                continue

            l0 = n_links
            l1 = l0 + 1
            for la, lb in permutations(range(n_links), 2):
                na = len(g.nodes)
                g.add_node(
                    na,
                    op=("f", l0, l1, la, lb),
                    n_links=l1 + 1,
                    n_motors=n_motors
                )
                g.add_edge(n, na)


def enum_all_plans(max_n_steps, max_n_followers, return_graph=False):
    # Directed graph (tree) for enumerating all possible operation sequences.
    # Each operation is either adding one motorized link (m) or adding two follower links (f)
    g = nx.DiGraph()

    # First operation is always m
    g.add_node(0, op=("m", 1, 0), n_links=1 + 1, n_motors=1)
    print(f"step: 1, plans: 1")

    all_plans = []
    for i in range(2, max_n_steps + 2):
        leaf_ns = [
            n for n in g.nodes
            if nx.shortest_path_length(g, 0, n) == i - 2
        ]

        plans = []
        for n in leaf_ns:
            plans.append(_gen_plan(n, g))
        all_plans.append(plans)

        if i == max_n_steps + 1:
            break

        count = 0
        for n in leaf_ns:
            _expand_leaf(n, g, max_n_followers)
            count += g.out_degree(n)
            print(f"step: {i}, plans: {count}", end="\r")
        print("")

    if return_graph:
        return all_plans, g

    return all_plans


def gen_graph(plan):
    g = nx.Graph()  # Graph for the topology
    g.add_node(0, type="g")
    for i, op in enumerate(plan):
        if op[0] == "m":
            l0, la = op[1:]
            g.add_node(l0, type="l")
            g.add_edge(l0, la, type="m")
            if i == len(plan) - 1:
                g.nodes[l0]["type"] = "o"
        else:
            l0, l1, la, lb = op[1:]
            g.add_node(l0, type="l")
            g.add_node(l1, type="l")
            g.add_edge(l0, l1, type="j")
            g.add_edge(l0, la, type="j")
            g.add_edge(l1, lb, type="j")

            if i == len(plan) - 1:
                g.nodes[l1]["type"] = "o"
    return g


def _remove_subgraph_isomorphic_plans(plans, unique_plans, verbose=True):
    _plans = []
    for i, plan in enumerate(plans):
        iso = False
        for unique_plan in unique_plans:
            iso_matcher = nx.isomorphism.GraphMatcher(
                gen_graph(plan), gen_graph(unique_plan),
                node_match=lambda n0, n1: n0["type"] == n1["type"],
                edge_match=lambda e0, e1: e0["type"] == e1["type"]
            )
            if iso_matcher.subgraph_is_isomorphic():
                iso = True
        if not iso:
            _plans.append(plan)

        if verbose:
            print(
                "removing subgraph isomorphic plans: "
                f"{len(_plans)}/{i + 1}",
                end="\r",
            )
    if verbose:
        print("")
    return _plans


def _parallel_remove_subgraph_isomorphic_plans_func(args):
    plans, unique_plans = args
    return _remove_subgraph_isomorphic_plans(plans, unique_plans, verbose=False)


def _parallel_remove_subgraph_isomorphic_plans(plans, unique_plans):
    n_batches = len(plans) // remove_subgraph_isomorphic_plans_batch_size + 1
    batched_plans = [
        (
            plans[
                i * remove_subgraph_isomorphic_plans_batch_size:
                (i + 1) * remove_subgraph_isomorphic_plans_batch_size
            ],
            unique_plans,
        )
        for i in range(n_batches)
    ]
    _plans = []
    with Pool(min(cpu_count(), n_batches)) as p:
        for i, __plans in enumerate(p.imap(
            _parallel_remove_subgraph_isomorphic_plans_func,
            batched_plans,
            chunksize=1
        )):
            _plans += __plans
            count = np.sum([
                len(_batched_plans[0])
                for _batched_plans in batched_plans
            ][:i + 1])
            print(
                "removing subgraph isomorphic plans: "
                f"{len(_plans)}/{count}",
                end="\r",
            )
        print("")
    return _plans


def remove_subgraph_isomorphic_plans(plans, unique_plans):
    if len(plans) < min_parallel_size:
        return _remove_subgraph_isomorphic_plans(plans, unique_plans)
    else:
        return _parallel_remove_subgraph_isomorphic_plans(plans, unique_plans)


def has_isomorphic_plan(plan, plans):
    for _plan in plans:
        iso_matcher = nx.isomorphism.GraphMatcher(
            gen_graph(plan), gen_graph(_plan),
            node_match=lambda n0, n1: n0["type"] == n1["type"],
            edge_match=lambda e0, e1: e0["type"] == e1["type"]
        )
        if iso_matcher.is_isomorphic():
            return True
    return False


def parallel_has_isomorphic_plan_func(args):
    plan, plans = args
    return has_isomorphic_plan(plan, plans)


def parallel_has_isomorphic_plan(plan, plans):
    n_batches = len(plans) // has_isomorphic_plan_batch_size + 1
    batched_plans = [
        (
            plan,
            plans[
                i * has_isomorphic_plan_batch_size:
                (i + 1) * has_isomorphic_plan_batch_size
            ],
        )
        for i in range(n_batches)
    ]
    with Pool(min(cpu_count(), n_batches)) as p:
        for _has_isomorphic_plan in p.imap(
            parallel_has_isomorphic_plan_func,
            batched_plans,
            chunksize=1
        ):
            if _has_isomorphic_plan:
                return True
    return False


def remove_isomorphic_plans(plans):
    _plans = []
    for i, plan in enumerate(plans):
        if len(_plans) < min_parallel_size:
            if not has_isomorphic_plan(plan, _plans):
                _plans.append(plan)
        else:
            if not parallel_has_isomorphic_plan(plan, _plans):
                _plans.append(plan)
        print(f"removing isomorphic plans: {len(_plans)}/{i + 1}", end="\r")
    print("")
    return _plans


def roughly_remove_isomorphic_plans_func(args):
    plans, id, counter = args
    _plans = []
    for i, plan in enumerate(plans):
        if not has_isomorphic_plan(plan, _plans):
            _plans.append(plan)
        if (i + 1) % 1000 == 0 or i == len(plans) - 1:
            counter[id] = [len(_plans), i + 1]
    return _plans


def _print_roughly_remove_isomorphic_counter(counter):
    counts = [0, 0]
    for k, v in counter.items():
        counts[0] += v[0]
        counts[1] += v[1]

    print(
        "roughly removing isomorphic plans: "
        f"{counts[0]}/{counts[1]}", end="\r"
    )


def _print_roughly_remove_isomorphic_counter_fun(counter, stop_event):
    while not stop_event.is_set():
        _print_roughly_remove_isomorphic_counter(counter)
        time.sleep(1)


def roughly_remove_isomorphic_plans(plans):
    indices = np.arange(len(plans))
    rng.shuffle(indices)
    plans = [plans[index] for index in indices]
    batch_size = len(plans) // roughly_remove_isomorphic_plans_n_batches + 1

    with Manager() as manager:
        counter = manager.dict()
        stop_event = Event()
        batched_plans = [
            (
                plans[i * batch_size:(i + 1) * batch_size],
                i, counter
            )
            for i in range(roughly_remove_isomorphic_plans_n_batches)
        ]

        counter_printer = Process(
            target=_print_roughly_remove_isomorphic_counter_fun, args=(
                counter, stop_event)
        )
        counter_printer.start()

        with Pool(min(cpu_count(), roughly_remove_isomorphic_plans_n_batches)) as e:
            _plans = e.map(
                roughly_remove_isomorphic_plans_func,
                batched_plans,
                chunksize=1
            )

        stop_event.set()
        counter_printer.join()

        _print_roughly_remove_isomorphic_counter(counter)
        print("")

    _plans = [plan for __plans in _plans for plan in __plans]
    return _plans


def enum(path, resume=False, max_n_steps=6, max_n_followers=6):
    if not resume:
        timestamp = int(time.time())
        folder_name = os.path.join(path, f"{timestamp}_plans")
        os.makedirs(folder_name)
    else:
        folder_name = path

    def load_plans(name):
        file_names = [
            f for f in os.listdir(folder_name)
            if name + ".npy" in f
        ]
        if len(file_names) > 0:
            assert len(file_names) == 1
            return np.load(
                os.path.join(folder_name, file_names[0]), allow_pickle=True
            ).item()["plans"]
        else:
            return None

    def save_plans(name, plans):
        timestamp = int(time.time())
        data = {"timestamp": timestamp, "plans": plans}
        np.save(
            os.path.join(folder_name, f"{timestamp}_{name}.npy"),
            data
        )

    print("enumerating all plans")
    name = f"all_plans_{max_n_steps}"
    loaded_plans = load_plans(name)
    if loaded_plans is None:
        all_plans = enum_all_plans(max_n_steps, max_n_followers)
        save_plans(name, all_plans)
    else:
        all_plans = loaded_plans

    print("removing isomorphic and subgraph isomorphic plans")
    unique_plans = []
    for i, plans in enumerate(all_plans):
        print(f"step: {i + 1}, total: {len(plans)}")

        name = f"non_subgraph_isomorphic_plans_{i + 1}"
        loaded_plans = load_plans(name)
        if loaded_plans is None:
            plans = remove_subgraph_isomorphic_plans(
                plans, unique_plans
            )
            save_plans(name, plans)
        else:
            plans = loaded_plans
        print(f"step: {i + 1}, non subgraph isomorphic: {len(plans)}")

        last_count = len(plans)
        for j in range(10):
            if len(plans) <= min_parallel_size:
                break

            name = f"roughly_non_isomorphic_plans_{j}_{i + 1}"
            loaded_plans = load_plans(name)
            if loaded_plans is None:
                plans = roughly_remove_isomorphic_plans(plans)
                save_plans(name, plans)
            else:
                plans = loaded_plans

            count = len(plans)
            if (last_count - count) / last_count < 0.1:
                break
            last_count = count
        print(f"step: {i + 1}, roughly non isomorphic: {len(plans)}")

        name = f"non_isomorphic_plans_{i + 1}"
        loaded_plans = load_plans(name)
        if loaded_plans is None:
            plans = remove_isomorphic_plans(plans)
            save_plans(name, plans)
        else:
            plans = loaded_plans
        print(f"step: {i + 1}, non isomorphic: {len(plans)}")

        unique_plans += plans

    name = f"final_plans_{max_n_steps}"
    loaded_plans = load_plans(name)
    if loaded_plans is None:
        save_plans(name, unique_plans)


def load(path=None):
    if path is None:
        with importlib.resources.open_binary(resources, "plans.npy") as f:
            return np.load(f, allow_pickle=True).item()["plans"]
    else:
        return np.load(path, allow_pickle=True).item()["plans"]


def plot(plan):
    g = gen_graph(plan)

    pos = nx.planar_layout(g)
    labels = nx.get_node_attributes(g, 'type')
    for k, v in labels.items():
        labels[k] = f"{k}{v}"

    edge_labels = nx.get_edge_attributes(g, 'type')
    for k, v in edge_labels.items():
        edge_labels[k] = f"{v}"

    nx.draw(
        g, pos,
        labels=labels,
        node_size=250,
        font_size=10,
        width=1,
    )
    nx.draw_networkx_edge_labels(
        g, pos,
        edge_labels=edge_labels,
        font_size=10,
    )
