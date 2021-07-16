from datetime import datetime
from multiprocessing import Queue, Process, freeze_support
from typing import Dict, List

import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame

import dashingRingtail
import fierceCrayfish
from actionGenerator import get_actions
from customerGenerator import generate_customers
from simulator import policy_sim

st.set_page_config(layout="wide")

matplotlib.use("agg")

row1_col1, row1_col2 = st.beta_columns(2)
with row1_col1:
    st.markdown("This is a Marketing Policy simulator that allows developers to test different Marketing policies.")

row2_col1, row2_col2 = st.beta_columns(2)
with row2_col1:
    runs_per_policies = st.slider(label="Threads per policy", min_value=1, max_value=4, value=1, step=1)
    sequential_runs = st.slider(label="Sequential runs per thread", min_value=1, max_value=10, value=1, step=1)


def do_simulations(runs_per_policies, sequential_runs):
    policies = [fierceCrayfish.FierceCrayfish, dashingRingtail.DashingRingtail]

    processes = list()
    customers = generate_customers(100000)
    actions = get_actions()
    output_queue = Queue()
    for policy_class in policies:
        for r in range(runs_per_policies):
            p = Process(target=policy_sim, args=(policy_class, customers, actions, 365, output_queue, r, sequential_runs))
            p.start()
            processes.append(p)

    all_logs: Dict[str, Dict[datetime, List[float]]] = dict()
    plot_dict: Dict[str, List[Dict[datetime, dict]]] = dict()
    for policy_class in policies:
        policy_name = policy_class.__name__
        all_logs[policy_name] = dict()
        plot_dict[policy_name] = list()

    for p in processes:
        output_logs = output_queue.get(block=True)
        #for policy_name, run_logs in logs.items():
        logs = output_logs["logs"]
        policy_name = output_logs["policy"]
        for log in logs:
            for log_line in log:
                ts = log_line["ts"]
                cum_reward = log_line["cumulative_reward"]
                if ts not in all_logs[policy_name]:
                    all_logs[policy_name][ts] = list()
                all_logs[policy_name][ts].append(cum_reward)

    for p in processes:
        if p.is_alive():
            p.join()

    plot_dfs: Dict[str, DataFrame] = dict()
    for policy, log in all_logs.items():
        for ts, sim_values in log.items():
            plot_dict[policy].append({"ts": ts, "mean": np.mean(sim_values), "std": np.std(sim_values)})
        plot_dfs[policy] = DataFrame(plot_dict[policy])

    fig, ax = plt.subplots()
    for policy_name, policy in plot_dfs.items():
        ax.plot(policy["ts"], policy["mean"]/1000, label=policy_name)

    ax.set(xlabel='time (days)', ylabel='Cumulative HLV (1000 Euros)',
           title='Policy performance')
    ax.grid()
    plt.legend()
    return fig


if __name__ == '__main__':
    freeze_support()

    row3_col1, row3_col2 = st.beta_columns(2)

    with row3_col1:
        st.subheader('Policy performance')
        fig = do_simulations(runs_per_policies, sequential_runs)
        st.pyplot(fig)