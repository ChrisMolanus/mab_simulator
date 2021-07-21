from datetime import datetime
from multiprocessing import Queue, Process, freeze_support
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from pandas import DataFrame
import scipy.stats as stats

import bayesianGroundhog
import epsilonRingtail
import randomCrayfish
from actionGenerator import get_actions
from customerGenerator import generate_customers, get_products
from simulator import policy_sim

st.set_page_config(layout="wide")

matplotlib.use("agg")



row1_col1, row1_col2 = st.beta_columns(2)
with row1_col1:
    st.markdown("This is a Marketing Policy simulator that allows developers to test different Marketing policies.")

st.write("##")
st.write("##")

customers = generate_customers(1)
actions = get_actions()
products, product_market_size = get_products()

cust_col1, cust_col2, cust_col3 = st.beta_columns((2, 1, 1))
with cust_col1:
    st.header("Customers")
    nr_of_customers: float = st.slider(label="Base Size", min_value=10000, max_value=800000, value=100000, step=10000)
    #resort_batch_size: int = st.slider(label="Batch size", min_value=1, max_value=201, value=51, step=10)

    customers = generate_customers(nr_of_customers)

    sample_cust = customers[0:8]
    cust_list = list()
    for c in sample_cust:
        cust_list.append({# "id": c.id,
                          "name": c.name,
                          "dob": c.dob,
                          "billing_address": str(c.billing_address),
                          "portfolio": str([str(p) for p in c.portfolio])})
    cust_df = pd.DataFrame(cust_list)
    cust_df

with cust_col2:
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('')

    portfolio_count = dict()
    for product in products:
        portfolio_count[product.name] = 0
    max_product_name_length = 0
    for cust in customers:
        product_name = cust.portfolio[0].name
        if product_name not in portfolio_count:
            portfolio_count[product_name] = 0
        portfolio_count[product_name] += 1
        if len(product_name) > max_product_name_length:
            max_product_name_length = len(product_name)

    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    fig, ax = plt.subplots()
    langs = list()
    students = list()
    for product in products:
    #for product_name, count in portfolio_count.items():
        product_name = product.name
        count = portfolio_count[product_name]
        langs.append(product_name)
        students.append(count)
    ax.bar(langs, students, alpha =0.3)
    ax.tick_params(labelrotation=90)
    ax.set_ylabel("Segment size")
    #ax.set_xlabel("Perceived Quality")
    st.pyplot(fig)



st.write("##")


products_col1, products_col2, products_col3 = st.beta_columns((2, 1, 1))
with products_col1:
    st.header("Products")
    arpu: int = st.slider(label="ARPU €", min_value=100, max_value=3000, value=2100, step=100)
    marketing_budget: int = st.slider(label="Marketing Budget (Million €)", min_value=18, max_value=50, value=25, step=1)

    prod = list()
    for p in products:
        prod.append({"name":p.name, "list_price":p.list_price + (arpu - 2100), "margin":p._margin + (arpu - 2100), "start_date":p.start_date,
                     "end_date": p.end_date, "download_speed":p.kwargs["download_speed"],
                     "upload_speed":p.kwargs["upload_speed"]})
    prod_df = pd.DataFrame(prod)
    prod_df

with products_col2:
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('')

    fig, ax = plt.subplots()

    # Triangles
    triangle1x = [0, 0.7, 0, 0]
    triangle1y = [620, 900, 900, 620]
    triangle2x = [0, 1, 1, 0]
    triangle2y = [400, 400, 780, 400]
    for i in range(3):
        ax.plot(triangle1x, triangle1y)
    ax.fill_between(triangle1x, triangle1y, alpha=0.2)
    ax.text(0.1, 810, "Expensive", fontsize=14,
            horizontalalignment='left',
            verticalalignment='center')

    for i in range(3):
        ax.plot(triangle2x, triangle2y)
    ax.fill_between(triangle2x, triangle2y, alpha=0.2)
    ax.text(0.7, 480, "Cheap", fontsize=14,
            horizontalalignment='left',
            verticalalignment='center')

    # Products
    # x = [0.14, 0.051, 0.48, 0.17, 0.005, 0.003, 0.02, 0.06, 0.05, 0.02, 0.001]
    x = [0.14, 0.051, 0.48, 0.17, 0.5, 0.5, 0.02, 0.06, 0.05, 0.5, 0.5]
    x = [p + ((marketing_budget - 25) /100) for p in x]
    y = [p.list_price + (arpu - 2100) for p in products]
    ax.scatter(x, y, alpha=0.5)
    ax.plot([0, 1], [500, 900], label="Decision bound", alpha=0.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(400, 900)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('€ %.0f'))
    ax.set_ylabel("List Price")
    ax.set_xlabel("Perceived Quality")

    st.pyplot(fig)

st.write("##")
st.write("##")

epsilon_col1, epsilon_col2, epsilon_col3 = st.beta_columns((2,1,1))
with epsilon_col1:
    st.header("Epsilon Greedy")
    epsilon: float = st.slider(label="Epsilon", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
    resort_batch_size: int = st.slider(label="Batch size", min_value=1, max_value=201, value=51, step=10)

with epsilon_col2:
    fig, ax = plt.subplots()

    ax.bar(["Base"], [(1 - epsilon) * 100], 5, bottom=[epsilon * 100], label='Explorer', alpha=0.2)
    ax.bar(["Base"], [epsilon*100], 5,  label='Exploit', alpha=0.2)


    ax.set_ylabel('Percentage Offers')
    #ax.set_title('Explorer vs Exploit')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend()
    st.pyplot(fig)

bayesian_col1, bayesian_col2 = st.beta_columns(2)
with bayesian_col1:
    st.header("Bayesian")
    initial_trials: int = st.slider(label="Initial Trails", min_value=0, max_value=500, value=99, step=1)
    initial_wins: int = st.slider(label="Initial Wins", min_value=0, max_value=500, value=1, step=1)

with bayesian_col2:
    # Define the multi-armed bandits
    nb_bandits = 3  # Number of bandits
    # True probability of winning for each bandit
    p_bandits = [0.45, 0.55, 0.60]

    def pull(i):
        """Pull arm of bandit with index `i` and return 1 if win,
        else return 0."""
        if np.random.rand() < p_bandits[i]:
            return 1
        else:
            return 0

    # Define plotting functions
    # Iterations to plot
    plots = [2, 10, 50, 200, 500, 1000]

    def plot(priors, step, ax):
        """Plot the priors for the current step."""
        plot_x = np.linspace(0.001, .999, 100)
        for prior in priors:
            y = prior.pdf(plot_x)
            p = ax.plot(plot_x, y)
            ax.fill_between(plot_x, y, 0, alpha=0.2)
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_title(f'Priors at step {step:d}')

    fig, axs = plt.subplots(2, 3,)
    axs = axs.flat

    # The number of trials and wins will represent the prior for each
    #  bandit with the help of the Beta distribution.
    trials = [initial_trials] * 3 #[0, 0, 0]  # Number of times we tried each bandit
    wins = [initial_wins] * 3 #[0, 0, 0]  # Number of wins for each bandit

    n = 1000
    # Run the trail for `n` steps
    for step in range(1, n + 1):
        # Define the prior based on current observations
        bandit_priors = [
            stats.beta(a=1 + w, b=1 + t - w) for t, w in zip(trials, wins)]
        # plot prior
        if step in plots:
            plot(bandit_priors, step, next(axs))
        # Sample a probability theta for each bandit
        theta_samples = [
            d.rvs(1) for d in bandit_priors
        ]
        # choose a bandit
        chosen_bandit = np.argmax(theta_samples)
        # Pull the bandit
        x = pull(chosen_bandit)
        # Update trials and wins (defines the posterior)
        trials[chosen_bandit] += 1
        wins[chosen_bandit] += x

    plt.tight_layout()
    st.pyplot(fig)


def do_simulations(runs_per_policies, sequential_runs, customers, actions,
                   epsilon, resort_batch_size,initial_trials, initial_conversions, day_count):
    policies = [randomCrayfish.RandomCrayfish, epsilonRingtail.EpsilonRingtail, bayesianGroundhog.BayesianGroundhog]

    processes = list()
    output_queue = Queue()
    for policy_class in policies:
        for r in range(runs_per_policies):
            keywords = {'epsilon': epsilon, 'resort_batch_size': resort_batch_size, "initial_trials": initial_trials, "initial_conversions": initial_conversions}
            p = Process(target=policy_sim,
                        args=(policy_class, customers, actions, day_count, output_queue, r, sequential_runs),
                        kwargs=keywords)
            p.start()
            processes.append(p)

    all_logs: Dict[str, Dict[datetime, List[float]]] = dict()
    plot_dict: Dict[str, List[Dict[datetime, dict]]] = dict()
    timeline_plot_dict: Dict[str, Dict[datetime, Dict[str, int]]] = dict()
    for policy_class in policies:
        policy_name = policy_class.__name__
        all_logs[policy_name] = dict()
        plot_dict[policy_name] = list()

    for p in processes:
        output_logs = output_queue.get(block=True)
        logs = output_logs["logs"]
        policy_name = output_logs["policy"]
        for log in logs:
            for log_line in log:
                ts = log_line["ts"]
                cum_reward = log_line["cumulative_reward"]
                if ts not in all_logs[policy_name]:
                    all_logs[policy_name][ts] = list()
                all_logs[policy_name][ts].append(cum_reward)
        if "chosen_action_log" in output_logs:
            # This was a run_id 0 sim
            timeline_plot_dict[policy_name] = output_logs["chosen_action_log"]

    for p in processes:
        if p.is_alive():
            p.join()

    # Timelines
    xs: Dict[str, List[datetime]] = dict()
    policy_labels: Dict[str, List[str]] = dict()
    ys: Dict[str, List[List[int]]] = dict()
    for policy_name, chosen_action_log in timeline_plot_dict.items():
        labels: List[str] = list()
        x: List[datetime] = list(chosen_action_log.keys())
        x.sort()
        y_per_action: List[List[int]] = list()
        for action in actions:
            y_per_action.append(list())
            labels.append(action.name)
        for day in x:
            chosen_action_counts = chosen_action_log[day]
            total_chosen_actions_that_day = sum(list(chosen_action_counts.values()))
            for i in range(len(labels)):
                action_name = labels[i]
                if action_name in chosen_action_counts:
                    y_per_action[i].append(chosen_action_counts[action_name]/total_chosen_actions_that_day)
                else:
                    y_per_action[i].append(0)
        xs[policy_name] = x
        policy_labels[policy_name] = labels
        ys[policy_name] = y_per_action


    # Performance
    plot_dfs: Dict[str, DataFrame] = dict()
    for policy, log in all_logs.items():
        for ts, sim_values in log.items():
            plot_dict[policy].append({"ts": ts, "mean": np.mean(sim_values), "std": np.std(sim_values)})
        plot_dfs[policy] = DataFrame(plot_dict[policy])

    return plot_dfs, xs, policy_labels, ys

def get_timeline_plot(x, y_per_action, label):
    # Basic stacked area chart.
    fig, ax = plt.subplots()
    ax.stackplot(x, *y_per_action)  # , labels=labels)
    # plt.title(policy_name)
    # plt.legend(loc='upper left')
    # plt.show()
    ax.set(xlabel='time (days)', ylabel='NBA allocations',
           title=label)
    return fig


def get_performance_plot(plot_dfs):
    fig, ax = plt.subplots()
    for policy_name, policy in plot_dfs.items():
        policy["mean_k"] = policy["mean"] / 1000
        policy["std_u"] = policy["mean_k"] + (policy["std"] / 1000)
        policy["std_l"] = policy["mean_k"] - (policy["std"] / 1000)

        ax.fill_between(policy["ts"], policy["std_l"], policy["std_u"])
        ax.plot(policy["ts"], policy["mean_k"], label=policy_name)

    ax.set(xlabel='time (days)', ylabel='Cumulative HLV (1000 Euros)',
           title='Policy performance')
    ax.grid()
    plt.legend()
    return fig

row3_col1, row3_col2, row3_col3 = st.beta_columns((2, 1, 1))
with row3_col1:
    st.header("Simulator")
    runs_per_policies = st.slider(label="Threads per policy", min_value=1, max_value=4, value=1, step=1)
    sequential_runs = st.slider(label="Sequential runs per thread", min_value=1, max_value=10, value=1, step=1)
    day_count = st.slider(label="Number of days to simulate", min_value=21, max_value=365, value=50, step=1)

if __name__ == '__main__':
    freeze_support()

    with row3_col2:
        run = st.checkbox("Run Simulator")
        if run:
            plot_dfs, xs, policy_labels, ys = do_simulations(runs_per_policies, sequential_runs, customers, actions,
                                 epsilon, resort_batch_size, initial_trials, initial_wins, day_count)
            st.pyplot(get_performance_plot(plot_dfs))
    with row3_col3:
        if run:
            for policy_name in policy_labels.keys():
                st.subheader(policy_name)
                st.pyplot(get_timeline_plot(xs[policy_name], ys[policy_name], policy_name))