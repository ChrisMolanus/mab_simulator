from datetime import datetime, date
from multiprocessing import freeze_support
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
from matplotlib import ticker, patches

import bayesianGroundhog
import epsilonRingtail
import randomCrayfish
import segmentJunglefowl
from actionGenerator import get_actions
from customerGenerator import generate_customers, get_products
from rewardCalculator import HlvCalculator
from simulator import TelcoSimulator

st.set_page_config(layout="wide")

matplotlib.use("agg")

start_ts = datetime.today()
today = start_ts.date()

st.header("Telco Marketing simulator")
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.markdown("""This is a Telecommunications company Marketing Policy simulator 
    that allows developers to test different Marketing policies.
    The simulator generates customer profile on every run
    and simulates the customers response to a marketing actions e.g. an outbound call. 
    The revenue generated by the policy's choice of customer actions is summed and plotted over time 
    to show the accumulated gains of each policy.""")

st.write("##")
st.write("##")

actions = get_actions()
products, product_market_size = get_products()

customer_h_col1, customer_h_col2, customer_h_col3 = st.columns((1, 2, 1))
with customer_h_col2:
    st.subheader("Customers")

cust_col1, cust_col2, cust_col3 = st.columns((2, 1, 1))
with cust_col1:
    st.write("""Customers are generated on every run using distributions of common first and last names of the 
    population of the Netherlands. 
    The date of births of the customers are generated using the age density of telecom service customers.
    The products are randomly assigned to customer portfolios with a weighed distribution that reflects a telecom 
    company with a base that mostly has portfolios with older products and only some with newer products.""")

    nr_of_customers: float = st.slider(label="Base Size", min_value=10000, max_value=800000, value=100000, step=10000)

    customers = generate_customers(int(nr_of_customers), today)

    sample_cust = customers[0:8]
    cust_list = list()
    for c in sample_cust:
        cust_list.append({  # "id": c.id,
            "name": c.name,
            "dob": c.dob,
            "billing_address": str(c.billing_address),
            "portfolio": str([str(p) for p in c.portfolio])})
    cust_df = pd.DataFrame(cust_list)
    st.dataframe(cust_df)

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

    fig, ax = plt.subplots()
    langs = list()
    students = list()
    for product in products:
        product_name = product.name
        count = portfolio_count[product_name]
        langs.append(product_name)
        students.append(count)
    ax.bar(langs, students, alpha=0.3)
    ax.tick_params(labelrotation=90)
    ax.set_ylabel("Segment size")
    st.pyplot(fig)

st.write("##")

products_h_col1, products_h_col2, products_h_col3 = st.columns((1, 2, 1))
with products_h_col2:
    st.subheader("Products")

products_col1, products_col2, products_col3 = st.columns((2, 1, 1))
with products_col1:
    st.write("""This simulator only considered Fixed internet services to allows the simulator to finish fast.
    The products are based in a Dutch Telco operator Ziggo but are fake products. 
    The yearly margins on the products are reasonable for a Dutch Telco but are not the actual margins of Ziggo.
    Adjusting the Average Price per Unit sold (ARPU) changes the list price of the product 
    in proportion the original list price. Since the costs can not change this also changes the margin. 
    Increasing the Marketing Budget allows for more costly actions like 
    better graphics which increase campaigns effectiveness,
    or make outbound calls which require most call center agents.""")

    arpu: int = st.slider(label="ARPU €", min_value=100, max_value=3000, value=2100, step=100)
    marketing_budget: int = st.slider(label="Marketing Budget (Million €)", min_value=18, max_value=50, value=25,
                                      step=1)

    prod = list()
    for p in products:
        prod.append({"name": p.name, "list_price": p.list_price + (arpu - 2100), "margin": p._margin + (arpu - 2100),
                     "start_date": p.start_date, "end_date": p.end_date, "download_speed": p.kwargs["download_speed"],
                     "upload_speed": p.kwargs["upload_speed"]})
    prod_df = pd.DataFrame(prod)
    st.dataframe(prod_df)

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
    # Alter the spread to make the plot look better
    x = [0.0, 0.0, 0.0, 0.14, 0.051, 0.48, 0.17, 0.5, 0.5, 0.02, 0.06, 0.05, 0.5, 0.5]
    x = [p + ((marketing_budget - 25) / 100) for p in x]
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

segment_h_col1, segment_h_col2, segment_h_col3 = st.columns((1, 2, 1))
with segment_h_col1:
    st.image(segmentJunglefowl.SegmentJunglefowl.icon, width=100)
with segment_h_col2:
    st.subheader("Gold Silver Bronze segments")

segment_col1, segment_col2, segment_col3 = st.columns((2, 1, 1))
with segment_col1:
    st.write("""The Gold Silver Bronze segments policy uses the traditional marketing segmentation 
    where we segment the base into High medium and low revenue groups. 
    Then for every group we assign actions that try and sell them a product 
    or service with a price point for that group""")

    gold_threshold: float = st.slider(label="Gold Segment", min_value=0.0, max_value=8000.0, value=5600.0, step=200.0)
    silver_threshold: float = st.slider(label="Silver Segment", min_value=0.0, max_value=8000.0, value=2800.0,
                                        step=200.0)

with segment_col2:
    hlv_calculator = HlvCalculator()

    margins = list()
    for customer in customers:
        margin = hlv_calculator.get_hlv(customer, today)
        margins.append(hlv_calculator.get_hlv(customer, today, 20))
    fig, ax = plt.subplots()
    ax.hist(margins, bins=20)

    gold_pathch = patches.Rectangle((gold_threshold, 0), (max(margins) - gold_threshold), 25000, angle=0.0, alpha=0.3,
                                    ec="gray", fc="CornflowerBlue")
    ax.add_patch(gold_pathch)

    silver_pathch = patches.Rectangle((silver_threshold, 0), (gold_threshold - silver_threshold), 25000, angle=0.0,
                                      alpha=0.3, ec="gray", fc="red")
    ax.add_patch(silver_pathch)

    bronze_pathch = patches.Rectangle((0, 0), silver_threshold, 25000, angle=0.0, alpha=0.3, ec="gray", fc="green")
    ax.add_patch(bronze_pathch)

    ax.set_ylabel('Number of customers')
    ax.legend(["Gold", "Silver", "Bronze"])
    st.pyplot(fig)

st.write("##")
epsilon_h_col1, epsilon_h_col2, epsilon_h_col3 = st.columns((1, 2, 1))
with epsilon_h_col1:
    st.image(epsilonRingtail.EpsilonRingtail.icon, width=100)
with epsilon_h_col2:
    st.subheader("Epsilon Greedy")

epsilon_col1, epsilon_col2, epsilon_col3 = st.columns((2, 1, 1))
with epsilon_col1:
    st.write("""The Epsilon Greedy policy uses a basic Explorer/Exploit ratio to test out new campaigns to better 
    estimate the conversion rate. Then for every customer the estimated conversion rate is multiplied by the increase in 
    Household Lifetime value (Delta HLV) to calculate teh estimated revenue. 
    The Epsilon parameter defines the percentage of instances the algorithm will Exploit the campaign that is 
    estimated to give the highest revenue. The rest of the time (1 - Epsilon) the algorithm will test the newer 
    campaigns.
    This is because we will never have enough chances to calculate the true conversion rate of a campaign. 
    This is to avoid campaigns that had bad luck to be tested on the difficult people first 
    still get another chance.""")
    epsilon: float = st.slider(label="Epsilon", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
    resort_batch_size: int = st.slider(label="Batch size", min_value=1, max_value=201, value=51, step=10)

with epsilon_col2:
    fig, ax = plt.subplots()

    ax.bar(["Base"], [(1 - epsilon) * 100], 5, bottom=[epsilon * 100], label='Explorer', alpha=0.2)
    ax.bar(["Base"], [epsilon * 100], 5, label='Exploit', alpha=0.2)

    ax.set_ylabel('Percentage Offers')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend()
    st.pyplot(fig)

st.write("##")
bayesian_h_col1, bayesian_h_col2, bayesian_h_col3 = st.columns((1, 2, 1))
with bayesian_h_col1:
    st.image(bayesianGroundhog.BayesianGroundhog.icon, width=100)
with bayesian_h_col2:
    st.subheader("Bayesian")
bayesian_col1, bayesian_col2 = st.columns(2)
with bayesian_col1:
    st.write("""The Bayesian policy uses Thompson-Sampling to estimate the rewards of serving the customer each campaign
    . Each Action reward is defined as a beta distribution that is updated when ever an Action succeeds or fails. 
    The Action beta distribution is sampled for every new customer to generate the expected rewards for that customer.
    The algorithm then chooses the action with the maximum expected reward (Delta Household Lifetime Value. 
    The plots here show the probability density of the conversion rate of three Action at different simulated update 
    steps.
    The Beta distribution of the Action's conversion rate is updated by the simulated success/fail reward steps.
    Here we can test how quickly the Action Beta distributions (Arms) conclude which action is clearly better (Orange).
    """)

    initial_trials: int = st.slider(label="Initial Trails", min_value=0, max_value=500, value=99, step=1)
    initial_wins: int = st.slider(label="Initial Wins", min_value=0, max_value=500, value=1, step=1)

with bayesian_col2:
    # Define the multi-armed bandits
    nb_bandits = 3  # Number of bandits
    # True probability of winning for each bandit
    p_bandits = [0.45, 0.55, 0.60]


    def pull(arm_index):
        """Pull arm of bandit with index `i` and return 1 if win,
        else return 0."""
        if np.random.rand() < p_bandits[arm_index]:
            return 1
        else:
            return 0


    # Define plotting functions
    # Iterations to plot
    plots = [2, 10, 50, 200, 500, 1000]


    def plot(priors, step_count, ax_of_plot):
        """Plot the priors for the current step."""
        plot_x = np.linspace(0.001, .999, 100)
        for prior in priors:
            plot_y = prior.pdf(plot_x)
            _ = ax_of_plot.plot(plot_x, plot_y)
            ax_of_plot.fill_between(plot_x, plot_y, 0, alpha=0.2)
        ax_of_plot.set_xlim([0, 1])
        ax_of_plot.set_ylim(bottom=0)
        ax_of_plot.set_title(f'Priors at step {step_count:d}')


    fig, axs = plt.subplots(2, 3, )
    axs = axs.flat

    # The number of trials and wins will represent the prior for each
    #  bandit with the help of the Beta distribution.
    trials = [initial_trials] * 3  # [0, 0, 0]  # Number of times we tried each bandit
    wins = [initial_wins] * 3  # [0, 0, 0]  # Number of wins for each bandit

    n = 1000
    # Run the trail for `n` steps
    for step in range(1, n + 1):
        # Define the prior based on current observations
        bandit_priors = [
            stats.beta(a=1 + w, b=1 + t) for t, w in zip(trials, wins)]
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

st.write("##")
st.write("##")

st.subheader("Simulator")
row3_col1, row3_col2, row3_col3 = st.columns((2, 1, 1))
with row3_col1:
    st.write("To run a simulation you must check the checkbox in the bottom of this section. "
             "Depending on the settings a simulation can take up to 30 minutes (10 sequential runs of 356 days)."
             "We recommend using the default settings so that a simulation can finish in 2 minutes. "
             "The simulation must run multiple sims per policy to estimate the mean revenue per policy."
             "Running multiple instance can be done by her increasing the number of sim threads per policy or "
             "the never of sequential runs within one thread")

    # Defaults are optimized for Linux OS, Windows take a long time ot start a Thread so 1 thread per policy
    # and more sequential runs is advised
    runs_per_policies = st.slider(label="Threads per policy", min_value=1, max_value=10, value=5, step=1)
    sequential_runs = st.slider(label="Sequential runs per thread", min_value=1, max_value=10, value=1, step=1)
    day_count = st.slider(label="Number of days to simulate", min_value=21, max_value=365, value=50, step=1)
    run = st.checkbox("Run Simulator")

if __name__ == '__main__':
    freeze_support()
    if gold_threshold == 0 or silver_threshold == gold_threshold:
        gold_t = None
        silver_t = None
    else:
        gold_t = gold_threshold
        silver_t = silver_threshold

    simulator = TelcoSimulator()
    chosen_action_logs: Dict[str, Dict[datetime, Dict[str, int]]] = dict()
    with row3_col2:
        if run:
            # Run simulations
            policies = [randomCrayfish.RandomCrayfish, segmentJunglefowl.SegmentJunglefowl,
                        epsilonRingtail.EpsilonRingtail, bayesianGroundhog.BayesianGroundhog]
            keywords = {'epsilon': epsilon, 'resort_batch_size': resort_batch_size, "initial_trials": initial_trials,
                        "initial_conversions": initial_wins, "current_base": customers,
                        "gold_threshold": gold_threshold, "silver_threshold": silver_threshold}

            all_logs, chosen_action_logs = simulator.do_simulations(policies, keywords, runs_per_policies,
                                                                    sequential_runs, customers, actions, day_count,
                                                                    start_ts)

            # Plot performance
            st.pyplot(simulator.plot_performance(all_logs, show=False, save=False))

    with row3_col3:
        if run:
            # Plot one timeline per policy
            plots = simulator.plot_timelines(chosen_action_logs, actions, show=False, save=False)
            for policy_name, fig in plots.items():
                st.subheader(policy_name)
                st.pyplot(fig)
