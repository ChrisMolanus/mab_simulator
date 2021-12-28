import csv
from datetime import datetime, timedelta
from random import seed
from typing import Dict, List, Any, Tuple, Type, Union
from multiprocessing import Process, Queue

from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

import bayesianGroundhog
import epsilonRingtail
import randomCrayfish
import segmentJunglefowl
from actionGenerator import get_actions
from customerGenerator import generate_customers, what_would_a_customer_do, get_products
from exportHistory import export_history_to_parquet
from policy import ServedActionPropensity, Policy, Customer, Channel, Action, Transaction, HistoricalActionPropensity, \
    get_channel_action_cost
from rewardCalculator import RewardCalculator

call_center_daily_quota = 200


def get_history(start_ts: datetime = datetime(2011, 1, 1),
                export: bool = False) -> Tuple[List[HistoricalActionPropensity], List[Customer], List[Action]]:
    """
    Generates transactions of generated customers using the JungleFowl policy
    :param start_ts: The date to start simulating a transaction history
    :param export: True if you would like all the generated data to be exported to the output dir
    :return: Historical transactions, the customers and the list of actions that could have been performed
    """
    seed(7837)
    np.random.seed(7837)

    sim_start_date = start_ts.date()

    reward_calculator = RewardCalculator()
    products, product_market_size = get_products(sim_start_date - timedelta(days=2190), sim_start_date)
    product_market_sizes = [0.0] * len(product_market_size)

    day_count = 365
    # Set the base product distribution to how it was on start_date
    product_market_sizes[0] = 0.19
    product_market_sizes[1] = 0.51
    product_market_sizes[2] = 0.3
    all_customers = generate_customers(100000, sim_start_date, product_market_sizes)
    keywords = {"current_base": all_customers}

    all_actions = get_actions()

    log, chosen_action_log, historical_action_propensities = sim_cycle_run(all_actions, all_customers, day_count,
                                                                           segmentJunglefowl.SegmentJunglefowl,
                                                                           reward_calculator, [], start_ts, **keywords)
    if export:
        export_history_to_parquet(historical_action_propensities, all_customers, all_actions)

    return historical_action_propensities, all_customers, all_actions


def policy_sim(policy_class, all_customers: List[Customer], all_actions: List[Action], day_count: int, output: Queue,
               run_id: int, sequential_runs: int, history: List[HistoricalActionPropensity],
               start_ts: datetime = datetime.today(), **kwargs) -> bool:
    """
    Run a number of cycles of a policy
    :param policy_class: The policy to run
    :param all_customers: The customers to use during the run
    :param all_actions: The actions that are allowed to be executed on the customers during the run
    :param day_count: The length of the simulation in days
    :param output: The Queue to output the results to
    :param run_id: The ID of this run
    :param sequential_runs: The number of sequential cycles to run, used to eliminate waiting time ot start a new thread
    :param history: The list of HistoricalActionPropensity to pass to the policy on startup
    :param start_ts: The timestamp to start the simulation at
    :param kwargs: kargs to pass to the policy
    :return:
    """
    print(policy_class.__name__ + str(run_id))
    seed(7831 * run_id)
    np.random.seed(7831 * run_id)
    reward_calculator = RewardCalculator()

    logs: List[List[Dict[str, Any]]] = list()

    if run_id == 0:
        # Only one sim should create a sample offer distribution plot
        chosen_action_log: Dict[datetime, Dict[str, int]] = dict()

    # We can run multiple sequential runs since there is a time overhead to create new processes
    for s_run in range(sequential_runs):
        # Just overwrite the chosen_action_log and take the last one
        log, chosen_action_log, _ = sim_cycle_run(all_actions, all_customers, day_count,
                                                  policy_class, reward_calculator, history, start_ts, **kwargs)
        logs.append(log)

    # Only the last chosen_action_log
    if run_id == 0:
        output.put({"policy": policy_class.__name__, "logs": logs, "chosen_action_log": chosen_action_log})
    else:
        output.put({"policy": policy_class.__name__, "logs": logs})
    print(f"{policy_class.__name__} end")
    return True


def sim_cycle_run(all_actions, all_customers, day_count, policy_class, reward_calculator,
                  history: List[HistoricalActionPropensity], start_ts: datetime = datetime.today(), **kwargs
                  ) -> Tuple[List[Dict[str, Any]], Dict[datetime, Dict[str, int]], List[HistoricalActionPropensity]]:
    """
    Run a single simulation cycle
    :param all_actions: The actions that can be executed on the customers
    :param all_customers: The customers
    :param day_count: The length of the simulation in days
    :param policy_class: The policy to use to execute the actions
    :param reward_calculator: The reward calculator to calculate eh reward of a customer action
    :param history: Historical transactions
    :param start_ts: The timestamp to start the simulation at
    :param kwargs: The kargs to pass to the policy
    :return: log, chosen_action_log, historicalActionPropensities
    """
    chosen_action_log: Dict[datetime, Dict[str, int]] = dict()
    log: List[Dict[str, Any]] = list()
    customers: List[Customer] = all_customers.copy()
    policy: Policy = policy_class(history, **kwargs)
    policy.set_datetime(start_ts)
    today = start_ts.date()
    actions: List[Action] = list()
    historicalActionPropensities: List[HistoricalActionPropensity] = history
    # Since we are simulating implementing this in a company already has customers
    # and has already ran campaigns in the past we initialize teh policy with the actions that are already active
    for action in all_actions:
        # So not today == action.start_date, that is later
        if action.start_date <= today < action.end_date:
            policy.add_arm(action, [1])
        elif action.start_date >= today:
            actions.append(action)
    served_action_propensities: Dict[int, ServedActionPropensity] = dict()
    action_timeout: Dict[datetime, dict[int, ServedActionPropensity]] = dict()
    # Run a simulation of day_count number of days and track the cumulative_reward
    cumulative_reward = 0.0
    for today_datetime in (start_ts + timedelta(n) for n in range(day_count)):
        today = today_datetime.date()
        if today not in action_timeout:
            action_timeout[today] = dict()

        # Only one sim should create a sample offer distribution plot
        chosen_action_log[today] = dict()
        chosen_action_log[today]["No Action"] = 0
        for action in all_actions:
            chosen_action_log[today][action.name] = 0

        # Since we are running a simulation we need to tell the policy it is the next day
        policy.set_datetime(today_datetime)

        # Add new actions that are available from today
        actions_to_remove = list()
        for action in actions:
            if today == action.start_date:
                policy.add_arm(action, [1])
                actions_to_remove.append(action)
        for action in actions_to_remove:
            actions.remove(action)

        # Select a set of people that we can call today
        todays_served_action_propensities = list()
        todays_call_served_action_propensities = list()
        todays_email_served_action_propensities = list()
        # TODO: Since emails don't effect the call_center_daily_quota we should do this in a different way
        while len(todays_served_action_propensities) < call_center_daily_quota and len(customers) > 0:
            customer = customers.pop()
            # Check if we ave already called them (Data quality)
            if customer.id not in served_action_propensities:
                served_action_propensity = policy.get_next_best_action(customer=customer, segment_ids=[1])
                # served_action_propensity can be None if there is no possible action due to constrains
                if served_action_propensity is not None:
                    todays_served_action_propensities.append(served_action_propensity)
                    # Find the right channel queue, needed since these are usually separate systems
                    # But this is only here for illustration purposes, it has no effect in teh simulator
                    if served_action_propensity.chosen_action.channel == Channel.OUTBOUND_CALL:
                        todays_call_served_action_propensities.append(served_action_propensity)
                    elif served_action_propensity.chosen_action.channel == Channel.OUTBOUND_EMAIL:
                        todays_email_served_action_propensities.append(served_action_propensity)

                    served_action_propensities[customer.id] = served_action_propensity

                    chosen_action_log[today][served_action_propensity.chosen_action.name] += 1
                else:
                    chosen_action_log[today]["No Action"] += 1

        # Look for old actions that have timed out
        if today in action_timeout:
            for customer_id, served_action_propensity in action_timeout[today].items():
                # Gave up on this customer since they did not convert
                reward = get_channel_action_cost(served_action_propensity.chosen_action.channel) * -1
                policy.add_customer_action(served_action_propensity=served_action_propensity,
                                           customer_action=None,
                                           reward=reward)
                historicalActionPropensities.append(
                    HistoricalActionPropensity(customer=served_action_propensity.customer,
                                               chosen_action=served_action_propensity.chosen_action,
                                               action_propensities=served_action_propensity.action_propensities,
                                               action_ts=served_action_propensity.action_ts,
                                               reward_ts=today_datetime,
                                               reward=reward
                                               )
                )
                cumulative_reward += reward
        del action_timeout[today]

        # Fist simulate late reactions of customers that were contacted on other days
        for deadline, customer_served_action_propensities in action_timeout.items():
            # Today's timeouts have already been removed above
            customers_who_responded: List[int] = list()
            for customer_id, served_action_propensity in customer_served_action_propensities.items():
                days_since_action = (today_datetime - served_action_propensity.action_ts).days
                customer_action = what_would_a_customer_do(served_action_propensity.customer,
                                                           served_action_propensity.chosen_action,
                                                           today_datetime, days_since_action)

                if isinstance(customer_action, Transaction):
                    reward = reward_calculator.calculate(served_action_propensity.customer, customer_action)
                    reward -= get_channel_action_cost(served_action_propensity.chosen_action.channel)
                    policy.add_customer_action(served_action_propensity=served_action_propensity,
                                               customer_action=customer_action,
                                               reward=reward)
                    historicalActionPropensities.append(
                        HistoricalActionPropensity(customer=served_action_propensity.customer,
                                                   chosen_action=served_action_propensity.chosen_action,
                                                   action_propensities=served_action_propensity.action_propensities,
                                                   action_ts=served_action_propensity.action_ts,
                                                   reward_ts=today_datetime,
                                                   reward=reward
                                                   )
                    )

                    cumulative_reward += reward
                    customers_who_responded.append(customer_id)

            # Remove timeouts for customers that responded (late)
            for customer_id in customers_who_responded:
                del customer_served_action_propensities[customer_id]

        # Simulate today's actions
        call_counter = 0
        for served_action_propensity in todays_served_action_propensities:

            # There is a hard limit of how many calls can be performed by a fixed number of agents
            # So we have to simulate that limit
            if call_counter >= call_center_daily_quota:
                break

            # Get a customer to call or email
            customer = served_action_propensity.customer
            cool_off_days = served_action_propensity.chosen_action.cool_off_days

            # Simulate what the customer would have done
            customer_action = what_would_a_customer_do(served_action_propensity.customer,
                                                       served_action_propensity.chosen_action,
                                                       today_datetime)

            # Accounting now that the call has been made
            call_counter += 1
            served_action_propensity.set_action_timestamp(today_datetime)
            policy.add_company_action(served_action_propensity,
                                      served_action_propensity.chosen_action,
                                      today,
                                      get_channel_action_cost(served_action_propensity.chosen_action.channel) * -1)

            # See if we have immediate reward, the customer bought the product during the call
            if isinstance(customer_action, Transaction):
                reward = reward_calculator.calculate(served_action_propensity.customer, customer_action)
                reward -= get_channel_action_cost(served_action_propensity.chosen_action.channel)
                policy.add_customer_action(served_action_propensity=served_action_propensity,
                                           customer_action=customer_action,
                                           reward=reward)
                historicalActionPropensities.append(
                    HistoricalActionPropensity(customer=served_action_propensity.customer,
                                               chosen_action=served_action_propensity.chosen_action,
                                               action_propensities=served_action_propensity.action_propensities,
                                               action_ts=served_action_propensity.action_ts,
                                               reward_ts=today_datetime,
                                               reward=reward
                                               )
                )

                cumulative_reward += reward
            else:
                # No immediate reward, but the customer might go on the website later to make the sale
                deadline = today + timedelta(days=cool_off_days)
                if deadline not in action_timeout:
                    action_timeout[deadline] = dict()
                action_timeout[deadline][customer.id] = served_action_propensity

            # TODO: implement the possibility that the customer buys the product at a later date within the cooloff

        log.append({"ts": today, "cumulative_reward": cumulative_reward})
    return log, chosen_action_log, historicalActionPropensities


def get_timeline_plot(x, y_per_action, label) -> plt.Figure:
    """
    Generate time line plot
    :param x:
    :param y_per_action:
    :param label:
    :return:
    """
    # Basic stacked area chart.
    fig, ax = plt.subplots()
    ax.stackplot(x, *y_per_action)  # , labels=labels)
    # plt.legend(loc='upper left')
    ax.set(xlabel='time (days)', ylabel='NBA allocations',
           title=label)
    return fig


def get_performance_plot(plot_dfs, sorted_policies):
    fig, ax = plt.subplots()
    for policy_name in sorted_policies:
        policy = plot_dfs[policy_name]
        euro = policy["mean"].iloc[-1]
        policy["mean_k"] = policy["mean"] / 1000
        policy["std_u"] = policy["mean_k"] + (policy["std"] / 1000)
        policy["std_l"] = policy["mean_k"] - (policy["std"] / 1000)

        ax.fill_between(policy["ts"], policy["std_l"], policy["std_u"], alpha=0.2)
        ax.plot(policy["ts"], policy["mean_k"], label=f"{policy_name} €{euro}")

    ax.set(xlabel='time (days)', ylabel='Cumulative HLV (1000 Euros)',
           title='Policy performance')
    ax.grid()
    plt.legend()
    return fig


def plot_timelines(chosen_action_logs: Dict[str, Dict[datetime, int]], actions: List[Action], show=True, save=True) -> Dict[str, plt.Figure]:
    # Plot policy timelines
    xs: Dict[str, List[datetime]] = dict()
    policy_labels: Dict[str, List[str]] = dict()
    ys: Dict[str, List[List[int]]] = dict()
    for policy_name, chosen_action_log in chosen_action_logs.items():
        labels: List[str] = list()
        x: List[datetime] = list(chosen_action_log.keys())
        x.sort()
        y_per_action: List[List[int]] = list()
        for action in actions:
            y_per_action.append(list())
            labels.append(action.name)
        for day in x:
            chosen_action_counts = chosen_action_log[day]
            for i in range(len(labels)):
                action_name = labels[i]
                if action_name in chosen_action_counts:
                    y_per_action[i].append(chosen_action_counts[action_name])
                else:
                    y_per_action[i].append(0)
        xs[policy_name] = x
        policy_labels[policy_name] = labels
        ys[policy_name] = y_per_action

    plots: Dict[str, plt.Figure] = dict()
    for policy_name in policy_labels.keys():
        fig = get_timeline_plot(xs[policy_name], ys[policy_name], policy_name)
        plots[policy_name] = fig
        if save:
            fig.savefig(f"{policy_name}.png")
        if show:
            plt.show()
    return plots


def plot_performance(all_logs: Dict[str, Dict[datetime, List[float]]], show=False, save=False):
    plot_dfs: Dict[str, DataFrame] = dict()
    plot_dict: Dict[str, List[Dict[str, Any]]] = dict()
    last_mean_value: Dict[str, float] = dict()
    for policy, log in all_logs.items():
        if policy not in plot_dict:
            plot_dict[policy] = list()
        for ts, sim_values in log.items():
            plot_dict[policy].append({"ts": ts, "mean": np.mean(sim_values), "std": np.std(sim_values)})
            last_mean_value[policy] = np.mean(sim_values)
        plot_dfs[policy] = DataFrame(plot_dict[policy])
    ordered_policies_by_clv = sorted(last_mean_value, key=last_mean_value.get)
    ordered_policies_by_clv.reverse()
    fig = get_performance_plot(plot_dfs, ordered_policies_by_clv)
    if save:
        fig.savefig("test.png")
    if show:
        plt.show()
    return fig


def export_log_data(all_logs: Dict[str, Dict[datetime, List[float]]], output_dir: str = "output"):
    """
    Export logs as YAML files to the "output" dir
    :param all_logs: The logs to export
    :param output_dir: The directory to put the files, will be created if not exist
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for policy_name, log in all_logs.items():
        with open(os.path.join(output_dir, policy_name + ".yaml"), "w") as f:
            f.write(yaml.safe_dump(log))


def do_simulations(policies: List[Type[Union[Any]]], keywords: Dict[str, Any], runs_per_policies: int, sequential_runs: int,
                   customers: List[Customer], actions: List[Action], day_count: int, start_ts: datetime,
                   historical_action_propensities: List[HistoricalActionPropensity] = list()
                   ) -> Tuple[Dict[str, Dict[datetime, List[float]]], Dict[str, Dict[datetime, Dict[str, int]]]]:


    processes = list()
    output_queue = Queue()
    for policy_class in policies:
        for r in range(runs_per_policies):

            p = Process(target=policy_sim,
                        args=(policy_class, customers, actions, day_count, output_queue, r, sequential_runs,
                              historical_action_propensities, start_ts),
                        kwargs=keywords)
            processes.append(p)

    for p in processes:
        p.start()

    all_logs: Dict[str, Dict[datetime, List[float]]] = dict()
    chosen_action_logs: Dict[str, Dict[datetime, Dict[str, int]]] = dict()
    for policy_class in policies:
        policy_name = policy_class.__name__
        all_logs[policy_name] = dict()

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
            chosen_action_logs[policy_name] = output_logs["chosen_action_log"]

    for p in processes:
        if p.is_alive():
            p.join()

    return all_logs, chosen_action_logs

if __name__ == "__main__":
    policies = [randomCrayfish.RandomCrayfish, segmentJunglefowl.SegmentJunglefowl, bayesianGroundhog.BayesianGroundhog,
                epsilonRingtail.EpsilonRingtail]
    #policies = [segmentJunglefowl.SegmentJunglefowl]

    runs_per_policies = 1
    sequential_runs = 5

    processes = list()
    historical_action_propensities, _, _ = get_history(export=False)
    start_ts = datetime.today()
    customers = generate_customers(100000, start_ts.date())
    actions = get_actions()
    keywords = {'epsilon': 0.8, 'resort_batch_size': 50, "initial_trials": 99, "initial_conversions": 1,
                'gold_threshold': None, 'silver_threshold': None, "current_base": customers}

    epsilon = 0.8
    resort_batch_size = 50
    initial_trials = 99
    initial_conversions = 1
    day_count = 365
    start_ts = datetime.today()
    all_logs, chosen_action_logs = do_simulations(policies, keywords, runs_per_policies, sequential_runs, customers,
                                                  actions, day_count, start_ts, historical_action_propensities)

    export_log_data(all_logs)

    plot_timelines(chosen_action_logs, actions, show=True, save=True)

    # Plot performance
    plot_performance(all_logs, show=True, save=True)
