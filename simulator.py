import csv
from datetime import datetime, timedelta
from random import seed
from typing import Dict, List, Any
from multiprocessing import Process, Queue

from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

import bayesianGroundhog
import epsilonRingtail
import randomCrayfish
from actionGenerator import get_actions
from customerGenerator import generate_customers, what_would_a_customer_do
from policy import ServedActionPropensity, Policy, Customer, Address, Product, Channel, Action, Offer, Transaction
from rewardCalculator import RewardCalculator

call_center_daily_quota = 200
cost_of_outbound_call = 8


def policy_sim(policy_class, all_customers: List[Customer], all_actions: List[Action], day_count: int, output: Queue,
               run_id:int, sequential_runs: int, **kwargs) -> DataFrame:
    print(policy_class.__name__ + str(run_id))
    seed(7831*run_id)
    np.random.seed(7831*run_id)
    reward_calculator = RewardCalculator()

    logs: List[List[Dict[str, Any]]] = list()

    if run_id == 0:
        # Only one sim should create a sample offer distribution plot
        chosen_action_log: Dict[datetime, Dict[str, int]] = dict()

    for s_run in range(sequential_runs):
        log: List[Dict[str, Any]] = list()
        customers: List[Customer] = all_customers.copy()
        policy: Policy = policy_class(**kwargs)
        today = datetime.today().date()
        actions: List[Action] = list()
        for action in all_actions:
            # So not today == action.start_date, that is later
            if action.start_date < today < action.end_date:
                policy.add_arm(action, [1])
            elif action.start_date > today:
                actions.append(action)
        served_action_propensities: Dict[int, ServedActionPropensity] = dict()
        action_timeout: Dict[datetime, dict[int, ServedActionPropensity]] = dict()

        cumulative_reward = 0.0
        start_date = datetime.today()
        for today_datetime in (start_date + timedelta(n) for n in range(day_count)):
            today = today_datetime.date()
            action_timeout[today] = dict()
            if run_id == 0:
                chosen_action_log[today] = dict()
                for action in all_actions:
                    chosen_action_log[today][action.name] = 0
            if run_id == 0:
                chosen_action_log[today]["No Action"] = 0

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
            while len(todays_served_action_propensities) < call_center_daily_quota and len(customers) > 0:
                customer = customers.pop()
                if customer.id not in served_action_propensities:
                    served_action_propensity = policy.get_next_best_action(customer=customer, segment_ids=[1])
                    if served_action_propensity is not None:
                        todays_served_action_propensities.append(served_action_propensity)
                        if served_action_propensity.chosen_action.channel == Channel.OUTBOUND_CALL:
                            todays_call_served_action_propensities.append(served_action_propensity)
                        elif served_action_propensity.chosen_action.channel == Channel.OUTBOUND_EMAIL:
                            todays_email_served_action_propensities.append(served_action_propensity)
                        served_action_propensities[customer.id] = served_action_propensity
                        if run_id == 0:
                            chosen_action_log[today][served_action_propensity.chosen_action.name] += 1
                    else:
                        if run_id == 0:
                            chosen_action_log[today]["No Action"] += 1

            # Look for old actions that have timed out
            if today in action_timeout:
                for served_action_propensity in action_timeout[today]:
                    # Gave up on this customer since they did not convert
                    policy.add_customer_action(served_action_propensity=served_action_propensity,
                                               customer_action=None,
                                               reward=0.0)
            del action_timeout[today]

            # Actually perform the action
            call_counter = 0
            for served_action_propensity in todays_served_action_propensities:
                if call_counter >= call_center_daily_quota:
                    break
                customer = served_action_propensity.customer
                cool_off_days = served_action_propensity.chosen_action.cool_off_days
                deadline = today + timedelta(days=cool_off_days)
                if deadline not in action_timeout:
                    action_timeout[deadline] = dict()
                action_timeout[deadline][customer.id] = served_action_propensity
                customer_action = what_would_a_customer_do(served_action_propensity.customer,
                                                           served_action_propensity.chosen_action,
                                                           today_datetime)
                call_counter += 1
                policy.add_company_action(served_action_propensity,
                                          served_action_propensity.chosen_action,
                                          today,
                                          cost_of_outbound_call)

                # See if we have inidiat reward
                if isinstance(customer_action, Transaction):
                    reward = reward_calculator.calculate(served_action_propensity.customer, customer_action)
                    reward -= cost_of_outbound_call
                    policy.add_customer_action(served_action_propensity=served_action_propensity,
                                               customer_action=customer_action,
                                               reward=reward)

                    deadline = today + timedelta(days=cool_off_days)
                    del action_timeout[deadline][customer.id]

                    cumulative_reward += reward

            log.append({"ts": today, "cumulative_reward": cumulative_reward})
        logs.append(log)
    if run_id == 0:
        # Only the last chosen_action_log
        output.put({"policy": policy_class.__name__, "logs": logs, "chosen_action_log": chosen_action_log})
    else:
        output.put({"policy": policy_class.__name__, "logs": logs})
    print(f"{policy_class.__name__} end")
    return True


def get_timeline_plot(x, y_per_action, label):
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


if __name__ == "__main__":
    policies = [bayesianGroundhog.BayesianGroundhog, randomCrayfish.RandomCrayfish, epsilonRingtail.EpsilonRingtail]
    runs_per_policies = 5
    sequential_runs = 1

    processes = list()
    customers = generate_customers(100000)
    actions = get_actions()
    output_queue = Queue()
    for policy_class in policies:
        for r in range(runs_per_policies):
            keywords = {'epsilon': 0.8, 'resort_batch_size': 50, "initial_trials": 99, "initial_conversions": 1}
            p = Process(target=policy_sim,
                        args=(policy_class, customers, actions, 365, output_queue, r, sequential_runs),
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


    # Plot timelines
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
            for i in range(len(labels)):
                action_name = labels[i]
                if action_name in chosen_action_counts:
                    y_per_action[i].append(chosen_action_counts[action_name])
                else:
                    y_per_action[i].append(0)
        xs[policy_name] = x
        policy_labels[policy_name] = labels
        ys[policy_name] = y_per_action

    # Plot performace
    plot_dfs: Dict[str, DataFrame] = dict()
    last_mean_value: Dict[str, float] = dict()
    for policy, log in all_logs.items():
        for ts, sim_values in log.items():
            plot_dict[policy].append({"ts": ts, "mean": np.mean(sim_values), "std": np.std(sim_values)})
            last_mean_value[policy] = np.mean(sim_values)
        plot_dfs[policy] = DataFrame(plot_dict[policy])

    for policy_name in policy_labels.keys():
        fig = get_timeline_plot(xs[policy_name], ys[policy_name], policy_name)
        plt.show()

    ordered_policies_by_clv = sorted(last_mean_value, key=last_mean_value.get)
    ordered_policies_by_clv.reverse()
    fig = get_performance_plot(plot_dfs, ordered_policies_by_clv)
    fig.savefig("test.png")
    plt.show()
