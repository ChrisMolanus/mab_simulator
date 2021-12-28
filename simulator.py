import csv
from datetime import datetime, timedelta
from random import seed
from typing import Dict, List, Any, Tuple
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
from customerGenerator import generate_customers, what_would_a_customer_do
from policy import ServedActionPropensity, Policy, Customer, Channel, Action, Transaction, HistoricalActionPropensity, \
    get_channel_action_cost
from rewardCalculator import RewardCalculator

call_center_daily_quota = 200


def policy_sim(policy_class, all_customers: List[Customer], all_actions: List[Action], day_count: int, output: Queue,
               run_id: int, sequential_runs: int, history: List[HistoricalActionPropensity],
               start_ts: datetime = datetime.today(), **kwargs) -> DataFrame:
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
    # policies = [randomCrayfish.RandomCrayfish, segmentJunglefowl.SegmentJunglefowl, bayesianGroundhog.BayesianGroundhog,
    #             epsilonRingtail.EpsilonRingtail]
    policies = [segmentJunglefowl.SegmentJunglefowl]
    runs_per_policies = 1
    sequential_runs = 1

    processes = list()
    start_ts = datetime.today()
    customers = generate_customers(100000, start_ts.date())
    actions = get_actions()
    output_queue = Queue()
    history: List[HistoricalActionPropensity] = list()
    for policy_class in policies:
        for r in range(runs_per_policies):
            keywords = {'epsilon': 0.8, 'resort_batch_size': 50, "initial_trials": 99, "initial_conversions": 1,
                        "current_base": customers}
            p = Process(target=policy_sim,
                        args=(policy_class, customers, actions, 365, output_queue, r, sequential_runs, history, start_ts),
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

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for policy_name, log in all_logs.items():
        with open(os.path.join(output_dir, policy_name + ".yaml"), "w") as f:
            f.write(yaml.safe_dump(log))


    # Plot policy timelines
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

    # Plot performance
    plot_dfs: Dict[str, DataFrame] = dict()
    last_mean_value: Dict[str, float] = dict()
    for policy, log in all_logs.items():
        for ts, sim_values in log.items():
            plot_dict[policy].append({"ts": ts, "mean": np.mean(sim_values), "std": np.std(sim_values)})
            last_mean_value[policy] = np.mean(sim_values)
        plot_dfs[policy] = DataFrame(plot_dict[policy])

    for policy_name in policy_labels.keys():
        fig = get_timeline_plot(xs[policy_name], ys[policy_name], policy_name)
        fig.savefig(f"{policy_name}.png")
        plt.show()

    ordered_policies_by_clv = sorted(last_mean_value, key=last_mean_value.get)
    ordered_policies_by_clv.reverse()
    fig = get_performance_plot(plot_dfs, ordered_policies_by_clv)
    fig.savefig("test.png")
    plt.show()
