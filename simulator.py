import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any
from multiprocessing import Process, Queue

from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

import dashingRingtail
import fierceCrayfish
from actionGenerator import get_actions
from customerGenerator import generate_customers, what_would_a_customer_do
from policy import ServedActionPropensity, Policy, Customer, Address, Product, Channel, Action, Offer, Transaction
from rewardCalculator import RewardCalculator

call_center_daily_quota = 200
cost_of_outbound_call = 8


def policy_sim(policy_class, all_customers: List[Customer], actions: List[Action], day_count: int, output: Queue, run_id:int, sequential_runs: int) -> DataFrame:
    print(policy_class.__name__ + str(run_id))
    reward_calculator = RewardCalculator()

    logs: List[List[Dict[str, Any]]] = list()

    for s_run in range(sequential_runs):
        log: List[Dict[str, Any]] = list()
        customers = all_customers.copy()
        policy: Policy = policy_class()
        today = datetime.today().date()
        for action in actions:
            # So not today == action.start_date, that is later
            if action.start_date < today < action.end_date:
                policy.add_arm(action, [1])
        served_action_propensities: Dict[int, ServedActionPropensity] = dict()
        action_timeout: Dict[datetime, dict[int, ServedActionPropensity]] = dict()

        cumulative_reward = 0.0
        start_date = datetime.today()
        for today_datetime in (start_date + timedelta(n) for n in range(day_count)):
            today = today_datetime.date()
            action_timeout[today] = dict()

            policy.set_datetime(today_datetime)

            # Add new actions that are available from today
            for action in actions:
                if today == action.start_date:
                    policy.add_arm(action, [1])

            # Select a set of people that we can call today
            todays_served_action_propensities = list()
            while len(todays_served_action_propensities) < call_center_daily_quota and len(customers) > 0:
                customer = customers.pop()
                if customer.id not in served_action_propensities:
                    served_action_propensity = policy.get_next_best_action(customer=customer, segment_ids=[1])
                    if served_action_propensity is not None:
                        todays_served_action_propensities.append(served_action_propensity)
                        served_action_propensities[customer.id] = served_action_propensity

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
                customer_action = what_would_a_customer_do(served_action_propensity.customer, served_action_propensity.chosen_action, today_datetime)
                call_counter += 1

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
    output.put({"policy": policy_class.__name__, "logs": logs})


if __name__ == "__main__":
    policies = [fierceCrayfish.FierceCrayfish, dashingRingtail.DashingRingtail]
    runs_per_policies = 2
    sequential_runs = 5

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
        #p.join()
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

    fig.savefig("test.png")
    plt.show()


