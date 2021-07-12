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


def policy_sim(policy_class, customers: List[Customer], actions: List[Action], day_count: int, output: Queue) -> DataFrame:
    print(policy_class.__name__)
    policy: Policy = policy_class()
    rewardCalculator = RewardCalculator()

    log: List[Dict[str, Any]] = list()

    servedActionPropensities: Dict[int, ServedActionPropensity] = dict()
    actionTimeout: Dict[datetime, dict[int, ServedActionPropensity]] = dict()

    cumulative_reward = 0.0
    start_date = datetime.today()
    for today_datetime in (start_date + timedelta(n) for n in range(day_count)):
        today = today_datetime.date()
        actionTimeout[today] = dict()

        for action in actions:
            if today == action.start_date:
                policy.add_arm(action, [1])

        todaysServedActionPropensities = list()
        for customer in customers:
            if len(todaysServedActionPropensities) >= call_center_daily_quota:
                break
            if customer.id not in servedActionPropensities:
                servedActionPropensity = policy.get_next_best_action(customer=customer, segment_ids=[1])
                todaysServedActionPropensities.append(servedActionPropensity)
                servedActionPropensities[customer.id] = servedActionPropensity

        if today in actionTimeout:
            for servedActionPropensity in actionTimeout[today]:
                policy.add_customer_action(customer_action=None, reward=0.0)

        # Actually perform the action
        call_counter = 0
        for servedActionPropensity in todaysServedActionPropensities:
            if call_counter >= call_center_daily_quota:
                break
            customer = servedActionPropensity.customer
            cool_off_days = servedActionPropensity.chosen_action.cool_off_days
            deadline = today + timedelta(days=cool_off_days)
            if deadline not in actionTimeout:
                actionTimeout[deadline] = dict()
            actionTimeout[deadline][customer.id] = servedActionPropensity
            customer_action = what_would_a_customer_do(servedActionPropensity.customer, servedActionPropensity.chosen_action, today_datetime)
            call_counter += 1

            # See if we have inidiat reward
            if isinstance(customer_action, Transaction):
                reward = rewardCalculator.calculate(servedActionPropensity.customer, customer_action)
                policy.add_customer_action(customer_action=customer_action, reward=reward)

                deadline = today + timedelta(days=cool_off_days)
                del actionTimeout[deadline][customer.id]
                #del servedActionPropensities[customer.id]

                cumulative_reward += reward

        log.append({"ts": today, "cumulative_reward": cumulative_reward})
    output.put({"policy": policy_class.__name__, "logs": [log]})


if __name__ == "__main__":
    policies = [fierceCrayfish.FierceCrayfish, dashingRingtail.DashingRingtail]

    processes = list()
    customers = generate_customers(100000)
    actions = get_actions()
    output_queue = Queue()
    for policy_class in policies:
        p = Process(target=policy_sim, args=(policy_class, customers, actions, 365, output_queue))
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
        ax.plot(policy["ts"], policy["mean"]/1000)

    ax.set(xlabel='time (days)', ylabel='Cumulative HLV (1000 Euros)',
           title='Policy performance')
    ax.grid()

    fig.savefig("test.png")
    plt.show()


