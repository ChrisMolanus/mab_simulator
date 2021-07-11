import csv
from datetime import datetime, timedelta
from typing import Dict, List
from multiprocessing import Process

import dashingRingtail
import fierceCrayfish
from actionGenerator import get_actions
from customerGenerator import generate_customers, what_would_a_customer_do
from policy import ServedActionPropensity, Policy, Customer, Address, Product, Channel, Action, Offer, Transaction
from rewardCalculator import RewardCalculator


def policy_sim(policy_class, customers: List[Customer], actions: List[Action], day_count: int):
    print(policy_class.__name__)
    policy: Policy = policy_class()
    rewardCalculator = RewardCalculator()

    servedActionPropensities: Dict[int, ServedActionPropensity] = dict()
    actionTimeout: Dict[datetime, dict[int, ServedActionPropensity]] = dict()

    start_date = datetime.today()
    for today_datetime in (start_date + timedelta(n) for n in range(day_count)):
        today = today_datetime.date()
        actionTimeout[today] = dict()

        for action in actions:
            if today == action.start_date:
                policy.add_arm(action, [1])

        todaysServedActionPropensities = list()
        for customer in customers:
            if customer.id not in servedActionPropensities:
                servedActionPropensity = policy.get_next_best_action(customer=customer, segment_ids=[1])
                todaysServedActionPropensities.append(servedActionPropensity)
                servedActionPropensities[customer.id] = servedActionPropensity


        if today in actionTimeout:
            for servedActionPropensity in actionTimeout[today]:
                policy.add_customer_action(customer_action=None, reward=0.0)

        # Actually perform the action
        for servedActionPropensity in todaysServedActionPropensities:
            cool_off_days = servedActionPropensity.chosen_action.cool_off_days
            deadline = today + timedelta(days=cool_off_days)
            if deadline not in actionTimeout:
                actionTimeout[deadline] = dict()
            actionTimeout[deadline][customer.id] = servedActionPropensity
            customerAction = what_would_a_customer_do(servedActionPropensity.customer, servedActionPropensity.chosen_action)

            # See if we have inidiat reward
            if isinstance(customerAction, Transaction):
                reward = rewardCalculator.calculate(servedActionPropensity.customer, customerAction)
                policy.add_customer_action(customer_action=customerAction, reward=reward)

                del actionTimeout[today + timedelta(days=cool_off_days)][customer.id]
                del servedActionPropensities[customer.id]







if __name__ == "__main__":
    policies = [fierceCrayfish.FierceCrayfish, dashingRingtail.DashingRingtail]

    processes = list()
    customers = generate_customers(10000)
    actions = get_actions()
    for policy_class in policies:
        p = Process(target=policy_sim, args=(policy_class, customers, actions))
        p.start()
        processes.append(p)
