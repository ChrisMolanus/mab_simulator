import csv
from datetime import datetime
from typing import Dict, List
from multiprocessing import Process

import dashingRingtail
import fierceCrayfish
from actionGenerator import get_actions
from customerGenerator import generate_customers
from policy import ServedActionPropensity, Policy, Customer, Address, Product, Channel, Action, Offer


def policy_sim(policy_class, customers: List[Customer], actions: List[Action]):
    print(policy_class.__name__)
    policy: Policy = policy_class()




    servedActionPropensities: Dict[int, ServedActionPropensity] = dict()





if __name__ == "__main__":
    policies = [fierceCrayfish.FierceCrayfish, dashingRingtail.DashingRingtail]

    processes = list()
    customers = generate_customers(10000)
    actions = get_actions()
    for policy_class in policies:
        p = Process(target=policy_sim, args=(policy_class, customers, actions))
        p.start()
        processes.append(p)
