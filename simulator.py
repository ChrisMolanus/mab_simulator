from typing import Dict, List
from multiprocessing import Process

import dashingRingtail
import fierceCrayfish
from policy import ServedActionPropensity, Policy, Customer


def policy_sim(policy, customers: List[Customer]):
    # policy = policy
    # customers = customers

    sservedActionPropensities: Dict[int, ServedActionPropensity] = dict()

def generate_customers() -> List[Customer]:


if __name__ == "__main__":
    policies = [fierceCrayfish, dashingRingtail]

    processes = list()
    customers = list()
    for policy in policies:
        processes.append(Process(target=policy_sim, args=(policy, customers)))