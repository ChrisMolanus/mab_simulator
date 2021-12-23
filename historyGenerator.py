import datetime
import json
import os
from random import seed

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import segmentJunglefowl
from actionGenerator import get_actions
from customerGenerator import generate_customers, get_products
from policy import get_channel_action_cost
from rewardCalculator import RewardCalculator
from simulator import sim_cycle_run

#if __name__ == "__main__":
seed(7837)
np.random.seed(7837)
start_ts = datetime.datetime(2011, 1, 1)
sim_start_date = start_ts.date()

reward_calculator = RewardCalculator()
products, product_market_size = get_products(sim_start_date - datetime.timedelta(days=2190), sim_start_date)
product_market_sizes = [0.0] * len(product_market_size)

day_count = 365
# Set the base product distribution to how it was on start_date
product_market_sizes[0] = 0.19
product_market_sizes[1] = 0.51
product_market_sizes[2] = 0.3
all_customers = generate_customers(100000, sim_start_date, product_market_sizes)
keywords = {"current_base": all_customers}
#products, product_market_size = get_products()
all_actions = get_actions()

log, chosen_action_log, historicalActionPropensities = sim_cycle_run(all_actions, all_customers, day_count,
                                                                     segmentJunglefowl.SegmentJunglefowl,
                                                                     reward_calculator, [], start_ts, **keywords)
output_dir = "output"

# Export customer
customer_list = list()
for customer in all_customers:
    customer_list.append(
        {
            "id":customer.id,
            "name": customer.name,
            "dob": customer.dob,
            "billing_postcode": customer.billing_address.postcode,
            "billing_house_number": customer.billing_address.house_number
        }
    )
customers = pd.DataFrame(customer_list)
customers.to_parquet(os.path.join(output_dir, "customers.parquet"), index=False)
del customers

# Export actions
action_list = list()
action_product_list = list()
for action in all_actions:
    action_list.append({
        "name": action.name,
        "start_date": action.start_date,
        "end_date": action.end_date,
        "cool_off_days": action.cool_off_days,
        "channel": str(action.channel),
        "offer_name": action.offer.name,
        "max_margin": action.get_max_margin(years_horizon=5),
        "cost": get_channel_action_cost(action.channel)
    })
    for product in action.offer.products:
        action_product_list.append({
            "action_name": action.name,
            "product_name": product.name
        })
actions = pd.DataFrame(action_list)
actions.to_parquet(os.path.join(output_dir, "actions.parquet"), index=False)
del actions

# Export Product / Action link
action_product = pd.DataFrame(action_product_list)
action_product.to_parquet(os.path.join(output_dir, "action_product.parquet"), index=False)
del action_product

# Export Transactions
transaction_list = list()
for t in historicalActionPropensities:
    transaction_list.append({
        "id": t.customer.id,
        "action_ts": t.action_ts,
        "action_name": t.chosen_action.name,
        "reward_ts": t.reward_ts,
        "reward": t.reward,
    })
transactions = pd.DataFrame(transaction_list)
transactions.to_parquet(os.path.join(output_dir, "transactions.parquet"), index=False)
del transactions
