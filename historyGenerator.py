import datetime
import json
from random import seed

import numpy as np
from matplotlib import pyplot as plt

import segmentJunglefowl
from actionGenerator import get_actions
from customerGenerator import generate_customers, get_products
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
                                                                     keywords, segmentJunglefowl.SegmentJunglefowl,
                                                                     reward_calculator, start_date)
