import datetime
import os
from random import seed
from typing import List, Tuple

import numpy as np
import pandas as pd

import segmentJunglefowl
from actionGenerator import get_actions
from customerGenerator import generate_customers, get_products
from policy import get_channel_action_cost, HistoricalActionPropensity, Customer, Action
from rewardCalculator import RewardCalculator
from simulator import sim_cycle_run




if __name__ == "__main__":
    historical_action_propensities, all_customers, all_actions = get_history(export=True)
