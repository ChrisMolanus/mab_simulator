from datetime import datetime, timedelta
from random import random
from typing import Dict, List, Set, OrderedDict

from policy import Policy, Action, ServedActionPropensity, CustomerAction, Customer, Transaction, Channel, \
    customer_product_from_product

import numpy as np

from rewardCalculator import RewardCalculator


class Arm:
    def __init__(self, action: Action, number_of_impressions: float = 99.0, number_of_conversions: float = 1.0):
        self.action = action
        self.number_of_impressions = number_of_impressions
        self.number_of_conversions = number_of_conversions
        self.customer_products = list()
        for product in action.offer.products:
            self.customer_products.append(customer_product_from_product(product,
                                          datetime.today().date(),
                                          datetime.today().date() + timedelta(weeks=52)))

    def get_conversion_rate(self) -> float:
        return self.number_of_conversions/self.number_of_impressions

    def __lt__(self, other):
        return self.get_conversion_rate() < other.get_conversion_rate()


class EpsilonRingtail(Policy):
    def __init__(self, epsilon: float, resort_batch_size: int, **kwargs):
        """
        Epsilon greedy
        """
        self.epsilon = epsilon
        self.resort_batch_size = resort_batch_size
        self.action_segments: Dict[str, Set[str]] = dict()
        self.ranked_arms: List[Arm] = list()
        self.action_arms: Dict[str, Arm] = dict()
        self.update_batch_counter = 0

        self.reward_calculator = RewardCalculator()
        self.now_ts = datetime.now()
        self.last_updated = datetime.now() - timedelta(days=1)

    def add_arm(self, action: Action, segment_ids: List[str]):
        arm = Arm(action)
        if action.name not in self.action_arms:
            self.action_segments[action.name] = set(segment_ids)
            self.action_arms[action.name] = arm
            self.ranked_arms.append(arm)
        self.update_customer_products()
        self.last_updated = self.now_ts.date()

    def update_customer_products(self):
        for arm in self.action_arms.values():
            for customer_products in arm.customer_products:
                customer_products.contract_start = self.now_ts.date()
                customer_products.contract_end = self.now_ts.date() + timedelta(weeks=52)

    def add_customer_action(self, served_action_propensity: ServedActionPropensity, customer_action: CustomerAction,
                            reward: float):
        # Check if action resulted in a conversion
        if reward > 0:
            self.action_arms[served_action_propensity.chosen_action.name].number_of_conversions += 1
        self.update_batch_counter += 1
        if self.update_batch_counter > 50:
            self.ranked_arms.sort(reverse=True)
            self.update_batch_counter = 0

    def add_company_action(self, customer: Customer, action: Action, ts: datetime, cost: float):
        self.action_arms[action.name].number_of_impressions += 1
        # This is optional, that if we server an action a lot before we get rewards
        # that is is less likely to be the top action
        self.update_batch_counter += 1
        if self.update_batch_counter > 50:
            self.ranked_arms.sort(reverse=True)
            self.update_batch_counter = 0

    def add_channel_quota(self, channel: Channel, daily_quota: int):
        pass

    def set_datetime(self, now_ts: datetime):
        # Remove Arms/Actions that should no longer be used for NBAs
        arms_to_remove: List[Arm] = list()
        for arm in self.ranked_arms:
            if arm.action.end_date <= now_ts.date():
                arms_to_remove.append(arm)
        for arm in arms_to_remove:
            del self.action_segments[arm.action.name]
            self.ranked_arms.remove(arm)

        self.now_ts = now_ts
        self.update_customer_products()
        self.last_updated = self.now_ts.date()

    def get_next_best_action(self, customer: Customer, segment_ids: List[str]) -> ServedActionPropensity:
        propensities: Dict[str, float] = dict()
        found_top = False
        explore_arms: List[Arm] = list()

        if self.last_updated < self.now_ts.date():
            self.update_customer_products()
            self.last_updated = self.now_ts.date()

        delta_hlvs = list()
        for arm in self.ranked_arms:
            transaction = Transaction(customer=customer, channel=arm.action.channel,
                                      removed=customer.portfolio, added=arm.customer_products, ts=datetime.now())
            delta_hlv = self.reward_calculator.calculate(customer, transaction)
            delta_hlvs.append(delta_hlv)
            # look for action that can be applied to any one of these segments
            if len(self.action_segments[arm.action.name].intersection(set(segment_ids))) > 0 and delta_hlv > 0.0:
                if not found_top:
                    exploit_arm = arm
                    propensities[arm.action.name] = self.epsilon
                    found_top = True
                else:
                    # We already found the Exploit action so the rest are explorer actions
                    explore_arms.append(arm)
            else:
                # This action could not be applied to this customer so the propensity is 0
                propensities[arm.action.name] = 0.0

        # The chances that the customer would be assigned an explorer action is 1 - epsilon
        if len(explore_arms) > 0:
            explore_prob = (1 - self.epsilon) / len(explore_arms)
            for arm in explore_arms:
                propensities[arm.action.name] = explore_prob
        else:
            for arm in explore_arms:
                propensities[arm.action.name] = 0.0

        if found_top:
            exploit = np.random.choice(a=[True, False], p=[self.epsilon, 1 - self.epsilon])
            if exploit:
                # Exploit
                return ServedActionPropensity(customer=customer, chosen_action=exploit_arm.action,
                                              action_propensities=propensities)
            else:
                # Explorer
                return ServedActionPropensity(customer=customer, chosen_action=np.random.choice(explore_arms).action,
                                              action_propensities=propensities)
        else:
            # None of the segment_ids could be found to be applicable to the available actions
            return None
