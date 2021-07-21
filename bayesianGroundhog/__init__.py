from datetime import datetime
from typing import Dict, List, Set

import numpy as np
from scipy import stats

from policy import Policy, Action, ServedActionPropensity, CustomerAction, Customer, Channel, Transaction
from rewardCalculator import RewardCalculator


class Arm:
    def __init__(self, action: Action, initial_trials, initial_conversions):
        """The continuous probability distribution of the conversion rate of an Next Best Offer
        :param action: The action for which this arm represents the conversion rate
        :param initial_trials:
        :param initial_conversions:
        """
        self.action = action
        self.alpha = initial_conversions
        self.beta = initial_trials

    def update_belief(self, n_trials: float = 99, n_conversions: float = 1):
        self.alpha += n_conversions
        self.beta += n_trials

    def sample(self, size=1):
        #return stats.beta(a=1+self.alpha, b=1+self.beta)
        return np.random.beta(1+self.alpha, 1+self.beta, size=size)

    def __str__(self):
        return f'alpha={self.alpha}, beta={self.beta}'


class BayesianGroundhog(Policy):
    # https://peterroelants.github.io/posts/multi-armed-bandit-implementation/
    def __init__(self, initial_trials: float = 99, initial_conversions: int = 1, **kwargs):
        """
        Bayesian Bandit
        """
        self.action_segments: Dict[str, Set[str]] = dict()
        self.action_arms: Dict[str, Arm] = dict()
        self.initial_trials = initial_trials
        self.initial_conversions = initial_conversions

        self.reward_calculator = RewardCalculator()

    def add_arm(self, action: Action, segment_ids: List[str]):
        arm = Arm(action, self.initial_trials, self.initial_conversions)
        if action.name not in self.action_arms:
            self.action_segments[action.name] = set(segment_ids)
            self.action_arms[action.name] = arm

    def add_customer_action(self, served_action_propensity: ServedActionPropensity, customer_action: CustomerAction,
                            reward: float):
        # Check if action resulted in a conversion
        if reward > 0:
            self.action_arms[served_action_propensity.chosen_action.name].update_belief(n_trials=1, n_conversions=1)
        else:
            self.action_arms[served_action_propensity.chosen_action.name].update_belief(n_trials=1, n_conversions=0)

    def add_company_action(self, customer: Customer, action: Action, ts: datetime, cost: float):
        pass
        # self.action_arms[action.name].update_belief(n_trials=1, n_conversions=0)
        # This is optional, that if we server an action a lot before we get rewards
        # that is is less likely to be the top action
        # Otherwise we could only add the trails in add_customer_action()

    def add_channel_quota(self, channel: Channel, daily_quota: int):
        pass

    def set_datetime(self, now_ts: datetime):
        pass

    def get_next_best_action(self, customer: Customer, segment_ids: List[str]) -> ServedActionPropensity:
        propensities: Dict[str, float] = dict()

        delta_hlvs: List[float] = list()
        expected_delta_hlvs: Dict[float, Arm] = dict()
        top_expected_delta_hlv = 0
        top_arm: Arm = None
        for action_name, arm in self.action_arms.items():
            transaction = Transaction(customer=customer, channel=arm.action.channel,
                                      removed=customer.portfolio, added=arm.action.offer.products, ts=datetime.now())
            delta_hlv = self.reward_calculator.calculate(customer, transaction)
            delta_hlvs.append(delta_hlv)

            # look for action that can be applied to any one of these segments
            if len(self.action_segments[arm.action.name].intersection(set(segment_ids))) > 0 and delta_hlv > 0.0:
                sample_conversion_rate = arm.sample(1)[0]
                expected_delta_hlv = sample_conversion_rate*delta_hlv
                expected_delta_hlvs[expected_delta_hlv] = arm
                if expected_delta_hlv > top_expected_delta_hlv:
                    top_expected_delta_hlv = expected_delta_hlv
                    top_arm = arm
                # Hard to saw what this should be
                propensities[arm.action.name] = 0.0
            else:
                # This action could not be applied to this customer so the propensity is 0
                propensities[arm.action.name] = 0.0

        if top_arm is not None:
            return ServedActionPropensity(customer=customer, chosen_action=top_arm.action,
                                          action_propensities=propensities)
        else:
            # None of the segment_ids could be found to be applicable to the available actions
            return None