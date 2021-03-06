from datetime import datetime
from typing import Dict, List, Set, Optional

import numpy as np

from policy import Policy, Action, ServedActionPropensity, CustomerAction, Customer, Channel, HistoricalActionPropensity
from rewardCalculator import RewardCalculator


class Arm:
    def __init__(self, action: Action, initial_trials, initial_conversions):
        """The continuous probability distribution of the conversion rate of an Next Best Offer
        :param action: The action for which this arm represents the conversion rate
        :param initial_trials: The initial number of impressions to start with
        :param initial_conversions: The initial number of conversions to start with
        """
        self.action = action
        self.alpha = initial_conversions
        self.beta = initial_trials
        self.sum_of_rewards = 1.0

    def update_belief(self, n_trials: float = 99, n_conversions: float = 1, reward: float = 0.0):
        """
        Update the beta distribution with the new data
        :param n_trials: The number of attempts made in this sample set
        :param n_conversions: The number of conversions made in this sample set
        :param reward: The sum of the reward see from the conversions in this sample set
        """
        self.alpha += n_conversions
        self.beta += n_trials
        self.sum_of_rewards += reward

    def sample(self, size=1):
        """
        Draw a sample from the distribution representing our belief in the conventions rate of this arm/action
        :param size: The number of samples, Default is 1
        :return: A sample conversion rate
        """
        return np.random.beta(1 + self.alpha, 1 + self.beta, size=size)

    def get_expected_reward(self) -> float:
        """
        Sample an expected reward based on a sample conversion rate and the current sum of rewards
        sample_conversion * average_reward
        :return: The sampled expected reward
        """
        # sample * average reward
        return self.sample()[0] * (self.sum_of_rewards / self.alpha)

    def __str__(self):
        return f'alpha={self.alpha}, beta={self.beta}'


class BayesianGroundhog(Policy):
    icon = "https://cdn-icons-png.flaticon.com/512/185/185716.png"

    # https://peterroelants.github.io/posts/multi-armed-bandit-implementation/
    def __init__(self, history: List[HistoricalActionPropensity], initial_trials: float = 99, initial_conversions: int = 1, **kwargs):
        """
        Bayesian Bandit
        """
        self.action_segments: Dict[str, Set[str]] = dict()
        self.action_arms: Dict[str, Arm] = dict()
        self.initial_trials = initial_trials
        self.initial_conversions = initial_conversions

        self.reward_calculator = RewardCalculator()
        self.now_ts = datetime.now()

    def add_arm(self, action: Action, segment_ids: List[str]):
        arm = Arm(action, self.initial_trials, self.initial_conversions)
        if action.name not in self.action_arms:
            self.action_segments[action.name] = set(segment_ids)
            self.action_arms[action.name] = arm

    def add_customer_action(self, served_action_propensity: ServedActionPropensity, customer_action: CustomerAction,
                            reward: float):
        # Check if action resulted in a conversion
        if reward > 0:
            self.action_arms[served_action_propensity.chosen_action.name].update_belief(n_trials=1, n_conversions=1,
                                                                                        reward=reward)
        else:
            self.action_arms[served_action_propensity.chosen_action.name].update_belief(n_trials=1, n_conversions=0,
                                                                                        reward=reward)

    def add_company_action(self, customer: Customer, action: Action, ts: datetime, cost: float):
        pass
        # self.action_arms[action.name].update_belief(n_trials=1, n_conversions=0)
        # This is optional, that if we server an action a lot before we get rewards
        # that is is less likely to be the top action
        # Otherwise we could only add the trails in add_customer_action()

    def add_channel_quota(self, channel: Channel, daily_quota: int):
        pass

    def set_datetime(self, now_ts: datetime):
        # Remove Arms/Actions that should no longer ne suggested as NBAs
        arms_to_remove: List[Arm] = list()
        for action_name, arm in self.action_arms.items():
            if arm.action.end_date <= now_ts.date():
                arms_to_remove.append(arm)
        for arm in arms_to_remove:
            del self.action_segments[arm.action.name]
            del self.action_arms[arm.action.name]

        self.now_ts = now_ts

    def get_next_best_action(self, customer: Customer, segment_ids: List[str]) -> Optional[ServedActionPropensity]:
        propensities: Dict[str, float] = dict()

        expected_delta_hlvs: Dict[float, Arm] = dict()
        top_expected_delta_hlv = 0
        top_arm: Optional[Arm] = None
        for action_name, arm in self.action_arms.items():
            expected_delta_hlv = arm.get_expected_reward()

            # look for action that can be applied to any one of these segments
            if len(self.action_segments[arm.action.name].intersection(
                    set(segment_ids))) > 0 and expected_delta_hlv > 0.0:
                expected_delta_hlvs[expected_delta_hlv] = arm
                if expected_delta_hlv > top_expected_delta_hlv:
                    top_expected_delta_hlv = expected_delta_hlv
                    top_arm = arm
                # Hard to calculate what this should be
                # TODO: use arm distributions in region covered by the arm with a tail in the highest conversion rate
                # Use area under the curve of normalized distributions as the propensities that that arm
                # could be the one that gave the winning conventions rate
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
