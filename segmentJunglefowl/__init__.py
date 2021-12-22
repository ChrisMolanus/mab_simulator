from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
from numpy import random

from policy import Policy, Action, ServedActionPropensity, CustomerAction, Customer, Channel, Product, CustomerProduct, \
    customer_product_from_product, HistoricalActionPropensity
from segmentJunglefowl.marketingSegment import GoldSilverBronze, GoldSilverBronzeSegment


class SegmentJunglefowl(Policy):
    icon = "https://icon-library.com/images/rooster-icon/rooster-icon-13.jpg"

    def __init__(self, history: List[HistoricalActionPropensity], current_base, **kwargs):
        """
        Simulates the logic of a marketing department using a Gold, Silver, Bronze segmenting system.
        Gold customers are ones that have the highest current margin.
        Bronze customers are the ones that have the lowest margin.
        Silver customers are the group in between.

        We fist collect all actions that can be performed.
        An action can be performed if the resulting transaction will not cause us to make less margin.
        This considered that an Up-Sell will require and existing product in the customer portfolio
        to be replaced by another product.

        We can perform a Up-Sell or a Cross-Sell
        If both a Up and Cross Sell can be performed we choose one at random.
        Of the actions possible in the Cross/Up Sell we choose a action at random.

        :param current_base: The current install base
        """
        self.now_ts = datetime.now()
        self.last_updated = datetime.now() - timedelta(days=1)

        self.segment_actions: Dict[str, List[Action]] = dict()
        self.current_base = current_base
        self.all_segment_actions: Dict[str, Set[Action]] = dict()
        self.segmentor = GoldSilverBronze(self.last_updated, self.current_base)
        self.segment_actions: Dict[GoldSilverBronzeSegment, Dict[str, Set[Action]]] = {
            GoldSilverBronzeSegment.Gold: dict(),
            GoldSilverBronzeSegment.Silver: dict(),
            GoldSilverBronzeSegment.Bronze: dict(),
        }
        self.actions: List[Action] = list()
        self.action_rewards: Dict[Action, List[float]] = dict()
        self.action_reward_last_check: Dict[Action, datetime] = dict()
        self.silver_product_threshold: float = 0.0
        self.gold_product_threshold: float = 0.0

    def add_arm(self, action: Action, segment_ids: List[str]):
        self.actions.append(action)
        self.action_rewards[action] = list()
        self.action_reward_last_check[action] = self.now_ts

        for segment_id in segment_ids:
            if segment_id not in self.all_segment_actions:
                self.all_segment_actions[segment_id] = set()
            self.all_segment_actions[segment_id].add(action)

        self.rebalance_product_segments()

    def rebalance_product_segments(self):
        """
        Re-balance the product segmentation
        :return:
        """

        # Get all margins of the actions we have
        margins: Dict[float, Set[Tuple[str, Action]]] = dict()
        for segment_id, actions in self.all_segment_actions.items():
            for action_1 in actions:
                margin = action_1.get_max_margin(years_horizon=5)
                if margin not in margins:
                    margins[margin] = set()
                margins[margin].add((segment_id, action_1))

        # triple = np.percentile(list(margins.keys()), [20, 80])
        #
        # self.silver_product_threshold = triple[0]
        # self.gold_product_threshold = triple[1]
        # self.segment_actions: Dict[GoldSilverBronzeSegment, Dict[str, Set[Action]]] = {
        #     GoldSilverBronzeSegment.Gold: dict(),
        #     GoldSilverBronzeSegment.Silver: dict(),
        #     GoldSilverBronzeSegment.Bronze: dict(),
        # }
        self.silver_product_threshold = self.segmentor.silver_threshold
        self.gold_product_threshold = self.segmentor.gold_threshold

        # Clear the current segment_actions
        self.segment_actions: Dict[GoldSilverBronzeSegment, Dict[str, Set[Action]]] = {
            GoldSilverBronzeSegment.Gold: dict(),
            GoldSilverBronzeSegment.Silver: dict(),
            GoldSilverBronzeSegment.Bronze: dict(),
        }

        # First make a logical deviation of actions over the customer segments
        for margin, pairs in margins.items():
            for segment_id, action_1 in pairs:
                if margin > self.gold_product_threshold :
                    if segment_id not in self.segment_actions[GoldSilverBronzeSegment.Gold]:
                        self.segment_actions[GoldSilverBronzeSegment.Gold][segment_id] = set()
                    self.segment_actions[GoldSilverBronzeSegment.Gold][segment_id].add(action_1)
                elif margin > self.silver_product_threshold:
                    if segment_id not in self.segment_actions[GoldSilverBronzeSegment.Silver]:
                        self.segment_actions[GoldSilverBronzeSegment.Silver][segment_id] = set()
                    self.segment_actions[GoldSilverBronzeSegment.Silver][segment_id].add(action_1)
                else:
                    if segment_id not in self.segment_actions[GoldSilverBronzeSegment.Bronze]:
                        self.segment_actions[GoldSilverBronzeSegment.Bronze][segment_id] = set()
                    self.segment_actions[GoldSilverBronzeSegment.Bronze][segment_id].add(action_1)

        # It can happen that the Product-Pricing-Board did not allow for
        # enough products that fit into every customer segment
        # So we add at the cheapest product from the higher segment
        sorted_margins = list(margins.keys())
        sorted_margins.sort()
        for our_segment, segment_actions in self.segment_actions.items():

            if len(segment_actions) == 0:
                for segment_id in self.all_segment_actions.keys():
                    segment_actions[segment_id] = set()

            for segment_id, actions in segment_actions.items():
                if len(actions) == 0:
                    # Fine the cheapest offer that has more margin than the segmentor threshold
                    if our_segment == GoldSilverBronzeSegment.Gold:
                        segmentor_threshold = self.segmentor.gold_threshold
                    elif our_segment == GoldSilverBronzeSegment.Silver:
                        segmentor_threshold = self.segmentor.silver_threshold
                    elif our_segment == GoldSilverBronzeSegment.Bronze:
                        segmentor_threshold = 0.0

                    found_one = False
                    for margin in sorted_margins:
                        if margin > segmentor_threshold:
                            for pair in margins[margin]:
                                if pair[0] == segment_id:
                                    actions.add(pair[1])
                                    # We only need one
                                    found_one = True
                                    break
                            if found_one:
                                break

    def add_customer_action(self, served_action_propensity: ServedActionPropensity, customer_action: CustomerAction,
                            reward: float):
        self.action_rewards[served_action_propensity.chosen_action].append(reward)

    def add_company_action(self, customer: Customer, action: Action, ts: datetime, cost: float):
        pass

    def add_channel_quota(self, channel: Channel, daily_quota: int):
        pass

    def should_remove_action(self, action, now_ts) -> bool:
        """
        Determines if a marketeer would have enough reason to say that this campaign(action) will not make us money
        :param action: The campaign
        :param now_ts: The current timestamp
        :return: True if a marketeer would have decided that the campaign (action) is no longer worth wile
        """
        d = now_ts - self.action_reward_last_check[action]
        if d.days > 30:
            self.action_reward_last_check[action] = now_ts
            if len(self.action_rewards[action]) > 50:
                return sum(self.action_rewards[action]) / len(self.action_rewards[action]) <= 0.0
            else:
                return False
        else:
            return False

    def set_datetime(self, now_ts: datetime):
        action_removed = False
        for our_segment_id, segmented_actions in self.segment_actions.items():
            for segment_id, segment_actions in segmented_actions.items():
                actions_to_remove: Set[Action] = set()
                for action in segment_actions:
                    if action.end_date <= now_ts.date():
                        actions_to_remove.add(action)

                    # Evaluate actions to see if they are making money on average, remove if not
                    if self.should_remove_action(action=action, now_ts=now_ts):
                        actions_to_remove.add(action)

                if len(actions_to_remove) > 0:
                    for acton in actions_to_remove:
                        segment_actions.remove(acton)
                    action_removed = True
                # TODO: clean up empty segment_id

        if action_removed:
            self.rebalance_product_segments()

        self.now_ts = now_ts
        self.last_updated = self.now_ts.date()

    def passes_sanity_check(self, customer: Customer, action: Action) -> bool:
        """
        Would a marketeer think if would make sense to try and sell this customer this offer
        :param customer: The customer that might want the offer
        :param action: The action that contains the offer
        :return: False is a marketeer would say a logical customer might buy the offer
        """
        # No one would want slower Internet
        if customer.portfolio[0].kwargs["download_speed"] < action.offer.products[0].kwargs["download_speed"]:
            # Calculate the HLV that the customer generates now
            before_hlv = self.segmentor.hlv_calculator.get_hlv(customer, self.last_updated)

            # Calculate what the HVL would be if the customer bought the product
            fake_portfolios: List[CustomerProduct] = list()
            p: Product
            for p in action.offer.products:
                contract_start = self.last_updated
                contract_end = (contract_start + timedelta(weeks=52))
                cp = customer_product_from_product(p, contract_start, contract_end)
                fake_portfolios = [cp]

            fake_customer = Customer(id=customer.id,
                                     name=customer.name,
                                     dob=customer.dob,
                                     billing_address=customer.billing_address,
                                     portfolio=fake_portfolios
                                     )
            after_hlv = self.segmentor.hlv_calculator.get_hlv(fake_customer, self.last_updated)

            # Our company would not sell a product to a customer where we loss money
            return after_hlv >= before_hlv
        else:
            return False

    def get_next_best_action(self, customer: Customer, segment_ids: List[str]) -> Optional[ServedActionPropensity]:
        our_segment = self.segmentor.get_segments_for_customer(customer, self.last_updated)
        intersecting_segments = set(segment_ids).intersection(self.segment_actions[our_segment].keys())

        allowed_actions: Dict[Action, float] = dict()
        for segment_id in intersecting_segments:
            for action in self.segment_actions[our_segment][segment_id]:
                # Because this would be done by marketers with domain specific knowledge they can do sanity checks
                if self.passes_sanity_check(customer, action):
                    allowed_actions[action] = action.get_max_margin()
        if len(allowed_actions) > 0:
            sorted_allowed_actions = [k for k, v in sorted(allowed_actions.items(), key=lambda item: item[1])]
            l = len(sorted_allowed_actions)
            s = (l * (1 + l)) / 2
            d = (1 / s)
            p = np.arange(d, (l+1)*d, d)
            p = p[0:l]
            p[::-1].sort()
            random_segment_action = random.choice(sorted_allowed_actions, p=p)
            prob = 1 / len(sorted_allowed_actions)
            propensities: Dict[str, float] = dict()
            for our_segment_id, segmented_actions in self.segment_actions.items():
                for segment_id, segment_actions in segmented_actions.items():
                    for action in segment_actions:
                        propensities[action.name] = prob if action in sorted_allowed_actions else 0.0

            return ServedActionPropensity(customer=customer, chosen_action=random_segment_action,
                                          action_propensities=propensities)
        else:
            # No actions that would be profitable
            return None
