from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
from numpy import random

from policy import Policy, Action, ServedActionPropensity, CustomerAction, Customer, Channel, Product
from segmentJunglefowl.marketingSegment import GoldSilverBronze, GoldSilverBronzeSegment


class SegmentJunglefowl(Policy):
    def __init__(self, current_base, **kwargs):
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
        self.silver_product_threshold: float = 0.0
        self.gold_product_threshold: float = 0.0

    def add_arm(self, action: Action, segment_ids: List[str]):
        self.actions.append(action)

        for segment_id in segment_ids:
            if segment_id not in self.all_segment_actions:
                self.all_segment_actions[segment_id] = set()
            self.all_segment_actions[segment_id].add(action)

        self.rebalance_product_segments()

    def rebalance_product_segments(self):
        # Re-balance product segmentation
        margins: Dict[float, Set[Tuple[str, Action]]] = dict()
        for segment_id, actions in self.all_segment_actions.items():
            for action_1 in actions:
                margin = action_1.get_max_margin()
                if margin not in margins:
                    margins[margin] = set()
                margins[margin].add((segment_id, action_1))

        triple = np.percentile(list(margins.keys()), [20, 80])

        self.silver_product_threshold = triple[0]
        self.gold_product_threshold = triple[1]
        self.segment_actions: Dict[GoldSilverBronzeSegment, Dict[str, Set[Action]]] = {
            GoldSilverBronzeSegment.Gold: dict(),
            GoldSilverBronzeSegment.Silver: dict(),
            GoldSilverBronzeSegment.Bronze: dict(),
        }

        for margin, pairs in margins.items():
            for segment_id, action_1 in pairs:
                if margin > self.gold_product_threshold:
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

    def add_customer_action(self, served_action_propensity: ServedActionPropensity, customer_action: CustomerAction,
                            reward: float):
        pass

    def add_company_action(self, customer: Customer, action: Action, ts: datetime, cost: float):
        pass

    def add_channel_quota(self, channel: Channel, daily_quota: int):
        pass

    def set_datetime(self, now_ts: datetime):
        action_removed = False
        for our_segment_id, segmented_actions in self.segment_actions.items():
            for segment_id, segment_actions in segmented_actions.items():
                actions_to_remove: Set[Action] = set()
                for action in segment_actions:
                    if action.end_date <= now_ts.date():
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

    def get_next_best_action(self, customer: Customer, segment_ids: List[str]) -> Optional[ServedActionPropensity]:
        our_segment = self.segmentor.get_segments_for_customer(customer, self.last_updated)
        intersecting_segments = set(segment_ids).intersection(self.segment_actions[our_segment].keys())

        allowed_actions: Set[Action] = set()
        for segment_id in intersecting_segments:
            allowed_actions = allowed_actions.union(self.segment_actions[our_segment][segment_id])

        if len(allowed_actions) > 0:
            random_segment_action = random.choice(list(allowed_actions))
            prob = 1 / len(allowed_actions)
            propensities: Dict[str, float] = dict()
            for our_segment_id, segmented_actions in self.segment_actions.items():
                for segment_id, segment_actions in segmented_actions.items():
                    for action in segment_actions:
                        propensities[action.name] = prob if action in allowed_actions else 0.0

            return ServedActionPropensity(customer=customer, chosen_action=random_segment_action,
                                          action_propensities=propensities)
        else:
            return None
