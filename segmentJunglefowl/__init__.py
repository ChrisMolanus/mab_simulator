from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional

from numpy import random

from customerGenerator import get_products
from policy import Policy, Action, ServedActionPropensity, CustomerAction, Customer, Channel
from rewardCalculator import HlvCalculator
from segmentJunglefowl.marketingSegment import GoldSilverBronze, GoldSilverBronzeSegment




class SegmentJunglefowl(Policy):
    def __init__(self, **kwargs):
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
        """
        self.now_ts = datetime.now()
        self.last_updated = datetime.now() - timedelta(days=1)

        self.segment_actions: Dict[str, List[Action]] = dict()

        product_list, _ = get_products()
        self.products = set(product_list)
        self.segmentor = GoldSilverBronze()
        self.hlv_calculator = HlvCalculator()

    def add_arm(self, action: Action, segment_ids: List[str]):
        for segment_id in segment_ids:
            if segment_id not in self.segment_actions:
                self.segment_actions[segment_id] = list()
            self.segment_actions[segment_id].append(action)

        for product in action.offer.products:
            if product not in self.products:
                self.products.add(product)

    def add_customer_action(self, served_action_propensity: ServedActionPropensity, customer_action: CustomerAction,
                            reward: float):
        pass

    def add_company_action(self, customer: Customer, action: Action, ts: datetime, cost: float):
        pass

    def add_channel_quota(self, channel: Channel, daily_quota: int):
        pass

    def set_datetime(self, now_ts: datetime):
        for segment_id, segment_actions in self.segment_actions.items():
            indexes_to_remove = list()
            for i in range(0, len(segment_actions)):
                if segment_actions[i].end_date <= now_ts.date():
                    indexes_to_remove.append(i)

            indexes_to_remove.sort(reverse=True)
            if len(indexes_to_remove) != len(self.segment_actions[segment_id]):
                for i in indexes_to_remove:
                    del self.segment_actions[segment_id][i]
            else:
                del self.segment_actions[segment_id]

        self.now_ts = now_ts
        self.last_updated = self.now_ts.date()

    def get_next_best_action(self, customer: Customer, segment_ids: List[str]) -> Optional[ServedActionPropensity]:



        random_action = random.choice(self.segment_actions[random_segment])
        propensities: Dict[str, float] = dict()
        for segment_id, segment_actions in self.segment_actions.items():
            for action in segment_actions:
                propensities[action.name] = (1.0 / len(intersecting_segments)) * (1.0 / len(segment_actions))

        return ServedActionPropensity(customer=customer, chosen_action=random_action,
                                      action_propensities=propensities)

    def get_action_from_segment(self, customer: Customer, their_segments: List[str]):
        intersecting_segments = set(their_segments).intersection(self.segment_actions.keys())
        allowed_actions: Dict[GoldSilverBronzeSegment, Set[Action]] = dict()
        for segment_id in intersecting_segments:
            for segment_actions in self.segment_actions[segment_id]:
                margin = 0.0
                for product in segment_actions.offer:
                    margin += product.get_margin()
                allowed_actions.union(set(segment_actions))


        our_segment = self.segmentor.get_segments_for_customer(customer, self.last_updated)
        if our_segment == GoldSilverBronzeSegment.Gold:

