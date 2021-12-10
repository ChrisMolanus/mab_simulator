from datetime import date
from enum import Enum
from typing import Dict, List

import numpy as np

from policy import Customer, Product
from rewardCalculator import HlvCalculator


class SegmentSystem(Enum):
    Gold_Silver_Bronze = "Gold, Silver, Bronze"
    MO_FO_MnF = "Mobile, Fixed, Mobile n Fixed"


class GoldSilverBronzeSegment(Enum):
    Gold = 1
    Silver = 2
    Bronze = 3


class MoFoMnFSegment(Enum):
    MobileOnly = 1
    FixedOnly = 2
    Mobile_and_Fixed = 3
    Lead = 4


class CustomerSegmentor:
    def get_segments_for_customer(self, customer: Customer):
        pass


class GoldSilverBronze(CustomerSegmentor):
    def __init__(self, context_date: date, current_base: List[Customer] = None, gold_threshold: float = None, silver_threshold: float = None):
        """
        A Gold, Silver, Bronze market segmentation model
        :param context_date: The date use to calculate the HLV of the customers
        :param current_base: The current base
        :param gold_threshold: The value to set the Gold threshold to and ignore the customer base (Optional)
        :param silver_threshold: The value to set the Silver threshold to and ignore the customer base (Optional)
        """
        self.hlv_calculator = HlvCalculator()

        if gold_threshold is not None and silver_threshold is not None:
            self.gold_threshold = gold_threshold
            self.silver_threshold = silver_threshold
        elif current_base is None or len(current_base) == 0:
            self.gold_threshold = 300.0
            self.silver_threshold = 200.0
        else:
            self.calibrate(context_date, current_base)

    def calibrate(self, context_date: date, current_base: List[Customer]):
        """
        Calibrate the Gold and Silver thresholds based on the current base.
        Just hack it up into three segments based on absolute customer margin (ignore density)
        :param context_date: The date use to calculate the HLV of the customers
        :param current_base: The current base
        """
        margins = list()
        for customer in current_base:
            margins.append(self.hlv_calculator.get_hlv(customer, context_date))
        triple = np.percentile(margins, [20, 80])

        self.silver_threshold = triple[0]
        self.gold_threshold = triple[1]

    def get_segments_for_customer(self, customer: Customer, context_date: date) -> GoldSilverBronzeSegment:
        """
        Get the segment the customer should be in.
        Since the expected margin of a customer depends on how far into a contract they are the context_date is needed
        :param customer: The customer object
        :param context_date: The date at which to calculate the customers expected margin
        :return:
        """
        margin = self.hlv_calculator.get_hlv(customer, context_date)
        if margin > self.gold_threshold:
            return GoldSilverBronzeSegment.Gold
        elif margin > self.silver_threshold:
            return GoldSilverBronzeSegment.Silver
        else:
            return GoldSilverBronzeSegment.Bronze


class MoFoMnF(CustomerSegmentor):
    def __init__(self, mobile_products: List[Product], fixed_products: List[Product]):
        self.mobile_products = set(mobile_products)
        self.fixed_products = set(fixed_products)

    def get_segments_for_customer(self, customer: Customer) -> MoFoMnFSegment:
        portfolio = set(customer.portfolio)
        m_intersection = portfolio.intersection(self.mobile_products)
        f_intersection = portfolio.intersection(self.fixed_products)

        if len(m_intersection) > 0 and len(f_intersection) > 0:
            return MoFoMnFSegment.Mobile_and_Fixed
        elif len(m_intersection) > 0:
            return MoFoMnFSegment.MobileOnly
        elif len(f_intersection) > 0:
            return MoFoMnFSegment.FixedOnly
        else:
            return MoFoMnFSegment.Lead
