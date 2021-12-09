from datetime import date
from enum import Enum
from typing import Dict, List

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
    def __init__(self, gold_threshold: float = 300.0, silver_threshold: float = 200.0):
        self.gold_threshold = gold_threshold
        self.silver_threshold = silver_threshold
        self.hlv_calculator = HlvCalculator()

    def get_segments_for_customer(self, customer: Customer, context_date: date) -> GoldSilverBronzeSegment:
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
