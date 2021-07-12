from typing import Dict, List
from itertools import chain

from policy import Customer, Transaction, Product


class RewardCalculator:
    def __init__(self):
        self.hlv_calculator = HlvCalculator()

    def calculate(self, customer: Customer, transaction: Transaction) -> float:
        hlv_before = self.hlv_calculator.get_hlv(customer)

        # index products
        portfolio_dict: Dict[str, List[Product]] = dict()
        for product in customer.portfolio:
            if product.name not in portfolio_dict:
                portfolio_dict[product.name] = list()
            portfolio_dict[product.name].append(product)

        # Remove products
        for product in transaction.removed:
            portfolio_dict[product.name].pop()

        # Add Products
        for product in transaction.added:
            if product.name not in portfolio_dict:
                portfolio_dict[product.name] = list()
            portfolio_dict[product.name].append(product)

        # flatten portfolio to list
        customer.portfolio = list(chain(*portfolio_dict.values()))

        hlv_after = self.hlv_calculator.get_hlv(customer)

        return hlv_after - hlv_before


class HlvCalculator:
    def get_hlv(self, customer: Customer) -> float:
        margin = 0.0
        for product in customer:
            margin += product.margin * 5  # years
        return margin