from datetime import timedelta, date
from typing import Dict, List
from itertools import chain

from policy import Customer, Transaction, Product


class RewardCalculator:
    def __init__(self):
        self.hlv_calculator = HlvCalculator()

    def calculate(self, customer: Customer, transaction: Transaction) -> float:
        hlv_before = self.hlv_calculator.get_hlv(customer, transaction.ts.date())

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
        fake_customer = Customer(id=customer.id,
                                 name=customer.name,
                                 dob=customer.dob,
                                 billing_address=customer.billing_address,
                                 portfolio=list(chain(*portfolio_dict.values()))
                                 )

        hlv_after = self.hlv_calculator.get_hlv(fake_customer, transaction.ts.date())

        return hlv_after - hlv_before


class HlvCalculator:

    def get_hlv(self, customer: Customer, date_of_transaction: date, hlv_horizon_years: int = 5) -> float:
        margin = 0.0
        for product in customer.portfolio:
            lifetime = max(0.5, (product.contract_start +
                               timedelta(weeks=52*hlv_horizon_years)).year - date_of_transaction.year)
            margin += product.get_margin() * lifetime
        return margin
