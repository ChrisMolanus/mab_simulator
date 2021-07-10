from datetime import datetime
from enum import Enum
from typing import List, Dict


class Product:
    def __init__(self, id: int, name: str, list_price: float, margin: float):
        """
        Something that was sold/given to a customer, E.g. Hardware, Service contract, discount
        :param id: The product ID as it is knows in the accounting system
        :param name: The Name as it would appear on a customer bill
        :param list_price: The price as it would appear on a customer bill
        :param margin: The margin that is directly made on the sale of this product
        """
        self.id = id
        self.name = name
        self.list_price = list_price
        self._margin = margin

    def get_margin(self, base_product) -> float:
        """
        Calculate to annual margin of this product
        :param base_product: The product of wich this product is a modifierof, None if this is a L0 product
        :return: The Euro annual margin
        """
        return self._margin

    def __eq__(self, other):
        return self.name == other.name


class Discount(Product):
    def __init__(self, id: int, name: str, list_price: float, margin: float):
        """
        A customer discount represented as a product that can appear on a bill
        :param id: The id
        :param name: The name it would appear on the bill
        :param list_price: The negative annual value in euros
        :param margin: The negative annual value in euros since this is a loss
        """
        self.id = id
        self.name = name
        self.list_price = list_price
        self._margin = margin

    def get_margin(self, base_product: Product) -> float:
        """
        Calculate to annual margin of this product
        :param base_product: The product of wich this product is a modifierof, None if this is a L0 product
        :return: The Euro annual margin
        """
        return base_product.get_margin() + self._margin

class Address:
    def __init__(self, postcode: str, house_number: int, ext: str):
        self.postcode = postcode
        self.house_number = house_number
        self.ext = ext


class Customer:
    def __init__(self, name: str, dob: datetime, billing_address: Address, portfolio: List[Product]):
        self.name = name
        self.dob = dob
        self.billing_address = billing_address
        # Portfolio can contain duplicate products since multiple people in a house hold can have the same product
        self.portfolio = portfolio


class Channel(Enum):
    OUTBOUND_CALL = "Outbound call"
    OUTBOUND_EMAIL = "Send Email"


class Offer:
    def __init__(self, name: str, products: List[Product]):
        self.name = name
        self.products = products


class Template:
    def __init__(self, name: str, channel: Channel, icon: str):
        self.name = name
        self.channel = channel
        self.icon = icon


class Content:
    def __init__(self, name:str, channel: Channel, template: Template, **kwargs):
        self.name = name
        self.channel = channel
        self.template = template
        self.kwargs = kwargs

    def print_args(self):
        for key, value in self.kwargs.items():
            print(f"{key} {value}")


class Action:
    def __init__(self, name: str, channel: Channel, offer: Offer, content: Content,
                 start_date: datetime, end_date: datetime):
        """
        Action as defined by Marketing
        :param name: Name of Action as it is know in Marketing
        :param channel: The Marketing channel
        :param offer: The group of products (and discounts) that are assosiated with this offer, can be None
        :param content: The marketing content use during serving of this Action
        :param start_date: The day that this action is allowed to be assigned to a customer
        :param end_date: The last day this action can be assigned to a customer
        """
        self.name = name
        self.channel = channel
        self.offer = offer
        self.content = content
        self.start_date = start_date
        self.end_date = end_date


class ServedActionPropensity:
    def __init__(self, chosen_action: Action, action_propensities: Dict[str, float]):
        """
        The NBA of the Policy and the propensity of the other actions for this customer
        :param chosen_action: The Next Best Action
        :param action_propensities: The propensity of the other actions
        """
        self.chosen_action = chosen_action
        self.action_propensities = action_propensities


# class ServedAction:
#     def __init__(self, customer: Customer, action: Action):
#         """
#         The Event performed in the channel
#         :param customer: The customer it wa performed on
#         :param action: The action that was performed
#         """
#         self.customer = customer
#         self.action = action

class Transaction:
    def __init__(self, added: List[Product], removed: List[Product]):
        self.added = added
        self.removed = removed


class Policy:
    def __init__(self):
        self.applicable_actions = dict()

    def add_arm(self, action: Action, segment_ids: List[str]):
        for segment_id in segment_ids:
            self.applicable_actions[segment_id] = action

