import random
from datetime import datetime
from datetime import date
from enum import Enum
from typing import List, Dict, Set, Optional


class ProductType(Enum):
    FIXED_INTERNET = "Fixed Internet Service"
    PSTN = "Plain old Telephone"
    MOBILE = "Mobile service"
    DISCOUNT = "Discount"


class Product:
    def __init__(self, id: int, name: str, list_price: float, margin: float, product_type: ProductType,
                 start_date: date, end_date: date, **kwargs):
        """
        Something that was sold/given to a customer, E.g. Hardware, Service contract, discount
        :param id: The product ID as it is knows in the accounting system
        :param name: The Name as it would appear on a customer bill
        :param list_price: The price as it would appear on a customer bill
        :param margin: The margin that is directly made on the sale of this product
        :param product_type: The type of product
        :param start_date: The date when this prod could have been sold for the first time
        :param end_date: The date when this product could npo longer be sold
        """
        self.id = id
        self.name = name
        self.list_price = list_price
        self._margin = margin
        self.product_type = product_type
        self.start_date = start_date
        self.end_date = end_date
        self.kwargs = kwargs

    def get_margin(self, base_product=None) -> float:
        """
        Calculate to annual margin of this product
        :param base_product: The product of wich this product is a modifierof, None if this is a L0 product
        :return: The Euro annual margin
        """
        return self._margin

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name


class Discount(Product):
    def __init__(self, id: int, name: str, list_price: float, margin: float, product_type: ProductType,
                 start_date: date, end_date: date, **kwargs):
        """
        A customer discount represented as a product that can appear on a bill
        :param id: The id
        :param name: The name it would appear on the bill
        :param list_price: The negative annual value in euros
        :param margin: The negative annual value in euros since this is a loss
        :param product_type: The typeof products, default DISCOUNT
        :param start_date: The date when this prod could have been cold for the first time
        :param end_date: The date when this product could npo longer be sold
        """
        super(Discount, self).__init__(id, name, list_price, margin, product_type, start_date, end_date, **kwargs)

    def get_margin(self, base_product: Product) -> float:
        """
        Calculate to annual margin of this product
        :param base_product: The product of wich this product is a modifierof, None if this is a L0 product
        :return: The Euro annual margin
        """
        return base_product.get_margin() + self._margin

    def __str__(self):
        return self.name


class Address:
    def __init__(self, postcode: str, house_number: int, ext: str):
        """
        A basic address
        :param postcode:
        :param house_number:
        :param ext:
        """
        self.postcode = postcode
        self.house_number = house_number
        self.ext = ext

    def __str__(self):
        return f"{self.postcode} {self.house_number} {self.ext}"


class Customer:
    def __init__(self, id: int, name: str, dob: date, billing_address: Address, portfolio: List[Product]):
        """
        A contract holder or a potential contract holder
        :param id: The internal ID of this customer
        :param name: The name of this customer
        :param dob: The Date of Birth of this customer
        :param billing_address: The billing address of this customer
        :param portfolio: The current portfolio of this customer, this can contain more than one of the same product
        """
        self.id = id
        self.name = name
        self.dob = dob
        self.billing_address = billing_address
        self.portfolio = portfolio


class Channel(Enum):
    OUTBOUND_CALL = "Outbound call"
    OUTBOUND_EMAIL = "Send Email"


class Offer:
    def __init__(self, name: str, products: List[Product]):
        """
        An offer that can be presented to the customer of lead
        :param name: The commercial name of the offer
        :param products: The collection of products that make up this offer
        """
        self.name = name
        self.products = products


class Template:
    def __init__(self, name: str, channel: Channel, icon: str):
        """
        A template that can be rendered in a given channel
        :param name: The name of the template as it is known in the channel's operational system
        :param channel: The channel
        :param icon: The icon for displaying it in a interface (not used)
        """
        self.name = name
        self.channel = channel
        self.icon = icon


class Content:
    def __init__(self, name: str, channel: Channel, template: Template, **kwargs):
        """
        Content that cna be rendered with a template in a channel to be shown to a customer
        :param name: The name of the content bundle
        :param channel: The channel where this content is intended to be used
        :param template: The template that this content can be rendered with
        :param kwargs: The parameters to be passed tot eh template
        """
        self.name = name
        self.channel = channel
        self.template = template
        self.kwargs = kwargs

    def print_args(self):
        for key, value in self.kwargs.items():
            print(f"{key} {value}")


class Action:
    def __init__(self, name: str, channel: Channel, offer: Offer, content: Content,
                 start_date: date, end_date: date, cool_off_days: int):
        """
        Action as defined by Marketing
        :param name: Name of Action as it is know in Marketing
        :param channel: The Marketing channel
        :param offer: The group of products (and discounts) that are assosiated with this offer, can be None
        :param content: The marketing content use during serving of this Action
        :param start_date: The day that this action is allowed to be assigned to a customer
        :param end_date: The last day this action can be assigned to a customer
        :param cool_off_days: The number of days to allow teh action to take effect
        """
        self.name = name
        self.channel = channel
        self.offer = offer
        self.content = content
        self.start_date = start_date
        self.end_date = end_date
        self.cool_off_days = cool_off_days


class ServedActionPropensity:
    def __init__(self, customer: Customer, chosen_action: Action, action_propensities: Dict[str, float]):
        """
        The NBA of the Policy and the propensity of the other actions for this customer
        :param customer: The customer
        :param chosen_action: The Next Best Action
        :param action_propensities: The propensity of the other actions
        """
        self.customer = customer
        self.chosen_action = chosen_action
        self.action_propensities = action_propensities


class CustomerAction:
    def __init__(self, customer: Customer, channel: Channel, ts: datetime, **kwargs):
        """
        An general action take by a customer, a service call
        :param customer: The Customer that performed the action
        :param channel: The Channel where the action was performed
        :param ts: The Time Stamp of when teh customer performed the action
        :param kwargs:
        """
        self.customer = customer
        self.channel = channel
        self.ts = ts
        self.kwargs = kwargs


class Transaction(CustomerAction):
    def __init__(self, customer: Customer, channel: Channel, added: List[Product], removed: List[Product],
                 ts: datetime):
        """
        An customer action that represents a customer requesting to change their portfolio
        :param customer: The Customer that performed the action
        :param channel: The Channel where the action was performed
        :param added: The products that where added to the customers portfolio
        :param removed: The products that were removed from teh portfolio
        :param ts: The Time Stamp of when teh customer performed the action
        """
        super(Transaction, self).__init__(customer, channel, ts)
        self.added = added
        self.removed = removed


class Policy:
    def __init__(self, **kwargs):
        """
        A Marketing policy that provides Next Best Actions for customers
        based on the customer context and historical action rewards
        :param kwargs:
        """
        self.applicable_actions: Dict[str, List[Action]] = dict()

    def add_arm(self, action: Action, segment_ids: List[str]):
        """
        Add a new action that can be assigned to a customer as a NBA
        :param action: The new action
        :param segment_ids: The list of customer segments where this action can be used
        """
        for segment_id in segment_ids:
            if segment_id not in self.applicable_actions:
                self.applicable_actions[segment_id] = list()
            self.applicable_actions[segment_id].append(action)

    def add_customer_action(self, served_action_propensity: ServedActionPropensity, customer_action: CustomerAction,
                            reward: float):
        """
        Updates the policy to inform it that a customer has taken an action
        :param served_action_propensity: The served_action_propensity(NBA) that we are assuming lead to the customer action
        :param customer_action: The action the customer toke
        :param reward: The monitory value of the customer taking the action
        """
        pass

    def add_company_action(self, customer: Customer, action: Action, ts: datetime, cost: float):
        """
        Updates the policy to inform it that the company has taken an action(likely the NBA)
        :param customer: The customer that received the action
        :param action: The action performed
        :param ts: The time stamp of when the action was performed
        :param cost: The cost of performing the actions e.g. cost of postage for a newsletter
        """
        pass

    def add_channel_quota(self, channel: Channel, daily_quota: int):
        """
        Add a daily limit of how many actions can be performed per day in a given channel
        For example the maximum amount of calls that can be performed by the outbound call center in a day
        :param channel: The channel where the quota applies
        :param daily_quota: The maximum sum amount of times an actions that uses this channel can be returned as NBAs
        """
        pass

    def set_datetime(self, now_ts: datetime):
        """
        Updates the policy to inform it that it is now a new day/hour
        Used for simulations
        :param now_ts: The current time stamp
        """
        for segment_id, actions in self.applicable_actions.items():
            actions_to_remove = list()
            for action in actions:
                if action.end_date <= now_ts.date():
                    actions_to_remove.append(action)
            for action in actions_to_remove:
                actions.remove(action)

    def get_next_best_action(self, customer: Customer, segment_ids: List[str]) -> Optional[ServedActionPropensity]:
        """
        Get the policies recommendation for the customer's Next Best Action
        :param customer: The customer
        :param segment_ids: The segments the customer is in
        :return: The NBA and the propensities of the other actions as a ServedActionPropensity object
        """
        actions: Set[Action] = set()
        for segment_id in segment_ids:
            if segment_id in self.applicable_actions:
                actions = actions.union(self.applicable_actions[segment_id])
        if len(actions) > 1:
            nba = random.sample(actions, k=1)[0]
            propensities: Dict[str, float] = dict()
            for action in actions:
                propensities[action.name] = 1 / len(actions)
            return ServedActionPropensity(customer=customer, chosen_action=nba, action_propensities=propensities)
        else:
            return None
