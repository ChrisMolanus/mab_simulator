import csv
from datetime import datetime, timedelta
from typing import List, Dict

from customerGenerator import get_products
from policy import Product, Action, Channel, Offer, Content, Template, ProductType


class EmailTemplate1(Template):
    def __init__(self, name: str, channel: Channel, icon: str):
        super().__init__(name, channel, icon)

    def render(self, subjectline: str, product: str,  **kwargs) -> str:
        return f"""Subject line: {subjectline}
        Body: Buy this awesome {product}
        """

    def __str__(self):
        return self.render(subjectline="{Subject Line}", product="{Product}")


class EmailTemplate2(Template):
    def __init__(self, name: str, channel: Channel, icon: str):
        super().__init__(name, channel, icon)

    def render(self, subjectline: str, product: str,  **kwargs) -> str:
        return f"""Subject line: {subjectline}
        Body: For a limited time you can buy a {product}
        """

    def __str__(self):
        return self.render(subjectline="{Subject Line}", product="{Product}")


class AdviceTemplate1(Template):
    def __init__(self, name: str, channel: Channel, icon: str):
        super().__init__(name, channel, icon)

    def render(self, product: str, **kwargs):
        return f"""Offer customer {product}"""

    def __str__(self):
        return self.render(product="{product}")


class AdviceTemplate2(Template):
    def __init__(self, name: str, channel: Channel, icon: str):
        super().__init__(name, channel, icon)

    def render(self, product: str, **kwargs):
        return f"""Because they have been our customer for more that two years offer them {product}"""

    def __str__(self):
        return self.render(product="{product}")


templates: Dict[Channel, List[Template]] = {
    Channel.OUTBOUND_EMAIL: [EmailTemplate1(name="Email Template 1", channel=Channel.OUTBOUND_EMAIL, icon=""),
                             EmailTemplate2(name="Email Template 2", channel=Channel.OUTBOUND_EMAIL, icon="")],
    Channel.OUTBOUND_CALL: [AdviceTemplate1(name="Advice Template 1", channel=Channel.OUTBOUND_CALL, icon=""),
                            AdviceTemplate2(name="Advice Template 2", channel=Channel.OUTBOUND_CALL, icon="")],
}


def get_actions() -> List[Action]:
    products, product_market_size = get_products()

    actions: List[Action] = list()
    for channel in templates:
        for product in products:
            for template in templates[channel]:
                offer = Offer(name=f"Offer {product.name}", products=[product])
                if channel == Channel.OUTBOUND_EMAIL:
                    content = Content(name=f"Content for Offer {product.name} in channel {channel}", channel=channel,
                                      template=template, product=product.name, subjectline="New offer")
                else:
                    content = Content(name=f"Content for Offer {product.name} in channel {channel}", channel=channel,
                                      template=template, product=product.name)
                actions.append(Action(name=f"Sell {product.name} in {channel} using template :{template}",
                                      channel=channel,
                                      offer=offer,
                                      content=content,
                                      start_date=product.start_date,
                                      end_date=product.end_date,
                                      cool_off_days=21)
                               )
    return actions
