import csv
from datetime import datetime, timedelta
from typing import List

from customerGenerator import get_products
from policy import Product, Action, Channel, Offer, Content, Template, ProductType


class EmailTemplate1(Template):
    def render(self, content: Content):
        return f"""Subject line: {content.kwargs["subjectline"]}
        Body: Buy this awesome {content.kwargs["product"]}
        """


class EmailTemplate2(Template):
    def render(self, content: Content):
        return f"""Subject line: {content.kwargs["subjectline"]}
        Body: For a limited time you can buy a {content.kwargs["product"]}
        """


class AdviceTemplate1(Template):
    def render(self, content: Content):
        return f"""Offer customer {content.kwargs["product"]}"""


templates = [
    EmailTemplate1(name="Email Template 1", channel=Channel.OUTBOUND_EMAIL, icon=""),
    EmailTemplate2(name="Email Template 2", channel=Channel.OUTBOUND_EMAIL, icon=""),
    AdviceTemplate1(name="Advice Template 1", channel=Channel.OUTBOUND_CALL, icon=""),
]


def get_actions() -> List[Action]:
    products, product_market_size = get_products()

    actions: List[Action] = list()
    for channel in Channel:
        for product in products:
            for template in templates:
                offer = Offer(name=f"Offer {product.name}", products=[product])
                content = Content(name=f"Content for Offer {product.name} in channel {channel}", channel=channel, template=template)
                actions.append(Action(name=f"Sell {product.name} in {channel} using {template}",
                                      channel=channel,
                                      offer=offer,
                                      content=content,
                                      start_date=product.start_date,
                                      end_date=product.end_date,
                                      cool_off_days=21
                               ))
    return actions