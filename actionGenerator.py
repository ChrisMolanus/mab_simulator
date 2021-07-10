import csv
from datetime import datetime, timedelta
from typing import List

from policy import Product, Action, Channel, Offer, Content, Template


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
    products: List[Product] = list()
    with open('data/products.csv', mode='r') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        for row in reader:
            products.append(Product(id=row["id"], name=row["name"], list_price=float(row["yearly_list_price"]),
                                    margin=float(row["yearly_margin"])))
    actions: List[Action] = list()
    for channel in Channel:
        for product in products:
            for template in templates:
                offer = Offer(name=f"Offer {product.name}", products=[product])
                content = Content(name=f"Content for Offer {product.name} in channel {channel}", channel=channel, template=template)
                actions.append(Action(name=f"Sell {product.name} in {channel}", channel=channel, offer=offer, content=content, start_date=datetime.today(), end_date=datetime.today()+timedelta(weeks=6)))