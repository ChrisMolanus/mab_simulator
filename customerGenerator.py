from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
import csv

from policy import ProductType, Product, Customer, Address, Action, CustomerAction, Transaction, Channel


def get_products() -> Tuple[List[Product], List[float]]:
    """
    Reads the data/products.csv file and return Product objects and the current market size per product
    :return: All products(well for now they are Offers) and teh proportion of the base that has the product
    """
    products: List[Product] = list()
    product_market_size: List[float] = list()
    with open('data/products.csv', mode='r') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        for row in reader:
            products.append(Product(id=row["id"], name=row["name"], list_price=float(row["yearly_list_price"]),
                                    margin=float(row["yearly_margin"]), product_type=ProductType.FIXED_INTERNET,
                                    start_date=datetime.strptime(row["start_date"], '%Y-%m-%d').date(),
                                    end_date=datetime.strptime(row["end_date"], '%Y-%m-%d').date(),
                                    download_speed=float(row["download_speed"]),
                                    upload_speed=float(row["upload_speed"])))
            product_market_size.append(float(row["segment_size"]))
    return products, product_market_size


def generate_portfolios(nr_of_customers) -> List[List[Product]]:
    """
    Generates fake portfolios
    :param nr_of_customers: The number of portfolios to generate
    :return: List[List[Product]]
    """
    products, product_market_size = get_products()

    return [[p] for p in np.random.choice(products, nr_of_customers, p=product_market_size)]


def generate_customers(nr_of_customers) -> List[Customer]:
    """
    Generates fake customers
    :param nr_of_customers: The number of customers to generate
    :return: List[Customer]
    """
    portfolios = generate_portfolios(nr_of_customers)
    names = generate_names(nr_of_customers)
    customers: List[Customer] = list()
    for i in range(nr_of_customers):
        fake_address = Address(postcode=f"123{i}AB", house_number=i, ext=None)
        customers.append(Customer(id=i,
                                  name=f"{names[i]['lastname']}, {names[i]['firstname']}",
                                  dob=(datetime.today() - timedelta(days=random.randint(6570, 15000))).date(),
                                  billing_address=fake_address,
                                  portfolio=portfolios[i], ))
    return customers


def generate_names(nr_of_customers) -> List[Dict[str, str]]:
    """
    Generates fake names based on common names used in the Netherlands
    :param nr_of_customers: The number of names teo generate
    :return: List[{"firstname": str, "lastname": str}]
    """
    last_names: List[str] = list()
    n2007: List[int] = list()
    # Nederlandse Familienamen Top 10.000 http://www.naamkunde.net/?page_id=294
    tree = ET.parse('data/fn_10kw.xml')
    root = tree.getroot()

    for record in root.iter('record'):
        prefix = record.find('prefix').text
        naam = record.find('naam').text
        if prefix is not None:
            naam = " ".join([prefix, naam])
        last_names.append(naam)

        n2007.append(int(record.find('n2007').text))
    proportions = np.array(n2007)
    p = proportions/ proportions.sum(axis=0, keepdims=1)
    lastnames = list(np.random.choice(last_names, nr_of_customers, p=p))

    first_names: List[str] = list()
    t8306: List[int] = list()
    # Nederlandse Voornamen Top 10.000 http://www.naamkunde.net/?page_id=293
    tree = ET.parse('data/voornamentop10000.xml')
    root = tree.getroot()

    for record in root.iter('record'):
        if record.find('voornaam') is not None:
            first_names.append(record.find('voornaam').text)

            t8306.append(int(record.find('t8306').text))

    proportions = np.array(t8306)
    p = proportions / proportions.sum(axis=0, keepdims=1)
    firstnames = list(np.random.choice(first_names, nr_of_customers, p=p))

    names: List[Dict[str, str]] = list()
    for i in range(len(lastnames)):
        names.append({"firstname": firstnames[i], "lastname": lastnames[i]})
    return names


def what_would_a_customer_do(customer: Customer, action: Action, ts: datetime) -> CustomerAction:
    """
    An agent that assumes a rational customer and simulates the decision process of a customer to buy a new product
    :param customer: The customer
    :param action: The Action performed on the customer
    :param ts: The timestamp it was performed
    :return: None: Customer did nothing or rejected offer, A Transaction: Customer accepted offer
    """
    for product in customer.portfolio:
        if product.product_type == ProductType.FIXED_INTERNET:
            current_internet = product
            break
    for product in action.offer.products:
        if product.product_type == ProductType.FIXED_INTERNET:
            offer_internet = product
            break

    # Realistic conventions rates for these channels [Convert, Reject/ignore]
    conversion_per_channel = {Channel.OUTBOUND_EMAIL: [0.001, 0.999], Channel.OUTBOUND_CALL: [0.03, 0.97]}

    # Only buy if offer is better that what we have and it costs less than 10% more and random chance is in your favour
    if current_internet.kwargs["download_speed"] < offer_internet.kwargs["download_speed"]\
            and offer_internet.list_price/current_internet.list_price < 1.1\
            and np.random.choice([True, False], 1, True, conversion_per_channel[action.channel]):
        return Transaction(customer=customer,
                           channel=action.channel,
                           added=action.offer.products,
                           removed=[current_internet],
                           ts=ts)
    return None
