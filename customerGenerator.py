from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import csv

from policy import ProductType, Product, Customer, Address, Action, CustomerAction, Transaction


def generate_portfolios(nr_of_customers) -> List[List[Product]]:
    products: List[Product] = list()
    product_market_size: List[float] = list()
    with open('data/products.csv', mode='r') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        for row in reader:
            products.append(Product(id=row["id"], name=row["name"], list_price=float(row["yearly_list_price"]),
                                    margin=float(row["yearly_margin"]), product_type=ProductType.FIXED_INTERNET, download_speed=float(row["download_speed"]), upload_speed=float(row["upload_speed"])))
            product_market_size.append(float(row["segment_size"]))

    return [[p] for p in np.random.choice(products, nr_of_customers, p=product_market_size)]


def generate_customers(nr_of_customers) -> List[Customer]:
    portfolios = generate_portfolios(nr_of_customers)
    names = generate_names(nr_of_customers)
    customers: List[Customer] = list()
    for i in range(nr_of_customers):
        fake_address = Address(postcode=f"123{i}AB", house_number=i, ext=None)
        customers.append(Customer(id=i,
                                  name=f"{names[i]['lastname']}, {names[i]['firstname']}",
                                  dob=datetime.today(),
                                  billing_address=fake_address,
                                  portfolio=portfolios[i], ))
    return customers


def generate_names(nr_of_customers) -> List[Dict[str, str]]:
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
    for product in customer.portfolio:
        if product.product_type == ProductType.FIXED_INTERNET:
            current_internet = product
            break
    for product in action.offer.products:
        if product.product_type == ProductType.FIXED_INTERNET:
            offer_internet = product
            break

    if current_internet.kwargs["download_speed"] < offer_internet.kwargs["download_speed"]:
        return Transaction(customer=customer, channel=action.channel, added=action.offer.products, removed=[current_internet], ts=ts)
    return None


if __name__ == "__main__":
    nr_of_customers = 10000

    #data/fn_10kw.xml

    # Nederlandse Voornamen Top 10.000 http://www.naamkunde.net/?page_id=293
    #data/voornamentop10000.xml



