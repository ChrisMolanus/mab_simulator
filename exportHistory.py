import os
from typing import List

import pandas as pd

from policy import get_channel_action_cost, Action, Customer, HistoricalActionPropensity


def export_history_to_parquet(historical_action_propensities: List[HistoricalActionPropensity],
                              all_customers: List[Customer],
                              all_actions: List[Action],
                              output_dir: str = "output"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created {os.path.abspath(output_dir)} directory")
    else:
        print(f"Exporting to {os.path.abspath(output_dir)} directory")

    # Export customer
    customer_list = list()
    for customer in all_customers:
        customer_list.append(
            {
                "id": customer.id,
                "name": customer.name,
                "dob": customer.dob,
                "billing_postcode": customer.billing_address.postcode,
                "billing_house_number": customer.billing_address.house_number
            }
        )
    customers = pd.DataFrame(customer_list)
    customers.to_parquet(os.path.join(output_dir, "customers.parquet"), index=False)
    print("Exported customers.parquet")
    del customers

    # Export customer portfolios
    customer_portfolio_list = list()
    for customer in all_customers:
        for product in customer.portfolio:
            customer_portfolio_list.append(
                {
                    "customer_id": customer.id,
                    "product_id": product.name
                }
            )
    customer_portfolios = pd.DataFrame(customer_portfolio_list)
    customer_portfolios.to_parquet(os.path.join(output_dir, "customer_portfolios.parquet"), index=False)
    print("Exported customer_portfolios.parquet")
    del customer_portfolios

    # Export actions
    action_list = list()
    action_product_list = list()
    for action in all_actions:
        action_list.append({
            "name": action.name,
            "start_date": action.start_date,
            "end_date": action.end_date,
            "cool_off_days": action.cool_off_days,
            "channel": str(action.channel),
            "offer_name": action.offer.name,
            "max_margin": action.get_max_margin(years_horizon=5),
            "cost": get_channel_action_cost(action.channel)
        })
        for product in action.offer.products:
            action_product_list.append({
                "action_name": action.name,
                "product_name": product.name
            })
    actions = pd.DataFrame(action_list)
    actions.to_parquet(os.path.join(output_dir, "actions.parquet"), index=False)
    print("Exported actions.parquet")
    del actions

    # Export Product / Action link
    action_product = pd.DataFrame(action_product_list)
    action_product.to_parquet(os.path.join(output_dir, "action_product.parquet"), index=False)
    print("Exported action_product.parquet")
    del action_product

    # Export Transactions
    transaction_list = list()
    for t in historical_action_propensities:
        transaction_list.append({
            "id": t.customer.id,
            "action_ts": t.action_ts,
            "action_name": t.chosen_action.name,
            "reward_ts": t.reward_ts,
            "reward": t.reward,
        })
    transactions = pd.DataFrame(transaction_list)
    transactions.to_parquet(os.path.join(output_dir, "transactions.parquet"), index=False)
    print("Exported transactions.parquet")
    del transactions
