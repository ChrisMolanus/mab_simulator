import os
import sys
from datetime import datetime, timedelta, date
from multiprocessing import Process, Queue
from random import seed
from typing import Dict, List, Any, Tuple, Type, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame

import bayesianGroundhog
import epsilonRingtail
import randomCrayfish
import segmentJunglefowl
from actionGenerator import get_actions
from customerGenerator import generate_customers, what_would_a_customer_do, get_products
from exportHistory import export_history_to_parquet
from policy import ServedActionPropensity, Policy, Customer, Channel, Action, Transaction, HistoricalActionPropensity, \
    get_channel_action_cost, customers_to_dataframe, actions_to_dataframe
from rewardCalculator import RewardCalculator


class TelcoSimulator:
    def __init__(self):
        """
        A simulator that simulates the interaction between a customer base
        and a marketing department of a telecommunications company.
        """
        seed(7837)
        np.random.seed(7837)
        self.call_center_daily_quota = 200

    def get_marketing_history(self, start_ts: datetime = None, day_count: int = 365,
                              nr_of_customers: int = 10000, export: bool = False
                              ) -> Tuple[List[HistoricalActionPropensity], List[Customer], List[Action]]:
        """
        Generates transactions of generated customers using the JungleFowl policy.
        This is meant to represent the policy of a human marketing department that ran the marketing actions in the past
        :param start_ts: The date to start simulating a transaction history, default 1st Jan 2011
        :param day_count: The number of days the generate a history for
        :param nr_of_customers: The number of customers to simulate as starting base
        :param export: True if you would like all the generated data to be exported to the output dir
        :return: Historical transactions, the customers and the list of actions that could have been performed
        """
        if start_ts is None:
            start_ts = datetime(2011, 1, 1)
        sim_start_date = start_ts.date()

        reward_calculator = RewardCalculator()
        products, product_market_size = get_products(sim_start_date - timedelta(days=2190), sim_start_date)
        product_market_sizes = [0.0] * len(product_market_size)

        # Set the base product distribution to how it was on start_date
        product_market_sizes[0] = 0.19
        product_market_sizes[1] = 0.51
        product_market_sizes[2] = 0.3
        all_customers = generate_customers(nr_of_customers, sim_start_date, product_market_sizes)
        keywords = {"current_base": all_customers}

        actions = get_actions()

        log, chosen_action_log, historical_action_propensities = self._sim_cycle_run(actions, all_customers,
                                                                                     day_count,
                                                                                     segmentJunglefowl.SegmentJunglefowl
                                                                                     ,reward_calculator, [], start_ts,
                                                                                     **keywords)
        if export:
            export_history_to_parquet(historical_action_propensities, all_customers, actions)

        return historical_action_propensities, all_customers, actions

    def _policy_sim(self, policy_class, all_customers: List[Customer], actions: List[Action], day_count: int,
                    output: Queue,
                    run_id: int, sequential_runs: int, history: List[HistoricalActionPropensity],
                    start_ts: datetime = datetime.today(), dump_log_to_csv: bool = False, **kwargs) -> bool:
        """
        Run a number of cycles of a policy
        :param policy_class: The policy to run
        :param all_customers: The customers to use during the run
        :param actions: The actions that are allowed to be executed on the customers during the run
        :param day_count: The length of the simulation in days
        :param output: The Queue to output the results to
        :param run_id: The ID of this run
        :param sequential_runs: The number of sequential cycles to run,
        used to eliminate waiting time ot start a new thread
        :param history: The list of HistoricalActionPropensity to pass to the policy on startup
        :param start_ts: The timestamp to start the simulation at
        :param dump_log_to_csv: True is log of first sim should be dumped to a CVS file
        :param kwargs: kargs to pass to the policy
        :return:
        """
        print(policy_class.__name__ + str(run_id))
        seed(7831 * run_id)
        np.random.seed(7831 * run_id)
        reward_calculator = RewardCalculator()

        logs: List[List[Dict[str, Any]]] = list()

        # The chosen_action_log of the last sim should create a sample offer distribution plot
        chosen_action_log: Dict[datetime, Dict[str, int]] = dict()

        # We can run multiple sequential runs since there is a time overhead to create new processes
        for s_run in range(sequential_runs):
            # Just overwrite the chosen_action_log and take the last one
            log, chosen_action_log, action_dump = self._sim_cycle_run(actions, all_customers, day_count,
                                                            policy_class, reward_calculator, history, start_ts,
                                                            **kwargs)
            logs.append(log)

        # Only the last chosen_action_log
        if run_id == 0:
            output.put({"policy": policy_class.__name__, "logs": logs, "chosen_action_log": chosen_action_log})
            if dump_log_to_csv:
                logfileList: List[dict] = list()
                for haps in action_dump:
                    logfileList.append({
                        "action_ts": haps.action_ts,
                        "customer_id": haps.customer.id,
                        "chosen_action.name": haps.chosen_action.name,
                        "reward_ts": haps.reward_ts,
                        "reward": haps.reward
                    })
                df = pd.DataFrame(logfileList)
                df.to_csv("output/nba_log.csv", index=False)
        else:
            output.put({"policy": policy_class.__name__, "logs": logs})
        print(f"{policy_class.__name__} end")
        return True

    def _sim_cycle_run(self, all_actions: List[Action], all_customers: List[Customer], day_count: int,
                       policy_class: Type,
                       reward_calculator: RewardCalculator, history: List[HistoricalActionPropensity],
                       start_ts: datetime = datetime.today(), **kwargs
                       ) -> Tuple[
        List[Dict[str, Any]], Dict[date, Dict[str, int]], List[HistoricalActionPropensity]]:
        """
        Run a single simulation cycle
        :param all_actions: The actions that can be executed on the customers
        :param all_customers: The customers
        :param day_count: The length of the simulation in days
        :param policy_class: The policy to use to execute the actions
        :param reward_calculator: The reward calculator to calculate eh reward of a customer action
        :param history: Historical transactions
        :param start_ts: The timestamp to start the simulation at
        :param kwargs: The kargs to pass to the policy
        :return: log, chosen_action_log, historicalActionPropensities
        """
        chosen_action_log: Dict[date, Dict[str, int]] = dict()
        log: List[Dict[str, Any]] = list()
        customers: List[Customer] = all_customers.copy()
        policy: Policy = policy_class(history, **kwargs)
        policy.set_datetime(start_ts)
        today = start_ts.date()
        actions: List[Action] = list()
        historical_action_propensities: List[HistoricalActionPropensity] = history
        # Since we are simulating implementing this in a company already has customers
        # and has already ran campaigns in the past we initialize teh policy with the actions that are already active
        for action in all_actions:
            # So not today == action.start_date, that is later
            if action.start_date <= today < action.end_date:
                policy.add_arm(action, ["basic"])
            elif action.start_date >= today:
                actions.append(action)
        served_action_propensities: Dict[int, ServedActionPropensity] = dict()
        action_timeout: Dict[date, dict[int, ServedActionPropensity]] = dict()
        # Run a simulation of day_count number of days and track the cumulative_reward
        cumulative_reward = 0.0
        for today_datetime in (start_ts + timedelta(n) for n in range(day_count)):
            today = today_datetime.date()
            if today not in action_timeout:
                action_timeout[today] = dict()

            # Only one sim should create a sample offer distribution plot
            chosen_action_log[today] = dict()
            chosen_action_log[today]["No Action"] = 0
            for action in all_actions:
                chosen_action_log[today][action.name] = 0

            # Since we are running a simulation we need to tell the policy it is the next day
            policy.set_datetime(today_datetime)

            # Add new actions that are available from today
            actions_to_remove = list()
            for action in actions:
                if today == action.start_date:
                    policy.add_arm(action, ["basic"])
                    actions_to_remove.append(action)
            for action in actions_to_remove:
                actions.remove(action)

            # Select a set of people that we can call today
            todays_served_action_propensities = list()
            todays_call_served_action_propensities = list()
            todays_email_served_action_propensities = list()
            # TODO: Since emails don't effect the call_center_daily_quota we should do this in a different way
            while len(todays_served_action_propensities) < self.call_center_daily_quota and len(customers) > 0:
                customer = customers.pop()
                # Check if we ave already called them (Data quality)
                if customer.id not in served_action_propensities:
                    served_action_propensity = policy.get_next_best_action(customer=customer, segment_ids=["basic"])
                    # served_action_propensity can be None if there is no possible action due to constrains
                    if served_action_propensity is not None:
                        todays_served_action_propensities.append(served_action_propensity)
                        # Find the right channel queue, needed since these are usually separate systems
                        # But this is only here for illustration purposes, it has no effect in teh simulator
                        if served_action_propensity.chosen_action.channel == Channel.OUTBOUND_CALL:
                            todays_call_served_action_propensities.append(served_action_propensity)
                        elif served_action_propensity.chosen_action.channel == Channel.OUTBOUND_EMAIL:
                            todays_email_served_action_propensities.append(served_action_propensity)

                        served_action_propensities[customer.id] = served_action_propensity

                        chosen_action_log[today][served_action_propensity.chosen_action.name] += 1
                    else:
                        chosen_action_log[today]["No Action"] += 1

            # Look for old actions that have timed out
            if today in action_timeout:
                for customer_id, served_action_propensity in action_timeout[today].items():
                    # Gave up on this customer since they did not convert
                    reward = get_channel_action_cost(served_action_propensity.chosen_action.channel) * -1
                    policy.add_customer_action(served_action_propensity=served_action_propensity,
                                               customer_action=None,
                                               reward=reward)
                    historical_action_propensities.append(
                        HistoricalActionPropensity(customer=served_action_propensity.customer,
                                                   chosen_action=served_action_propensity.chosen_action,
                                                   action_propensities=served_action_propensity.action_propensities,
                                                   action_ts=served_action_propensity.action_ts,
                                                   reward_ts=today_datetime,
                                                   reward=reward
                                                   )
                    )
                    cumulative_reward += reward
            del action_timeout[today]

            # Fist simulate late reactions of customers that were contacted on other days
            for deadline, customer_served_action_propensities in action_timeout.items():
                # Today's timeouts have already been removed above
                customers_who_responded: List[int] = list()
                for customer_id, served_action_propensity in customer_served_action_propensities.items():
                    days_since_action = (today_datetime - served_action_propensity.action_ts).days
                    customer_action = what_would_a_customer_do(served_action_propensity.customer,
                                                               served_action_propensity.chosen_action,
                                                               today_datetime, days_since_action)

                    if isinstance(customer_action, Transaction):
                        reward = reward_calculator.calculate(served_action_propensity.customer, customer_action)
                        reward -= get_channel_action_cost(served_action_propensity.chosen_action.channel)
                        policy.add_customer_action(served_action_propensity=served_action_propensity,
                                                   customer_action=customer_action,
                                                   reward=reward)
                        historical_action_propensities.append(
                            HistoricalActionPropensity(customer=served_action_propensity.customer,
                                                       chosen_action=served_action_propensity.chosen_action,
                                                       action_propensities=served_action_propensity.action_propensities,
                                                       action_ts=served_action_propensity.action_ts,
                                                       reward_ts=today_datetime,
                                                       reward=reward
                                                       )
                        )

                        cumulative_reward += reward
                        customers_who_responded.append(customer_id)

                # Remove timeouts for customers that responded (late)
                for customer_id in customers_who_responded:
                    del customer_served_action_propensities[customer_id]

            # Simulate today's actions
            call_counter = 0
            for served_action_propensity in todays_served_action_propensities:

                # There is a hard limit of how many calls can be performed by a fixed number of agents
                # So we have to simulate that limit
                if call_counter >= self.call_center_daily_quota:
                    break

                # Get a customer to call or email
                customer = served_action_propensity.customer
                cool_off_days = served_action_propensity.chosen_action.cool_off_days

                # Simulate what the customer would have done
                customer_action = what_would_a_customer_do(served_action_propensity.customer,
                                                           served_action_propensity.chosen_action,
                                                           today_datetime)

                # Accounting now that the call has been made
                call_counter += 1
                served_action_propensity.set_action_timestamp(today_datetime)
                policy.add_company_action(served_action_propensity.customer,
                                          served_action_propensity.chosen_action,
                                          today_datetime,
                                          get_channel_action_cost(served_action_propensity.chosen_action.channel) * -1)

                # See if we have immediate reward, the customer bought the product during the call
                if isinstance(customer_action, Transaction):
                    reward = reward_calculator.calculate(served_action_propensity.customer, customer_action)
                    reward -= get_channel_action_cost(served_action_propensity.chosen_action.channel)
                    policy.add_customer_action(served_action_propensity=served_action_propensity,
                                               customer_action=customer_action,
                                               reward=reward)
                    historical_action_propensities.append(
                        HistoricalActionPropensity(customer=served_action_propensity.customer,
                                                   chosen_action=served_action_propensity.chosen_action,
                                                   action_propensities=served_action_propensity.action_propensities,
                                                   action_ts=served_action_propensity.action_ts,
                                                   reward_ts=today_datetime,
                                                   reward=reward
                                                   )
                    )

                    cumulative_reward += reward
                else:
                    # No immediate reward, but the customer might go on the website later to make the sale
                    deadline = today + timedelta(days=cool_off_days)
                    if deadline not in action_timeout:
                        action_timeout[deadline] = dict()
                    action_timeout[deadline][customer.id] = served_action_propensity

                # TODO: implement the possibility that the customer buys the product at a later date within the cooloff

            log.append({"ts": today, "cumulative_reward": cumulative_reward})
        return log, chosen_action_log, historical_action_propensities

    def plot_timelines(self, chosen_action_logs: Dict[str, Dict[datetime, Dict[str, int]]], actions: List[Action],
                       show=True, save=True
                       ) -> Dict[str, plt.Figure]:
        """
        Create plots for the timelines of the policies
        :param chosen_action_logs: The Chosen Action Logs of the policies where the policy name is the key,
        and the value is the number of times an action was chosen on that date were the action name is the key
        :param actions: A list of all actions so we can know which are never chosen
        :param show: True if the plots should be shown
        :param save: True if the plots should be saved as {policy_name}.png
        :return: plots , a dictionary where the policy name is the key and the matplotlib figure is the value
        """
        # Plot policy timelines
        plots: Dict[str, plt.Figure] = dict()
        for policy_name, chosen_action_log in chosen_action_logs.items():
            labels: List[str] = list()
            x: List[datetime] = list(chosen_action_log.keys())
            x.sort()
            y_per_action: List[List[int]] = list()
            for action in actions:
                y_per_action.append(list())
                labels.append(action.name)
            for day in x:
                chosen_action_counts = chosen_action_log[day]
                for i in range(len(labels)):
                    action_name = labels[i]
                    if action_name in chosen_action_counts:
                        y_per_action[i].append(chosen_action_counts[action_name])
                    else:
                        y_per_action[i].append(0)


            fig, ax = plt.subplots()
            ax.stackplot(x, *y_per_action)  # , labels=labels)
            # plt.legend(loc='upper left')
            # Text in the x axis will be displayed in 'YYYY-mm' format.
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
            # Rotates and right-aligns the x labels so they don't crowd each other.
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')
            ax.set(xlabel='time (days)', ylabel='NBA allocations',
                   title=policy_name)

            plots[policy_name] = fig
            if save:
                fig.savefig(f"{policy_name}.png")
            if show:
                plt.show()
        return plots

    def plot_performance(self, all_logs: Dict[str, Dict[datetime, List[float]]], show=False, save=False) -> plt.Figure:
        """
        Create plot of the aggregated performance of the policies
        :param all_logs: The logs of the runs
        :param show: True
        :param show: True if the plot should be shown
        :param save: True if the plot should be saved as test.png
        :return: fig, the matplotlib figure
        """
        plot_dfs: Dict[str, DataFrame] = dict()
        plot_dict: Dict[str, List[Dict[str, Any]]] = dict()
        last_mean_value: Dict[str, np.ndarray] = dict()
        for policy, log in all_logs.items():
            if policy not in plot_dict:
                plot_dict[policy] = list()
            for ts, sim_values in log.items():
                plot_dict[policy].append({"ts": ts, "mean": np.mean(sim_values), "std": np.std(sim_values)})
                last_mean_value[policy] = np.mean(sim_values)
            plot_dfs[policy] = DataFrame(plot_dict[policy])
        ordered_policies_by_clv = sorted(last_mean_value, key=last_mean_value.get)
        ordered_policies_by_clv.reverse()

        fig, ax = plt.subplots()
        for policy_name in ordered_policies_by_clv:
            policy = plot_dfs[policy_name]
            euro = policy["mean"].iloc[-1]
            policy["mean_k"] = policy["mean"] / 1000
            policy["std_u"] = policy["mean_k"] + (policy["std"] / 1000)
            policy["std_l"] = policy["mean_k"] - (policy["std"] / 1000)

            ax.fill_between(policy["ts"], policy["std_l"], policy["std_u"], alpha=0.2)
            ax.plot(policy["ts"], policy["mean_k"], label=f"{policy_name} €{euro}")

        # Text in the x axis will be displayed in 'YYYY-mm' format.
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        # Rotates and right-aligns the x labels so they don't crowd each other.
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')
        ax.set(xlabel='time (days)', ylabel='Cumulative HLV (1000 Euros)',
               title='Policy performance')

        ax.grid()
        plt.legend()

        if save:
            fig.savefig("test.png")
        if show:
            plt.show()
        return fig

    def export_log_data(self, all_logs: Dict[str, Dict[datetime, List[float]]], output_dir: str = "output"):
        """
        Export logs as YAML files to the "output" dir
        :param all_logs: The logs to export
        :param output_dir: The directory to put the files, will be created if not exist
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for policy_name, log in all_logs.items():
            with open(os.path.join(output_dir, policy_name + ".yaml"), "w") as f:
                f.write(yaml.safe_dump(log))

    def do_simulations(self, policies_to_simulate: List[Type[Union[Any]]], keywords: Dict[str, Any], runs_per_policies: int,
                       sequential_runs: int,
                       customers: List[Customer], actions: List[Action], day_count: int, start_ts: datetime,
                       historical_action_propensities=None
                       ) -> Tuple[Dict[str, Dict[datetime, List[float]]], Dict[str, Dict[datetime, Dict[str, int]]]]:
        """
        Run a set of simulations of the policies in "policies"
        :param policies_to_simulate: A list of policies to simulate
        :param keywords: A dictionary of kargs to pas to teh policies
        :param runs_per_policies: The number of Threads to start per policy
        :param sequential_runs: The number of sequential runs to do in one Thread
        :param customers: A list of customers to be used as the starting base
        :param actions: A list of all actions that can be executed on the customers
        :param day_count: The number of days to simulate
        :param start_ts: The starting date of the simulation
        :param historical_action_propensities: A list of historical transaction the be used to initialize the policies
        :return: all_logs, chosen_action_logs
        """
        if historical_action_propensities is None:
            historical_action_propensities = list()
        processes = list()
        output_queue = Queue()
        for policy_class in policies_to_simulate:
            for r in range(runs_per_policies):
                p = Process(target=self._policy_sim,
                            args=(policy_class, customers, actions, day_count, output_queue, r, sequential_runs,
                                  historical_action_propensities, start_ts, dump_to_csv),
                            kwargs=keywords)
                processes.append(p)

        for p in processes:
            p.start()

        all_logs: Dict[str, Dict[datetime, List[float]]] = dict()
        chosen_action_logs: Dict[str, Dict[datetime, Dict[str, int]]] = dict()
        for policy_class in policies_to_simulate:
            policy_name = policy_class.__name__
            all_logs[policy_name] = dict()

        for p in processes:
            output_logs = output_queue.get(block=True)
            logs = output_logs["logs"]
            policy_name = output_logs["policy"]
            for log in logs:
                for log_line in log:
                    ts = log_line["ts"]
                    cum_reward = log_line["cumulative_reward"]
                    if ts not in all_logs[policy_name]:
                        all_logs[policy_name][ts] = list()
                    all_logs[policy_name][ts].append(cum_reward)
            if "chosen_action_log" in output_logs:
                # This was a run_id 0 sim
                chosen_action_logs[policy_name] = output_logs["chosen_action_log"]

        for p in processes:
            if p.is_alive():
                p.join()

        return all_logs, chosen_action_logs


if __name__ == "__main__":
    dump_to_csv = False
    for i, arg in enumerate(sys.argv):
        if arg == "dump":
            dump_to_csv = True
            break


    policies = [randomCrayfish.RandomCrayfish, segmentJunglefowl.SegmentJunglefowl, bayesianGroundhog.BayesianGroundhog,
                epsilonRingtail.EpsilonRingtail]
    policies = [segmentJunglefowl.SegmentJunglefowl]

    nr_of_threads_per_policies = 1
    sequential_runs_per_thread = 1

    simulator = TelcoSimulator()
    #generated_historical_action_propensities, _, _ = simulator.get_marketing_history(export=False)
    generated_historical_action_propensities = list()
    start_time_stamp = datetime.today()

    generated_customers = generate_customers(100000, start_time_stamp.date())
    customer_df, portfolios_df = customers_to_dataframe(generated_customers)
    if dump_to_csv:
        customer_df.to_csv("output/customers.csv", index=False)
        portfolios_df.to_csv("output/portfolios.csv", index=False)

    all_actions = get_actions()
    if dump_to_csv:
        actions_df, offers_products_df = actions_to_dataframe(all_actions)
        actions_df.to_csv("output/actions.csv", index=False)
    offers_products_df.to_csv("output/offers_products.csv", index=False)

    policy_keywords = {'epsilon': 0.8, 'resort_batch_size': 50, "initial_trials": 99, "initial_conversions": 1,
                       'gold_threshold': None, 'silver_threshold': None, "current_base": generated_customers}

    day_to_simulate = 365
    out_logs, sample_chosen_action_logs = simulator.do_simulations(policies, policy_keywords,
                                                                   nr_of_threads_per_policies,
                                                                   sequential_runs_per_thread, generated_customers,
                                                                   all_actions, day_to_simulate, start_time_stamp,
                                                                   generated_historical_action_propensities)

    simulator.export_log_data(out_logs)

    simulator.plot_timelines(sample_chosen_action_logs, all_actions, show=True, save=True)

    # Plot performance
    simulator.plot_performance(out_logs, show=True, save=True)




