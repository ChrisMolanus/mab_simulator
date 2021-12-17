import operator
import os
from typing import Dict, List, Any

import numpy as np
import yaml
import streamlit as st
from pandas import DataFrame

import bayesianGroundhog
import epsilonRingtail
import randomCrayfish
import segmentJunglefowl


from simulator import get_performance_plot

st.header("Telco Marketing Championship Lead Board")

policies = [randomCrayfish.RandomCrayfish, segmentJunglefowl.SegmentJunglefowl, bayesianGroundhog.BayesianGroundhog,
            epsilonRingtail.EpsilonRingtail]

# Get Data
output_dir = "output"
policy_icons = dict()
plot_dict: Dict[str, List[Dict[str, Any]]] = dict()
plot_dfs: Dict[str, DataFrame] = dict()
last_mean_value: Dict[str, float] = dict()
for policy_class in policies:
    policy_name = policy_class.__name__
    plot_dict[policy_name] = list()
    with open(os.path.join(output_dir, policy_name + ".yaml"), "r") as f:
        log = yaml.safe_load(f)
        for ts, sim_values in log.items():
            plot_dict[policy_name].append({"ts": ts, "mean": np.mean(sim_values), "std": np.std(sim_values)})
            last_mean_value[policy_name] = float(np.mean(sim_values))
        plot_dfs[policy_name] = DataFrame(plot_dict[policy_name])
    plot_dfs[policy_name] = DataFrame(plot_dict[policy_name])
    policy_icons[policy_name] = policy_class.icon


# Render table header
col_h0, col_h1, col_h2, col_h3 = st.columns((1, 1, 2, 1))
with col_h0:
    st.write("Rank")
with col_h1:
    st.write("")

with col_h2:
    st.write("Team")

with col_h3:
    st.write("Score")

# Render table ranking rows
i = 1
sorted_policies = sorted(last_mean_value.items(), key=operator.itemgetter(1), reverse=True)
for policy_name, euro in sorted_policies:
    col0, col1, col2, col3 = st.columns((1, 1, 2, 1))
    with col0:
        st.subheader(str(i))
    with col1:
        st.image(policy_icons[policy_name], width=100)
    with col2:
        st.subheader(policy_name)
    with col3:
        st.subheader(f"€{euro}")
    i += 1

# Plot performance graph
ordered_policies_by_clv = sorted(last_mean_value, key=last_mean_value.get)
ordered_policies_by_clv.reverse()
fig = get_performance_plot(plot_dfs, ordered_policies_by_clv)
st.pyplot(fig)
