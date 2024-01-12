import json

import pandas as pd
from eth_account.account import Account

import streamlit as st
from helper_funcs import (create_distance_matrix_azure,
                          create_distance_matrix_osrm, decrypt_data_from_chain,
                          encrypt_and_transfer_data, load_data, optimize,
                          plot_routes)

st.set_page_config(
    page_title="Route Optimization - DEMO",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables if they don't exist
if "distance_matrix_created" not in st.session_state:
    st.session_state["distance_matrix_created"] = False

if "data_bytes" not in st.session_state:
    st.session_state["data_bytes"] = None

if "encrypted_data" not in st.session_state:
    st.session_state["encrypted_data"] = None

if "tx_address" not in st.session_state:
    st.session_state["tx_address"] = None



azure_primary_key = st.secrets["azure_instance_maps"]["primary_key"]
alice_acc = Account.from_key(st.secrets["wallet"]["alice_private_key"])
bob_acc = Account.from_key(st.secrets["wallet"]["bob_private_key"])
bobs_asymmetric_public_key = st.secrets["user_asymmetric_keys"]["bobs_asymmetric_public_key"]
bobs_asymmetric_private_key = st.secrets["user_asymmetric_keys"]["bobs_asymmetric_private_key"]
mapbox_pk = st.secrets["mapbox"]["private_key"]


# Sidebar
with st.sidebar:
    st.title("Demo Dataset")
    st.subheader("Options")
    number_of_orders = st.slider("Number of orders", min_value=1, max_value=8, value=5, step=1, key="orders")
    number_of_vehicles = st.slider("Number of vehicles", min_value=1, max_value=5, value=2, step=1, key="vehicles")
    number_of_depots = st.slider(
        "Number of depots", min_value=1, max_value=10, value=1, step=1, disabled=True, key="depots"
    )
    vehicle_max_travel_distance = st.slider(
        "Vehicle max travel distance",
        min_value=1000,
        max_value=50000,
        value=25000,
        step=1000,
        key="vehicle_distance",
        help="The maximum distance a vehicle can travel before returning to the depot.",
    )

# Main page
st.header("Demo")

# Load dataset based on side-bar parameters
orders_data, vehicles_data, depots_data = load_data(number_of_orders, number_of_vehicles, number_of_depots)

st.subheader("1. Dataset Preview")
with st.expander("Orders"):
    st.dataframe(orders_data)

with st.expander("Vehicles"):
    st.dataframe(vehicles_data)

with st.expander("Depots"):
    st.dataframe(depots_data)

st.subheader("2. Transfer Data Model")
if st.button("Create distance matrix"):
    with st.expander("Data Model"):
        distance_matrix = create_distance_matrix_osrm(depots_data, orders_data)

        st.write("Data Model")
        data_model = {}
        data_model["distance_matrix"] = distance_matrix.values.tolist()
        data_model["num_vehicles"] = number_of_vehicles
        data_model["depot"] = 0  # The depot is the starting point of the route. In this case, it refers to depot1.
        data_model["demands"] = [0] + orders_data["Demand"].values.tolist()
        data_model["vehicle_capacities"] = vehicles_data["vehicle_capacity"].values.tolist()
        st.write(data_model)

        data = json.dumps(data_model)

        # Encrypt data with Bob's public key and transfer to Bob
        tx, data_nft = encrypt_and_transfer_data(
            data=data,
            from_account=alice_acc,
            recipient_wallet_address=bob_acc.address,
            recipient_asymmetric_public_key=bobs_asymmetric_public_key,
        )

        st.write("data_nft")
        st.write(data_nft)
        st.session_state["data_nft"] = data_nft

        st.write(f"Transation {tx}")
        st.session_state["tx_address"] = tx


if st.session_state["tx_address"] is not None:
    st.subheader("3. Optimized Routes")
    tx_hash = st.session_state["tx_address"].transactionHash.hex()
    st.write(f"Transaction hash: {tx_hash}")
    tx_link = "https://mumbai.polygonscan.com/tx/" + tx_hash
    st.write(tx_link)
    st.write(st.session_state["data_nft"])

    if st.button("Consume data & optimize"):
        decrypted_data = decrypt_data_from_chain(
            data_nft=st.session_state["data_nft"],
            asymmetric_pk=bobs_asymmetric_private_key,
            data_label="my_data",
        )

        solution = optimize(decrypted_data)
        st.write(solution)

        locations_df = pd.concat([depots_data, orders_data], ignore_index=True)

        route_df_list = [
            pd.DataFrame({"Destination": route}).join(locations_df, on="Destination") for route in solution
        ]

        # Concatenate all DataFrames with appropriate keys and plot figure
        route_dfs = pd.concat(route_df_list, keys=[f"Truck_{i}" for i in range(len(solution))])

        fig = plot_routes(route_dfs, mapbox_access_token=mapbox_pk)

        st.plotly_chart(fig, use_container_width=True)
