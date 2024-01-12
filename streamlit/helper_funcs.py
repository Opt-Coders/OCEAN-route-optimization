import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from ecies import decrypt as asymmetric_decrypt
from ecies import encrypt as asymmetric_encrypt
from eth_utils import decode_hex
from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def load_data(number_of_orders: int, number_of_vehicles: int, number_of_depots: int):
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    orders_data_path = os.path.join(dir_path, "orders_lmd.csv")
    vehicles_data_path = os.path.join(dir_path, "vehicles_lmd.csv")
    depots_data_path = os.path.join(dir_path, "depots_lmd.csv")

    orders_df = pd.read_csv(
        orders_data_path, usecols=["Latitude", "Longitude", "Name", "Demand"], nrows=number_of_orders
    )
    orders_df["Demand"] = orders_df["Demand"].round(0).astype(int)

    vehicles_df = pd.read_csv(
        vehicles_data_path,
        usecols=["vehicle_id", "vehicle_type", "assigned_depot", "vehicle_capacity"],
        nrows=number_of_vehicles,
    )    
    
    depots_df = pd.read_csv(depots_data_path, usecols=["Latitude", "Longitude", "Name"], nrows=number_of_depots)
    

    return orders_df, vehicles_df, depots_df


def create_distance_matrix_azure(depots_df, orders_df, azure_subscription_key):
    """
    Create the distance & travel matrix using Azure Maps.
    """

    df = pd.concat([depots_df, orders_df], ignore_index=True)
    points = df[["Longitude", "Latitude"]].to_numpy()
    assert points.shape[0] ** points.shape[1] <= 100, "The maximum size of a matrix for sync request is 100."

    multi_point = {"type": "MultiPoint", "coordinates": points.tolist()}
    location_data = {"origins": multi_point, "destinations": multi_point}

    params = {"subscription-key": azure_subscription_key, "api-version": "1.0"}

    optional_params = {
        "travelMode": "truck",
        "routeType": "eco",
    }

    params.update(optional_params)

    url = "https://atlas.microsoft.com/route/matrix/sync/json"
    r = requests.post(url, params=params, json=location_data)
    r.raise_for_status()

    routes = json.loads(r.content)

    route_matrix = {
        "lengthInMeters": [],
        "travelTimeInSeconds": [],
        "trafficDelayInSeconds": [],
        "trafficLengthInMeters": [],
    }

    for i in routes["matrix"]:
        route_matrix["lengthInMeters"].append([j["response"]["routeSummary"]["lengthInMeters"] for j in i])
        route_matrix["travelTimeInSeconds"].append([j["response"]["routeSummary"]["travelTimeInSeconds"] for j in i])
        route_matrix["trafficDelayInSeconds"].append(
            [j["response"]["routeSummary"]["trafficDelayInSeconds"] for j in i]
        )
        route_matrix["trafficLengthInMeters"].append(
            [j["response"]["routeSummary"]["trafficLengthInMeters"] for j in i]
        )

    locations_list = df["Name"].values.tolist()
    distances_df = pd.DataFrame(route_matrix["lengthInMeters"], index=locations_list, columns=locations_list)
    durations_df = pd.DataFrame(route_matrix["travelTimeInSeconds"], index=locations_list, columns=locations_list)
    traffic_delay_df = pd.DataFrame(
        route_matrix["trafficDelayInSeconds"], index=locations_list, columns=locations_list
    )
    traffic_length_df = pd.DataFrame(
        route_matrix["trafficLengthInMeters"], index=locations_list, columns=locations_list
    )

    combined_df = pd.concat(
        [distances_df, durations_df, traffic_delay_df, traffic_length_df],
        keys=["distances", "durations", "traffic_delay_seconds", "traffic_length_meters"],
    )

    return combined_df


def create_distance_matrix_osrm(depot_df, orders_df):
    location_df = pd.concat([depot_df, orders_df], ignore_index=True)[["Longitude", "Latitude"]].to_numpy()
    locations = ";".join([f"{lon},{lat}" for lon, lat in location_df])

    r = requests.get(f"http://router.project-osrm.org/table/v1/driving/{locations}")
    r.raise_for_status()
    routes = json.loads(r.content)

    df = pd.DataFrame(
        routes["durations"],
        columns=depot_df["Name"].tolist() + orders_df["Name"].tolist(),
        index=depot_df["Name"].tolist() + orders_df["Name"].tolist(),
    )

    df = df.round().astype(int)

    return df


def encrypt_and_transfer_data(
    data, from_account, recipient_wallet_address, recipient_asymmetric_public_key, data_label="my_data", rpc_url = "https://rpc-mumbai.maticvigil.com"
):  
    config = get_config_dict(rpc_url)
    ocean = Ocean(config)

    data_nft = ocean.data_nft_factory.create({"from": from_account}, "NFT1", "NFT1")

    data_label = data_label
    data_values = data

    data_values_string_b = data_values.encode("utf-8")
    data_values_encrypted = asymmetric_encrypt(recipient_asymmetric_public_key, data_values_string_b)
    data_values_encrypted_h = data_values_encrypted.hex()

    data_nft.set_data(data_label, data_values_encrypted_h, {"from": from_account})

    token_id = 1
    tx = data_nft.safeTransferFrom(from_account.address, recipient_wallet_address, token_id, {"from": from_account})

    return tx, data_nft


def decrypt_data_from_chain(data_nft, asymmetric_pk, data_label="my_data"):
    data_nft_encrypted = data_nft.get_data(data_label)
    value_enc_b = decode_hex(data_nft_encrypted)
    value_b = asymmetric_decrypt(asymmetric_pk, value_enc_b)
    decrypted_data = value_b.decode("ascii")
    decrypted_data_j = json.loads(decrypted_data)
    return decrypted_data_j


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    st.write(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        st.write(plan_output, unsafe_allow_html=True)
        total_distance += route_distance
        total_load += route_load
    st.write(f"Total distance of all routes: {total_distance}m", unsafe_allow_html=True)
    st.write(f"Total load of all routes: {total_load}", unsafe_allow_html=True)
    return None


def optimize(data, vehicle_max_travel_distance=25000):
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,
        vehicle_max_travel_distance,
        True,
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data["vehicle_capacities"],
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(5)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(data, manager, routing, solution)

        def get_routes(solution, routing, manager):
            routes = []
            for route_nbr in range(routing.vehicles()):
                index = routing.Start(route_nbr)
                route = [manager.IndexToNode(index)]
                while not routing.IsEnd(index):
                    index = solution.Value(routing.NextVar(index))
                    route.append(manager.IndexToNode(index))
                routes.append(route)
            return routes

        routes = get_routes(solution, routing, manager)

        if routes:
            return routes
        else:
            # st.write("No solution found.")
            return None
    else:
        return None


def plot_routes(route_dfs, mapbox_access_token):
    fig = go.Figure()
    traces = []
    for truck in route_dfs.index.get_level_values(0).unique():
        n_destinations = len(route_dfs.loc[truck])
        lat = route_dfs.loc[truck]["Latitude"].values.tolist()
        lon = route_dfs.loc[truck]["Longitude"].values.tolist()
        text = route_dfs.loc[truck]["Name"].values.tolist()

        warehouse_symbol = "place-of-worship"
        customer_symbol = "circle"

        symbols = [warehouse_symbol] + [customer_symbol for i in range(n_destinations - 2)] + [warehouse_symbol]

        if (n_destinations == 2) and (lat[0] == lat[1] and lon[0] == lon[1]):
            # Skip the truck if it doesn't have any deliveries (i.e. it's just going to the warehouse)
            print(f"Skipping truck {truck} because it doesn't have any deliveries.")
            continue

        trace = go.Scattermapbox(
            mode="markers+lines+text",
            lon=lon,
            lat=lat,
            marker={"size": 15, "symbol": symbols, "allowoverlap": True},
            textposition="top center",
            textfont=dict(color="white"),
            text=text,
            name=truck,
        )
        traces.append(trace)

    for trace in traces:
        fig.add_trace(trace)

    center = {"lon": route_dfs["Longitude"].mean(), "lat": route_dfs["Latitude"].mean()}
    fig.update_layout(
        margin={"l": 0, "t": 0, "b": 0, "r": 0},
        autosize=True,
        showlegend=True,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(lat=center["lat"], lon=center["lon"]),
            pitch=0,
            zoom=11.5,
            style="dark",
        ),
    )

    return fig
