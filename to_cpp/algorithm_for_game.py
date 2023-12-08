import gurobipy as gp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


def get_recommended_route_simple(arc_set, origin_node, dest_node, trust, free_flow_time,
                                 report_threshold=1, num_dangerous_reports=None):
    """
        This function solves the static model for the given arc set, origin node, destination node and number of nodes.
        :param arc_set: available arc set at current time, type: list of tuples
        :param origin_node: origin node set, type: list of integers
        :param dest_node: destination node set, type: list of integers
        :param trust: trust value for each origin node (each player), type: dictionary {int: float}
        :param free_flow_time: free flow time for each arc, type: dictionary {(int, int): float}
        :param report_threshold: threshold for reports. If the number of dangerous reports is larger than this threshold,
        the arc is considered as dangerous and should be deleted from arc_set, type: integer
        :param num_dangerous_reports: number of dangerous reports of each arc, type: dictionary {(int, int): int}
        :return:
        """
    # check if the dangerous reports are larger than the threshold:
    # delete the arc from arc_set
    if num_dangerous_reports is not None:
        arc_set = {(i, j) for i, j in arc_set if num_dangerous_reports[i, j] <= report_threshold}
    # update shortest path set
    G = nx.DiGraph()
    for i, j in arc_set:
        G.add_edge(i, j, weight=free_flow_time[(i, j)])
    # find the shortest path
    shortest_path = {}
    travel_time = {}
    shortest_path_per_node = {}
    travel_time_per_node = {}
    for i in origin_node:
        j_min = sum(free_flow_time.values())
        for j in dest_node:
            shortest_path[i, j] = nx.dijkstra_path(G, i, j)
            travel_time[i, j] = nx.dijkstra_path_length(G, i, j)
            if travel_time[i, j] < j_min:
                j_min = travel_time[i, j]
                shortest_path_per_node[i] = shortest_path[i, j]
                travel_time_per_node[i] = travel_time[i, j]
    result = {}
    # make shortest_path_per_node to a dictionary with key as origin node and value as the shortest path
    # from origin node to destination node
    result['recommended_route'] = shortest_path_per_node
    return result


def get_recommended_route(arc_set, origin_node, dest_node, num_nodes, trust, demand, free_flow_time,
                          capacity=None, bpr_params=None, required_explore=None, eta=0, report_threshold=1,
                          num_dangerous_reports=None):
    """
        This function solves the static model for the given arc set, origin node, destination node and number of nodes.
        :param arc_set: available arc set at current time, type: list of tuples
        :param origin_node: origin node set, type: list of integers
        :param dest_node: destination node set, type: list of integers
        :param num_nodes: total number of nodes, type: integer
        :param bpr_params: BPR function parameters, type: [float, float]
        :param trust: trust value for each origin node (each player), type: dictionary {int: float}
        :param demand: demand for each origin node (each player), type: dictionary {int: float}
        :param capacity: capacity for each arc, type: dictionary {(int, int): float}
        :param free_flow_time: free flow time for each arc, type: dictionary {(int, int): float}
        :param required_explore: expected explored travelers number of each arc, type: dictionary {(int, int): float}
        :param eta: parameter for information gain, type: float
        :param report_threshold: threshold for reports. If the number of dangerous reports is larger than this threshold,
        the arc is considered as dangerous and should be deleted from arc_set, type: integer
        :param num_dangerous_reports: number of dangerous reports of each arc, type: dictionary {(int, int): int}
        :return:
    """
    # check if the dangerous reports are larger than the threshold:
    # delete the arc from arc_set
    if bpr_params is None:
        bpr_params = [0, 1]
    if num_dangerous_reports is not None:
        arc_set = [(i, j) for i, j in arc_set if num_dangerous_reports[i, j] <= report_threshold]
    # update shortest path set
    G = nx.DiGraph()
    for i, j in arc_set:
        G.add_edge(i, j, weight=free_flow_time[(i, j)])
    # find the shortest path
    shortest_path = {}
    travel_time = {}
    shortest_path_per_node = {}
    travel_time_per_node = {}
    shortest_path_link = {}
    for i in origin_node:
        j_min = sum(free_flow_time.values())
        for j in dest_node:
            shortest_path[i, j] = nx.dijkstra_path(G, i, j)
            travel_time[i, j] = nx.dijkstra_path_length(G, i, j)
            if travel_time[i, j] < j_min:
                j_min = travel_time[i, j]
                shortest_path_per_node[i] = shortest_path[i, j]
                travel_time_per_node[i] = travel_time[i, j]
        # convert shortest path to link formulation
        path = shortest_path_per_node[i]
        for j in range(len(path) - 1):
            cur_link = (path[j], path[j + 1])
            if cur_link not in shortest_path_link.keys():
                shortest_path_link[cur_link] = [i]
            else:
                shortest_path_link[cur_link].append(i)

    # set default value for capacity and bpr_params
    if capacity is None:
        capacity = {(i, j): 1 for i, j in arc_set}
    if bpr_params is None:
        bpr_params = [0, 1]
    # solve the static model
    print("Solve the optimization model to get optimal recommendation with weight on exploration.")
    m, x, z, obj1, obj2, x_val, z_val = solve_static_model(arc_set, origin_node, dest_node, num_nodes, bpr_params,
                                                           trust, demand, capacity, free_flow_time,
                                                           shortest_path_link, required_explore, eta)
    # print(x_val)
    # find the recommended route
    print("Find recommended route from optimized flow")
    m_origin, x_origin = get_flow_per_origin(x_val, arc_set, origin_node, dest_node, num_nodes)
    x_origin_val = {}
    for o in origin_node:
        x_origin_val[o] = m_origin.getAttr('x', x_origin[o])
    # get recommended route
    recommended_route = determine_path_flows(G, x_origin_val, origin_node, dest_node)
    result = {}
    result['planned_arc_flow_origin'] = x_origin_val
    result['planned_total_time'] = obj1
    result['planned_exploration_term'] = obj2
    result['recommended_route_with_flow'] = recommended_route
    result['recommended_route'] = parse_recommendation(recommended_route, True)
    return result


def solve_static_model(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust, demand, capacity, free_flow_time,
                       shortest_path, require_explore=None, eta=0):
    '''
    This function solves the static model for the given arc set, origin node, destination node and number of nodes.
    :param arc_set: available arc set at current time
    :param origin_node: origin node set
    :param dest_node: destination node set
    :param num_nodes: total number of nodes
    :param bpr_params: BPR function parameters
    :param trust: trust value for each origin node (each player)
    :param demand: demand for each origin node (each player)
    :param capacity: capacity for each arc
    :param free_flow_time: free flow time for each arc
    :param shortest_path: shortest path for each origin node given the current arc set A
    :param require_explore: required exploration of arcs
    :param eta: parameter for information gain
    the arc is considered as dangerous and should be deleted from arc_set
    :return:
    '''
    m = gp.Model("static_model")
    # Add variables by dictionary arc_set
    # x = m.addVars(arc_set, name="x", vtype=gp.GRB.INTEGER)
    x = m.addVars(arc_set, name="x")
    z = m.addVars(arc_set, name="z")
    # Add auxiliary variables
    gamma = m.addVars(arc_set, name="gamma")
    lambda_ = m.addVars(arc_set, name="lambda")
    mu_ = m.addVars(arc_set, name="mu")
    # Add objective function
    obj1 = gp.quicksum(free_flow_time[i, j] * (z[i, j] + bpr_params[0] * gamma[i, j] / capacity[i, j] ** bpr_params[1])
                       for i, j in arc_set)
    obj2 = 0
    # if we consider reports of blocked arcs, add an information gain term
    if require_explore is not None:
        # for i, j in arc_set:
        #     weight[i, j] = 1 / (require_explore[i, j] + 1e-3)
        #     obj2 += (z[i, j] - weight[i, j] / sum(weight.values()) * sum(demand.values())) * (
        #             z[i, j] - weight[i, j] / sum(weight.values()) * sum(demand.values()))
        # define binary variables
        u = m.addVars(arc_set, vtype=gp.GRB.BINARY, name="u")
        y = m.addVars(arc_set, name="y")
        for (i, j) in arc_set:
            obj2 += y[i, j]
        obj2 = eta * obj2
        print('Add flow variance objective')
        # Add constraints for u and z, if z>0, then u=1
        epsilon = 1e-3
        M = sum(demand.values())
        m.addConstrs((z[i, j] >= epsilon + M * (u[i, j] - 1) for i, j in arc_set), name='u_z_1')
        m.addConstrs((z[i, j] <= epsilon + M * u[i, j] for i, j in arc_set), name='u_z_2')
        # add constraints for y, y=max(ru-z, 0)
        m.addConstrs((y[i, j] >= u[i, j] * require_explore[i, j] - z[i, j] for i, j in arc_set), name='y')
    m.setObjective(obj1 + obj2, gp.GRB.MINIMIZE)
    # # Add SOCP constraints
    socp_cons_1 = m.addConstrs((4 * z[i, j] * z[i, j] + (lambda_[i, j] - 1) * (lambda_[i, j] - 1) -
                                (lambda_[i, j] + 1) * (lambda_[i, j] + 1) <= 0 for i, j in arc_set), name='socp_1')
    socp_cons_2 = m.addConstrs((4 * lambda_[i, j] * lambda_[i, j] + (mu_[i, j] - z[i, j]) * (mu_[i, j] - z[i, j]) <=
                                (mu_[i, j] + z[i, j]) * (mu_[i, j] + z[i, j]) for i, j in arc_set), name='socp_2')
    socp_cons_3 = m.addConstrs((4 * mu_[i, j] * mu_[i, j] + (gamma[i, j] - z[i, j]) * (gamma[i, j] - z[i, j]) <=
                                (gamma[i, j] + z[i, j]) * (gamma[i, j] + z[i, j]) for i, j in arc_set), name='socp_3')
    # Flow conservation constraints
    flow_cons_lb = m.addConstrs((gp.quicksum(x[i, j] for j in range(num_nodes) if (i, j) in arc_set) - gp.quicksum(
        x[j, i] for j in range(num_nodes) if (j, i) in arc_set) >= -1e-5 for i in range(num_nodes)
                                 if i not in origin_node + dest_node), name='flow_lb')
    flow_cons_ub = m.addConstrs((gp.quicksum(x[i, j] for j in range(num_nodes) if (i, j) in arc_set) - gp.quicksum(
        x[j, i] for j in range(num_nodes) if (j, i) in arc_set) <= 1e-5 for i in range(num_nodes)
                                 if i not in origin_node + dest_node), name='flow_ub')
    # demand constraints
    demand_cons = m.addConstrs((gp.quicksum(x[i, j] for j in range(num_nodes) if (i, j) in arc_set) - gp.quicksum(
        x[j, i] for j in range(num_nodes) if (j, i) in arc_set) == trust[i] * demand[i] for i in origin_node),
                               name='origin')
    # constraints between x and z
    x_z_constraints = m.addConstrs((z[i, j] - x[i, j] == 0 for i, j in arc_set
                                    if (i, j) not in list(shortest_path.keys())), name='total')
    x_z_constraints_add = m.addConstrs(
        (z[i, j] - x[i, j] == gp.quicksum((1 - trust[o]) * demand[o] for o in shortest_path[i, j])
         for i, j in arc_set if (i, j) in list(shortest_path.keys())), name='add')
    # solve the model
    m.Params.LogToConsole = 0
    m.Params.LogFile = 'algorithm_process.log'
    m.Params.DualReductions = 0
    m.Params.BarHomogeneous = 1
    m.Params.BarConvTol = 1e-6
    m.optimize()
    # get the value
    x_val = m.getAttr('x', x)
    z_val = m.getAttr('x', z)
    # get value of obj1
    obj1_val = obj1.getValue()
    # get value of obj2
    if require_explore is not None:
        obj2_val = obj2.getValue()
    else:
        obj2_val = 0
    # return the solution
    return m, x, z, obj1_val, obj2_val, x_val, z_val


def get_flow_per_origin(flow, arc_set, origin_node, dest_node, num_nodes):
    """
    Get the flow per origin node
    :param flow: flow of each arc, returned as x_val in the function solve_static_model
    :param arc_set: current arc set
    :param origin_node: origin node set
    :param dest_node: destination node set
    :param num_nodes: number of nodes
    :return: flow for each origin node and each arc
    """
    # build a simple model
    m = gp.Model("get_flow_per_origin")
    # Add variables by dictionary arc_set
    x = {}
    for o in origin_node:
        x[o] = m.addVars(arc_set, name=f"x{o}")
    # Add objective function
    m.setObjective(0, gp.GRB.MINIMIZE)
    # Flow conservation constraints
    flow_cons = m.addConstrs((gp.quicksum(x[o][o, j] - flow[o, j] for j in range(num_nodes) if (o, j) in arc_set)
                              - gp.quicksum(x[o][j, o] - flow[j, o] for j in range(num_nodes) if (j, o) in arc_set)
                              == 0 for o in origin_node), name='demand')
    flow_cons_no_origin = m.addConstrs((gp.quicksum(x[o][i, j] for j in range(num_nodes) if (i, j) in arc_set) -
                                        gp.quicksum(x[o][j, i] for j in range(num_nodes) if (j, i) in arc_set) == 0
                                        for i in range(num_nodes) if i not in origin_node + dest_node
                                        for o in origin_node), name='flow')
    total_flow_cons_ub = m.addConstrs(
        (gp.quicksum(x[o][i, j] for o in origin_node) <= flow[i, j] + 1e-4 for i, j in arc_set),
        name='total_ub')
    total_flow_cons_lb = m.addConstrs(
        (gp.quicksum(x[o][i, j] for o in origin_node) >= flow[i, j] - 1e-4 for i, j in arc_set),
        name='total_lb')
    # solve the model
    m.Params.LogToConsole = 0
    m.Params.LogFile = 'algorithm_process.log'
    m.Params.DualReductions = 0
    m.optimize()
    # return the solution
    return m, x


def determine_path_flows(G, arc_flows, origin_node, dest_node):
    """
    Determine path flows given arc flows.
    :param G: directed graph
    :param arc_flows: A dictionary where keys are arcs and values are their flows.
    :param origin_node: origin node set
    :param dest_node: destination node set
    :return: path flows
    """
    path_flow = {}
    for s in origin_node:
        # Get all simple paths from s to t
        paths = []
        for t in dest_node:
            paths += list(nx.all_simple_paths(G, source=s, target=t))
        # Create a new model
        m = gp.Model("path_flow_model")
        # Create variables for path flows
        x = m.addVars(len(paths), vtype=gp.GRB.CONTINUOUS, name="x")
        # Set objective: This is arbitrary as we're mainly interested in constraints
        m.setObjective(gp.quicksum(x[p] for p in range(len(paths))), gp.GRB.MAXIMIZE)
        # Constraints based on arc flows
        for (u, v), flow in arc_flows[s].items():
            paths_with_arc = [p for p in range(len(paths)) if (u, v) in zip(paths[p], paths[p][1:])]
            m.addConstr(gp.quicksum(x[p] for p in paths_with_arc) <= flow, f"arc_{u}_{v}")
        m.Params.LogToConsole = 0
        m.Params.LogFile = 'algorithm_process.log'
        m.Params.DualReductions = 0
        # Solve the model
        m.optimize()
        path_flow[s] = {}
        for i in range(len(paths)):
            # print(paths[i], x[i].x)
            if x[i].x > 1e-3:
                path_flow[s][tuple(paths[i])] = x[i].x
    return path_flow


def parse_tntp_file(file_path):
    """
    Parse the tntp file
    :param file_path:
    :return: coordinates of nodes
    """
    coords = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split('\t')
            if len(parts) >= 3 and parts[0].isdigit():
                node = int(parts[0]) - 1
                x, y = float(parts[1]), float(parts[2])
                coords[node] = (x, y)
    return coords


def parse_recommendation(recommendation_results, single_path=True):
    """
    Parse the recommendation results
    :param recommendation_results:
    :param single_path:
    :return:
    """
    recommendation = {}
    for origin, paths in recommendation_results.items():
        if not single_path:
           recommendation[origin] = [path for path in paths.keys() if paths[path] > 1e-3]
        else:
            # get uniformed weight for each path by flow
            total_flow = sum(paths.values())
            weight_arr = [flow / total_flow for flow in paths.values()]
            # randomly choose a path based on the weight
            path = random.choices(list(paths.keys()), weight_arr, k=1)[0]
            recommendation[origin] = [list(path)]
    return recommendation


def visualize_recommendation(recommendation, coord_file, arc_set):
    """
    Visualize the recommendation results
    :param recommendation: recommendation results
    :param coord_file: coordinates of nodes
    :param arc_set: arc set
    :return: None
    """

    # Parse the TNTP file to get coordinates
    coords = parse_tntp_file(coord_file)

    # Create a graph
    G = nx.Graph()

    # Add nodes with their coordinates
    for node, pos in coords.items():
        G.add_node(node, pos=pos)

    # Add edges from the provided arc set
    for edge in arc_set:
        G.add_edge(*edge)

    # Plot the graph
    fig = plt.figure(figsize=(10, 8), dpi=300)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300, font_size=12, font_weight='bold',
            edge_color='gray')

    # Highlight the recommendation routes
    colors = ['red', 'green', 'blue', 'orange', 'purple']  # Extend this list if more colors are needed
    legend_handles = []
    for idx, (origin, routes) in enumerate(recommendation.items()):
        color = colors[idx % len(colors)]  # Cycle through colors
        legend_label = f"Route from {origin}"
        # if the first element of the route is not a list
        if not isinstance(routes[0], list):
            routes = [routes]
        for route in routes:
            route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=route_edges, width=2, edge_color=color)
            legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=legend_label))

    # Add legend to the plot
    plt.legend(handles=legend_handles, loc='upper left', fontsize=16)

    # set title with font size 20
    plt.title('Recommendation Results', fontsize=20)
    return fig
