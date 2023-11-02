import gurobipy as gp
import networkx as nx


def get_recommended_route(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust, demand, capacity,
                          free_flow_time, explore=None, eta=0, report_threshold=1,
                          num_dangerous_reports=None):
    """
        This function solves the static model for the given arc set, origin node, destination node and number of nodes.
        :param arc_set: available arc sets at current time
        :param origin_node: origin node set (where the players are)
        :param dest_node: destination node set (where the players want to go)
        :param num_nodes: total number of nodes
        :param bpr_params: BPR function parameters (this is for our computation, if is usually set as [0.15, 4], but
        for current game setting, we set it as [0, 1] so that the number of people on each arc does not affect the
        travel time)
        :param trust: trust value for each origin node (each player in the game)
        :param demand: demand for each origin node (how many people the player represents, for current game setting,
        it is 1 for each player)
        :param capacity: capacity for each arc (for current game setting, it is 1 for each arc)
        :param free_flow_time: free flow time for each arc (for current game setting, it is exactly the travel time
        of each arc, i.e. how long it will take from i to j)
        :param explore: explored rate of each arc (this can be further defined, e.g., num_reports / expected_num_reports,
        if the rate is higher, it means that we have less preference to send people to this arc for exploration)
        :param eta: parameter for exploration (how much we want to explore)
        :param report_threshold: threshold for reports. If the number of dangerous reports is larger than this threshold,
        the arc is considered as dangerous and should be deleted from arc_set
        :param num_dangerous_reports: number of dangerous reports
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
    # solve the static model
    print("Solve the optimization model to get optimal recommendation with weight on exploration.")
    m, x, z, obj1, obj2, x_val, z_val = solve_static_model(arc_set, origin_node, dest_node, num_nodes, bpr_params,
                                                           trust, demand, capacity, free_flow_time,
                                                           shortest_path_per_node, explore, eta)
    print(x_val)
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
    result['recommended_route'] = recommended_route
    return result


def solve_static_model(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust, demand, capacity, free_flow_time,
                       shortest_path, explore=None, eta=0):
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
    :param explore: explored rate of each arc (dangerous / not dangerous)?
    :param eta: parameter for information gain
    :param report_threshold: threshold for reports. If the number of dangerous reports is larger than this threshold,
    the arc is considered as dangerous and should be deleted from arc_set
    :param num_dangerous_reports: number of dangerous reports
    :return:
    '''
    m = gp.Model("static_model")
    # Add variables by dictionary arc_set
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
    weight = {}
    if explore is not None:
        for i, j in arc_set:
            weight[i, j] = 1 / (explore[i, j] + 1e-3)
            obj2 += (z[i, j] - weight[i, j] / sum(weight.values()) * sum(demand.values())) * (
                    z[i, j] - weight[i, j] / sum(weight.values()) * sum(demand.values()))
        obj2 = eta * obj2
        print('Add flow variance objective')
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
    if explore is not None:
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
        for path in paths:
            if path == [5, 1, 0]:
                print(paths.index(path))
            if len(path) < 6:
                print(path)
        for (u, v), flow in arc_flows[s].items():
            paths_with_arc = [p for p in range(len(paths)) if (u, v) in zip(paths[p], paths[p][1:])]
            if u == 5 and v == 1:
                print(paths_with_arc)
                print(flow)
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
