import networkx as nx
import cvxpy as cp
import gurobipy as gp
from read_data import report_net_data_tntp, read_flow_data_tntp, find_shortest_path, read_flow_data_trip_tntp


def solve_static_model_cp(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust, demand, capacity,
                          free_flow_time, shortest_path):
    # add an index dictionary for arc_set
    arc_index = {}
    for i in range(len(arc_set)):
        arc_index[arc_set[i]] = i
    # build a cvxpy model
    # add variables by arc_set
    x = cp.Variable(len(arc_set), name="x")
    z = cp.Variable(len(arc_set), name="z")
    # add constraints
    constraints = []
    # add flow conservation constraints
    for i in range(num_nodes):
        if i in origin_node:
            constraints.append(cp.sum([x[arc_index[i, j]] for j in range(num_nodes) if (i, j) in arc_set]) - cp.sum(
                [x[arc_index[j, i]] for j in range(num_nodes) if (j, i) in arc_set]) == trust[i] * demand[i])
        elif i not in dest_node:
            # out_flow = 0
            # in_flow = 0
            # for j in range(num_nodes):
            #     if (i, j) in arc_set:
            #         out_flow += x[arc_index[i, j]]
            #     if (j, i) in arc_set:
            #         in_flow += x[arc_index[j, i]]
            constraints.append(cp.sum([x[arc_index[i, j]] for j in range(num_nodes) if (i, j) in arc_set]) - cp.sum(
                [x[arc_index[j, i]] for j in range(num_nodes) if (j, i) in arc_set]) == 0)
    # add x_z constraints
    for i in range(len(arc_set)):
        arc = arc_set[i]
        if arc not in list(shortest_path.keys()):
            constraints.append(x[i] == z[i])
        else:
            constraints.append(z[i] == x[i] + cp.sum([(1 - trust[o]) * demand[o] for o in shortest_path[arc]]))
        constraints.append(z[i] >= 0)
        constraints.append(x[i] >= 0)
    # define objective function
    obj = cp.Minimize(cp.sum([z[arc_index[i, j]] * free_flow_time[i, j] + free_flow_time[i, j] * bpr_params[0] *
                              z[arc_index[i, j]] ** (bpr_params[1] + 1) / capacity[i, j] ** bpr_params[1]
                              for i, j in arc_set]))
    # solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True, solver=cp.GUROBI)
    # print the results
    print("status:", prob.status)
    print("optimal value", prob.value)
    print('Average time', prob.value / sum(demand.values()))
    z_val = {}
    x_val = {}
    for i in range(len(arc_set)):
        arc = arc_set[i]
        x_val[arc] = x[i].value
        z_val[arc] = z[i].value
    print(bpr(z_val, bpr_params, free_flow_time, capacity, arc_set))
    # print("optimal var", x.value)
    # print("optimal var", z.value)
    return prob, x, z, prob.value, x_val, z_val


def solve_static_model(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust, demand, capacity, free_flow_time,
                       shortest_path, explore=None, eta=0, report_threshold=1):
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
        print('Add information gain objective')
    m.setObjective(obj1 + obj2, gp.GRB.MINIMIZE)
    # m.setObjective(gp.quicksum(free_flow_time[i, j] * (
    #         z[i, j] + bpr_params[0] * gamma[i, j])
    #                            for i, j in arc_set), gp.GRB.MINIMIZE)
    # # Add SOCP constraints
    socp_cons_1 = m.addConstrs((4 * z[i, j] * z[i, j] + (lambda_[i, j] - 1) * (lambda_[i, j] - 1) -
                                (lambda_[i, j] + 1) * (lambda_[i, j] + 1) <= 0 for i, j in arc_set), name='socp_1')
    socp_cons_2 = m.addConstrs((4 * lambda_[i, j] * lambda_[i, j] + (mu_[i, j] - z[i, j]) * (mu_[i, j] - z[i, j]) <=
                                (mu_[i, j] + z[i, j]) * (mu_[i, j] + z[i, j]) for i, j in arc_set), name='socp_2')
    socp_cons_3 = m.addConstrs((4 * mu_[i, j] * mu_[i, j] + (gamma[i, j] - z[i, j]) * (gamma[i, j] - z[i, j]) <=
                                (gamma[i, j] + z[i, j]) * (gamma[i, j] + z[i, j]) for i, j in arc_set), name='socp_3')
    # socp_cons_1 = m.addConstrs((z[i, j] * z[i, j] - lambda_[i, j] <= 0 for i, j in arc_set), name='socp_1')
    # socp_cons_2 = m.addConstrs((lambda_[i, j] * lambda_[i, j] - mu_[i, j] * z[i, j] <= 0 for i, j in arc_set),
    #                            name='socp_2')
    # socp_cons_3 = m.addConstrs((mu_[i, j] * mu_[i, j] - gamma[i, j] * z[i, j] <= 0 for i, j in arc_set), name='socp_3')
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
    m.Params.LogFile = 'static_model.log'
    m.Params.DualReductions = 0
    m.Params.BarHomogeneous = 1
    m.Params.BarConvTol = 1e-6
    m.optimize()
    # get the value
    x_val = m.getAttr('x', x)
    z_val = m.getAttr('x', z)
    # get value of obj1
    obj_val = obj1.getValue()
    # # get value of obj2
    # obj2_val = sum((z_val[i, j] - sum(demand.values()) / len(arc_set)) ** 2 for i, j in arc_set)
    # return the solution
    return m, x, z, obj_val, x_val, z_val


def get_flow_per_origin(flow, arc_set, origin_node, dest_node, num_nodes):
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
    m.Params.LogFile = 'out_sample_test.log'
    m.Params.DualReductions = 0
    m.optimize()
    # return the solution
    return m, x


def true_time(flow, bpr_params, free_flow_time, capacity, arc_set, est_trust, true_trust,
              num_nodes, origin_node, dest_node, demand, shortest_path):
    '''
    This function defines the BPR function.
    :param x:
    :return:
    '''
    # get the flow from every origin
    m_origin, flow_origin = get_flow_per_origin(flow, arc_set, origin_node, dest_node, num_nodes)
    # assume that we assign the recommendation for all the people, then true flow following the recommendation is
    # flow * true_trust / est_trust
    trust_ratio = {}
    for o in origin_node:
        if est_trust[o] == 0:
            trust_ratio[o] = 0
            true_trust[o] = 0
        else:
            trust_ratio[o] = true_trust[o] / est_trust[o]
    true_flow = {}
    for i, j in arc_set:
        true_flow[i, j] = sum(flow_origin[o][i, j].x * trust_ratio[o] for o in origin_node)
    # we compute the total flow on each link
    total_flow = {}
    for i, j in arc_set:
        if (i, j) not in list(shortest_path.keys()):
            total_flow[i, j] = true_flow[i, j]
        else:
            total_flow[i, j] = true_flow[i, j] + sum((1 - true_trust[o]) * demand[o] for o in shortest_path[i, j])
    # we compute the true travel time
    bpr_term = {}
    for i, j in arc_set:
        bpr_term[i, j] = (1 + bpr_params[0] * (total_flow[i, j] / capacity[i, j]) ** bpr_params[1]) * total_flow[i, j] * \
                         free_flow_time[i, j]
    bpr = sum(bpr_term[i, j] for i, j in arc_set)
    # print(bpr_term)
    return bpr


def bpr(x, bpr_params, free_flow_time, capacity, arc_set):
    bpr = sum(x[i, j] * free_flow_time[i, j] * (1 + bpr_params[0] * (x[i, j] / capacity[i, j]) ** bpr_params[1])
              for i, j in arc_set)
    return bpr


def test_simple():
    # define parameters for static model
    num_nodes = 8
    origin_node = [0, 3, 6]
    dest_node = [2, 5, 7]
    # trust = {0: 0.8, 3: 0.3, 6: 0.5}
    single_trust = 1
    trust = {0: single_trust, 3: single_trust, 6: single_trust}
    demand = {0: 90 / 60, 3: 50 / 60, 6: 100 / 60}
    arc_set = [(0, 1), (0, 3), (1, 2), (3, 4), (3, 6), (4, 2), (4, 5), (4, 7), (6, 7)]
    capacity = {}
    free_flow_time = {}
    for (i, j) in arc_set:
        capacity[i, j] = 1  # units: veh/sec
        free_flow_time[i, j] = 20  # units: minutes
    free_flow_time[0, 1] = 40
    free_flow_time[1, 2] = 40
    free_flow_time[4, 5] = 16
    shortest_path = {(0, 3): [0], (3, 4): [0, 3], (4, 5): [0, 3], (6, 7): [6]}
    bpr_params = [0.15, 4]
    # solve the static model
    m, x, z, obj, x_val, z_val = solve_static_model(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust,
                                                    demand, capacity, free_flow_time, shortest_path)
    print(x_val)
    print(true_time(x_val, bpr_params, free_flow_time, capacity, arc_set, 0.25, single_trust, num_nodes, origin_node,
                    dest_node, demand, shortest_path) / sum(demand.values()))


def test_sioux_falls():
    # for sioux falls
    net_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_net.tntp'
    num_nodes, num_arcs, arc_set, free_flow_time, capacity = report_net_data_tntp(net_file_name)
    flow_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_flow.tntp'
    origin_node = [6 - 1, 10 - 1, 23 - 1]
    dest_node = [1 - 1, 13 - 1, 18 - 1, 20 - 1]
    # demand = read_flow_data_tntp(flow_file_name, origin_node)
    trip_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_trips.tntp'
    demand = read_flow_data_trip_tntp(trip_file_name, origin_node)
    bpr_params = [0.15, 4]
    # solve the static model
    shortest_path_per_node, _ = find_shortest_path(free_flow_time, origin_node, dest_node)
    # convert shortest path to arc set
    shortest_path = {}
    for i in origin_node:
        path = shortest_path_per_node[i]
        for j in range(len(path) - 1):
            cur_link = (path[j], path[j + 1])
            if cur_link not in shortest_path.keys():
                shortest_path[cur_link] = [i]
            else:
                shortest_path[cur_link].append(i)
    # define trust parameters
    single_trust = 1
    trust = {}
    for i in origin_node:
        trust[i] = single_trust
    # m, x, z, obj, x_val, z_val = solve_static_model_cp(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust,
    #                                               demand, capacity, free_flow_time, shortest_path)
    # print(obj)
    m, x, z, obj, x_val, z_val = solve_static_model(arc_set, origin_node, dest_node, num_nodes, bpr_params, trust,
                                                    demand,
                                                    capacity, free_flow_time, shortest_path)
    # print(x_val)
    print('average travel time for each vehicle:', obj / sum(demand.values()))
    # get true travel time
    single_true_trust = 1
    true_trust = {}
    for i in origin_node:
        true_trust[i] = single_true_trust
    true_time_val = true_time(x_val, bpr_params, free_flow_time, capacity, arc_set, trust, true_trust,
                              num_nodes, origin_node, dest_node, demand, shortest_path)
    print('average travel time with true trust:', true_time_val / sum(demand.values()))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_sioux_falls()
