import networkx as nx
def report_net_data_tntp(file_name):
    # read the data from tntp file
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # get the number of nodes and arcs
    num_nodes = int(lines[0].split('> ')[1])
    num_arcs = int(lines[3].split('> ')[1])
    # get the information for each link
    arc_set = []
    free_flow_time = {}
    capacity = {}
    for i in range(9, 9 + num_arcs):
        line = lines[i].split('\t')
        init_node, term_node = int(line[1]) - 1, int(line[2]) - 1
        arc_set.append((init_node, term_node))
        capacity[init_node, term_node] = float(line[3]) / 3600
        free_flow_time[init_node, term_node] = float(line[5])
    return num_nodes, num_arcs, arc_set, free_flow_time, capacity


def read_flow_data_tntp(file_name, origin_node):
    # read flow file
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # initialize demand for origin node
    demand = {}
    for i in origin_node:
        demand[i] = 0
    # read each line to get demand
    for line in lines[1:]:
        line = line.split('\t')
        init_node, term_node = int(line[0]) - 1, int(line[1]) - 1
        if init_node in origin_node:
            demand[init_node] += float(line[2]) / 3600
    return demand


def read_flow_data_trip_tntp(file_name, origin_node):
    # read flow file
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # initialize demand for origin node
    demand = {}
    for i in origin_node:
        demand[i] = 0
    # read each line to get demand
    record = False
    node = 0
    for line in lines[5:]:
        # remove all the spaces and '\n'
        line = line.replace(' ', '').strip('\n').split('\t')
        if line[0] == '':
            continue
        if line[0] == 'Origin':
            node = int(line[1]) - 1
            if node in origin_node:
                demand[node] = 0
                record = True
            else:
                record = False
        else:
            if record:
                demand_line = line[0].split(';')
                for demand_single in demand_line:
                    if demand_single != '':
                        demand[node] += float(demand_single.split(':')[1]) / 3600
    return demand


def find_shortest_path(free_flow_time, origin_node, dest_node):
    # find the shortest path from origin node to destination node
    # initialize the graph
    G = nx.DiGraph()
    for i in free_flow_time.keys():
        G.add_edge(i[0], i[1], weight=free_flow_time[i])
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
    return shortest_path_per_node, travel_time_per_node



if __name__ == '__main__':
    # for sioux falls
    net_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_net.tntp'
    num_nodes, num_arcs, arc_set, free_flow_time, capacity = report_net_data_tntp(net_file_name)
    flow_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_flow.tntp'
    origin_node = [6 - 1, 10 - 1, 23 - 1]
    dest_node = [1 - 1, 13 - 1, 18 - 1, 20 - 1]
    # demand = read_flow_data_tntp(flow_file_name, origin_node)
    trip_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_trips.tntp'
    demand = read_flow_data_trip_tntp(trip_file_name, origin_node)
    shortest_path_per_node, travel_time_per_node = find_shortest_path(free_flow_time, origin_node, dest_node)
    # get maximize capacity
    max_capacity = max(capacity.values())
    print(max_capacity ** 4)
    print(min(capacity.values()) ** 4)

