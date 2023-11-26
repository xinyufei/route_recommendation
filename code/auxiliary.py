import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import static_model
import bandit
import read_data


class TestExpectation:
    def __init__(self, network_structure, mu, sigma):
        self.network_structure = network_structure
        self.mu = mu
        self.sigma = sigma
        self.cur_bandit = bandit.Bandit(network_structure, mu)
        self.cur_bandit.get_solution()
        self.m_origin, self.flow_origin = static_model.get_flow_per_origin(
            self.cur_bandit.solution['x_val'], self.cur_bandit.arc_set,
            self.cur_bandit.origin_node, self.cur_bandit.dest_node, self.cur_bandit.num_nodes)

    def traveltime(self, x):
        true_flow = {}
        for i, j in self.cur_bandit.arc_set:
            true_flow[i, j] = sum(self.flow_origin[o][i, j].x * x / mu for o in self.cur_bandit.origin_node)
        # we compute the total flow on each link
        total_flow = {}
        for i, j in self.cur_bandit.arc_set:
            if (i, j) not in list(self.cur_bandit.shortest_path.keys()):
                total_flow[i, j] = true_flow[i, j]
            else:
                total_flow[i, j] = true_flow[i, j] + sum((1 - x) * self.cur_bandit.demand[o]
                                                         for o in self.cur_bandit.shortest_path[i, j])
        # we compute the true travel time
        bpr_term = {}
        for i, j in self.cur_bandit.arc_set:
            bpr_term[i, j] = (1 + self.cur_bandit.bpr_params[0] * (total_flow[i, j] / self.cur_bandit.capacity[i, j])
                              ** self.cur_bandit.bpr_params[1]) * total_flow[i, j] * self.cur_bandit.free_flow_time[
                                 i, j]
        bpr = sum(bpr_term[i, j] for i, j in self.cur_bandit.arc_set)
        # print(bpr_term)
        return bpr

    # Define the function to integrate: f(x) * normal PDF
    def integrand(self, x):
        return self.traveltime(x) * stats.norm.pdf(x, loc=self.mu, scale=self.sigma)

    def test(self):
        # Compute the integral over all x to find the expectation of f(X)
        result = integrate.quad(self.integrand, -np.inf, np.inf)
        avg_time = result[0] / sum(self.cur_bandit.demand.values())
        return result, avg_time


if __name__ == '__main__':
    # for sioux falls, read all the files
    net_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_net.tntp'
    num_nodes, num_arcs, arc_set, free_flow_time, capacity = read_data.report_net_data_tntp(net_file_name)
    origin_node = [6 - 1, 10 - 1, 23 - 1]
    dest_node = [1 - 1, 13 - 1, 18 - 1, 20 - 1]
    trip_file_name = '../TransportationNetworks/SiouxFalls/SiouxFalls_trips.tntp'
    demand = read_data.read_flow_data_trip_tntp(trip_file_name, origin_node)
    bpr_params = [0.15, 4]
    # solve the static model
    shortest_path_per_node, _ = read_data.find_shortest_path(free_flow_time, origin_node, dest_node)
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
    # assign parameters to network structure
    network_structure = {
        'arc_set': arc_set,
        'origin_node': origin_node,
        'dest_node': dest_node,
        'num_nodes': num_nodes,
        'bpr_params': bpr_params,
        'demand': demand,
        'capacity': capacity,
        'free_flow_time': free_flow_time,
        'shortest_path': shortest_path
    }
    # set the mean and standard deviation of the normal distribution
    mu = 0.8
    sigma = 1e-2
    test = TestExpectation(network_structure, mu, sigma)
    result, avg_time = test.test()
    print(result, avg_time)
    print(test.traveltime(mu) / sum(test.cur_bandit.demand.values()))
    print(test.cur_bandit.pull(mu))
