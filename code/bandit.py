import static_model
import numpy as np


# define a bandit class with given trust rate
class Bandit:
    # initialize the bandit with given trust rate
    def __init__(self, network_structure, trust_rate):
        self.trust_rate_mean = trust_rate
        self.trust_rate_std = 0.1
        self.network_structure = network_structure

        # assign values of network structure dictionary to variables
        self.arc_set = self.network_structure['arc_set']
        self.origin_node = self.network_structure['origin_node']
        self.dest_node = self.network_structure['dest_node']
        self.num_nodes = self.network_structure['num_nodes']
        self.bpr_params = self.network_structure['bpr_params']
        self.demand = self.network_structure['demand']
        self.capacity = self.network_structure['capacity']
        self.free_flow_time = self.network_structure['free_flow_time']
        self.shortest_path = self.network_structure['shortest_path']

        self.solution = {}
        self.trust = {}

    def get_solution(self):
        trust_rate = self.trust_rate_mean
        for i in self.origin_node:
            self.trust[i] = trust_rate
        m, x, z, obj, x_val, z_val = static_model.solve_static_model(
            self.arc_set, self.origin_node, self.dest_node, self.num_nodes, self.bpr_params,
            self.trust, self.demand, self.capacity, self.free_flow_time, self.shortest_path)
        self.solution = {'m': m, 'x': x, 'z': z, 'obj': obj, 'x_val': x_val, 'z_val': z_val,
                         'time': obj / sum(self.demand.values())}

    # define function of pulling the arm
    def pull(self, true_trust_rate):
        # # generate a random trust rate with the given mean and std and truncate it to [0,1]
        # trust_rate = np.random.normal(self.trust_rate_mean, self.trust_rate_std)
        # if trust_rate < 0:
        #     trust_rate = 0
        # elif trust_rate > 1:
        #     trust_rate = 1
        # conduct out-of-sample test to get reward
        true_trust = {}
        for i in self.origin_node:
            true_trust[i] = true_trust_rate
        true_time_val = static_model.true_time(
            self.solution['x_val'], self.bpr_params, self.free_flow_time, self.capacity, self.arc_set,
            self.trust, true_trust, self.num_nodes, self.origin_node, self.dest_node, self.demand, self.shortest_path)
        # return reward
        return true_time_val / sum(self.demand.values())


# define class of optimizer for bandit
class GreedyOptimizer:
    def __init__(self, network_structure, true_trust_rate, true_trust_rate_std=0.001, epsilon=0.05):
        self.network_structure = network_structure
        self.true_trust_rate = true_trust_rate
        self.true_trust_rate_std = true_trust_rate_std
        self.epsilon = epsilon
        self.bandit_list = []

    def compute_upper_bound(self):
        # assume that all the people going along the shortest path
        # get the shortest path travel time
        shortest_path_time = 0
        shortest_path_flow = {}
        for key, val in self.network_structure['shortest_path'].items():
            if key not in shortest_path_flow:
                shortest_path_flow[key] = 0
            for o in val:
                shortest_path_flow[key] += self.network_structure['demand'][o]
            shortest_path_time += self.network_structure['free_flow_time'][key] * \
                                  (1 + self.network_structure['bpr_params'][0] *
                                   (shortest_path_flow[key] / self.network_structure['capacity'][key])
                                   ** self.network_structure['bpr_params'][1]) * shortest_path_flow[key]
        return shortest_path_time / sum(self.network_structure['demand'].values())

    def build_bandit(self, trust_list):
        for trust in trust_list:
            trust = max(min(trust, 1), 0)
            cur_bandit = Bandit(self.network_structure, trust)
            cur_bandit.get_solution()
            self.bandit_list.append(cur_bandit)

    def get_optimal_travel_time(self):
        trust = {}
        for i in self.network_structure['origin_node']:
            trust[i] = self.true_trust_rate
        m, x, z, obj, x_val, z_val = static_model.solve_static_model(
            self.network_structure['arc_set'], self.network_structure['origin_node'],
            self.network_structure['dest_node'], self.network_structure['num_nodes'],
            self.network_structure['bpr_params'], trust, self.network_structure['demand'],
            self.network_structure['capacity'], self.network_structure['free_flow_time'],
            self.network_structure['shortest_path'])
        return obj / sum(self.network_structure['demand'].values())

    def run(self, num_iter):
        actions = []
        travel_time = []
        total_time_bandit = np.zeros(len(self.bandit_list))
        pull_time_bandit = np.zeros(len(self.bandit_list))
        opt_time = self.get_optimal_travel_time()
        sample_trust_list = []
        for i in range(num_iter):
            # get trust rate by truncated normal distribution
            trust_rate = np.random.normal(self.true_trust_rate, self.true_trust_rate_std)
            trust_rate = max(min(trust_rate, 1), 0)
            sample_trust_list.append(trust_rate)
            # pull each arm first
            if i < len(self.bandit_list):
                # pull the arm
                actions.append(i)
                cur_time = self.bandit_list[i].pull(trust_rate)
                total_time_bandit[i] += cur_time
                pull_time_bandit[i] += 1
                travel_time.append(cur_time)
            else:
                # get a probability of epsilon to pull a random arm
                if np.random.random() < self.epsilon:
                    # pull a random arm
                    arm = np.random.randint(len(self.bandit_list))
                    actions.append(arm)
                    cur_time = self.bandit_list[arm].pull(trust_rate)
                    total_time_bandit[arm] += cur_time
                    pull_time_bandit[arm] += 1
                    travel_time.append(cur_time)
                else:
                    # pull the arm with the largest average reward
                    arm = np.argmin(total_time_bandit / pull_time_bandit)
                    actions.append(arm)
                    cur_time = self.bandit_list[arm].pull(trust_rate)
                    total_time_bandit[arm] += cur_time
                    pull_time_bandit[arm] += 1
                    travel_time.append(cur_time)
        return actions, travel_time, total_time_bandit, pull_time_bandit, opt_time, sample_trust_list

    def update_ucb_learn_final(self, total_time_bandit, pull_time_bandit, i, beta=1):
        ucb = -total_time_bandit / pull_time_bandit + beta * np.sqrt(2 * np.log(i + 1) / pull_time_bandit)
        return ucb

    def update_ucb_learn_diff(self, time_bandit_diff, pull_time_bandit, i, beta=1):
        ucb = -time_bandit_diff / pull_time_bandit + beta * np.sqrt(2 * np.log(i + 1) / pull_time_bandit)
        return ucb

    def run_ucb(self, num_iter, beta=1, reward='final'):
        max_time = self.compute_upper_bound()
        actions = []
        travel_time = []
        total_time_bandit = np.zeros(len(self.bandit_list))
        time_bandit_diff = np.zeros(len(self.bandit_list))
        time_bandit_iteration = np.zeros(num_iter)
        pull_time_bandit = np.zeros(len(self.bandit_list))
        opt_time = self.get_optimal_travel_time()
        sample_trust_list = []
        for i in range(num_iter):
            # get trust rate by truncated normal distribution
            trust_rate = np.random.normal(self.true_trust_rate, self.true_trust_rate_std)
            trust_rate = max(min(trust_rate, 1), 0)
            sample_trust_list.append(trust_rate)
            # pull each arm first
            if i < len(self.bandit_list):
                # pull the arm
                arm = i
            else:
                # compute the upper confidence bound
                if reward == 'final':
                    ucb = self.update_ucb_learn_final(total_time_bandit, pull_time_bandit, i, beta)
                elif reward == 'learn':
                    ucb = self.update_ucb_learn_diff(time_bandit_diff, pull_time_bandit, i, beta)
                # pull the arm with the largest upper confidence bound
                arm = np.argmax(ucb)
            actions.append(arm)
            cur_time = self.bandit_list[arm].pull(trust_rate)
            total_time_bandit[arm] += cur_time
            time_bandit_diff[arm] += abs(self.bandit_list[arm].solution['time'] - cur_time)
            time_bandit_iteration[i] = abs(self.bandit_list[arm].solution['time'] - cur_time)
            pull_time_bandit[arm] += 1
            travel_time.append(cur_time)
            if (i + 1) % 1000 == 0:
                print(f'At iteration {i + 1}, the cumulative average travel time is '
                      f'{np.sum(travel_time) / (i + 1) * 0.6}, '
                      f'and the regret is {(np.sum(travel_time) / (i + 1) - opt_time) * 0.6}')

        # store all the results in a dictionary
        result = {}
        result['actions'] = actions
        result['travel_time'] = travel_time
        result['total_time_bandit'] = total_time_bandit
        result['pull_time_bandit'] = pull_time_bandit
        result['opt_time'] = opt_time
        result['sample_trust_list'] = sample_trust_list
        result['time_bandit_diff'] = time_bandit_diff
        result['time_bandit_iteration'] = time_bandit_iteration
        return result

    # set the true trust rate
    def set_true_trust_rate(self, true_trust_rate, true_trust_rate_std=0.01):
        self.true_trust_rate = true_trust_rate
        self.true_trust_rate_std = true_trust_rate_std
