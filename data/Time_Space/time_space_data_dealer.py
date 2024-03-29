#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import numpy as np

data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(data_dir)

from DataDealer import DataDealer

class TimeSpaceDataDealer(DataDealer):

    def __init__(self, hyperparams):
        super(TimeSpaceDataDealer, self).__init__(hyperparams)
        return

    # generate a set of fake data given a specified random seed
    def generate_data(self, random_seed=0):
        self.rng = np.random.RandomState(random_seed)
        # time information
        self.time_info = {}  # a dict about time information with key as each time key and value as corresponding time indice
        self.time_seq = []  # a list with time keys as elements, indexed as the time indice
        self.T = int((self._hyperparams['ts_end'] - self._hyperparams['ts_start']) / self._hyperparams['delta_ts'] + 1)
        for t in range(self.T):
            time = str(self._hyperparams['ts_start'] + t)
            self.time_info[t] = time
            self.time_seq.append(time)


        # location infomation
        self.location_info = {}
        self.location_seq = []

        rectangular_size = self._hyperparams['rectangular_size']
        locations_inner = []
        locations_outer = []
        i_grid = 0
        for grid_x in range(rectangular_size[0]):
            for grid_y in range(rectangular_size[1]):
                grid = {'id': str(i_grid), 'coordinate': (grid_x, grid_y)}
                if rectangular_size[0]/4 <= grid_x < 3*rectangular_size[0]/4 and rectangular_size[1]/4 <= grid_y < 3*rectangular_size[1]/4:
                    grid['region'] = 'inner'
                    locations_inner.append(grid)
                else:
                    grid['region'] = 'outer'
                    locations_outer.append(grid)
                #self.location_info[str(i_grid)] = grid

                self.location_info[str(i_grid)] = (grid_x, grid_y)
                self.location_seq.append(str(i_grid))
                i_grid += 1
        self.nLocation = i_grid

        ## data

        # initial resources
        self.init_resource_info = {}
        nR0 = self.rng.randint(self._hyperparams['resource_max'], size=self.nLocation)
        for i_grid in range(self.nLocation):
            #self.location_info[str(i_grid)]['nR_t0'] = nR0[i_grid]
            self.init_resource_info[str(i_grid)] = nR0[i_grid]

        # generate a pool of locations, with number of each location proportional to its weight
        def generate_locations_pool(locations_inner, locations_outer, weight_inner, weight_outer):
            locations_pool = []
            for location in locations_inner:
                locations_pool.extend([location]*weight_inner)
            for location in locations_outer:
                locations_pool.extend([location]*weight_outer)
            return locations_pool

        # tasks
        self.tasks_info = []

        n0T = np.zeros(self.T)
        n0T_ = self._hyperparams['task_number_mean'] + self._hyperparams['task_number_std']*self.rng.randn(1, self.T).flatten()
        for i, n in enumerate(n0T_):
            n0T[i] = max(int(n), 0)

        for t in range(self.T):
            tasks_t = []

            # order assign weights: tuple, with the first element for start and the second element for destination
            w = 2
            if t in range(6, 10):
                task_weight_inner = (1, w)
                task_weight_outer = (w, 1)
            elif t in range(18, 22):
                task_weight_inner = (w, 1)
                task_weight_outer = (1, w)
            else:
                task_weight_inner = (1, 1)
                task_weight_outer = (1, 1)

            # start
            start_locations_pool = generate_locations_pool(locations_inner, locations_outer, task_weight_inner[0],
                                                           task_weight_outer[0])

            # destination
            dest_locations_pool = generate_locations_pool(locations_inner, locations_outer, task_weight_inner[1],
                                                          task_weight_outer[1])

            # generate each task
            for i in range(int(n0T[t])):
                while True:
                    i_start = self.rng.choice(len(start_locations_pool), 1)[0]
                    i_dest = self.rng.choice(len(dest_locations_pool), 1)[0]

                    start = start_locations_pool[i_start]['id']
                    destination = dest_locations_pool[i_dest]['id']

                    if (start != destination):

                        distance = self.cal_location_distance(start, destination)
                        duration = self.cal_duration(distance, self._hyperparams['task_speed'])
                        income = distance*(max(self._hyperparams['task_income_mean'] + self._hyperparams['task_income_std'] * self.rng.randn(), self._hyperparams['rep_cost'] * 1.1))
                        tasks_t.append({'time': self.time_info[t], 'start': start, 'destination': destination, 'distance': distance, 'duration': duration, 'income': income})
                        break
                    '''
                    else:
                        print("start and dest are the same, retry sampling")
                    '''
            self.tasks_info.append({self.time_info[t]: tasks_t})

        # possible repositions
        self.repositions = []
        for i_start in range(self.nLocation):
            for i_destination in range(self.nLocation):
                start = self.location_seq[i_start]
                destination = self.location_seq[i_destination]
                distance = self.cal_location_distance(start, destination)
                duration = self.cal_duration(distance, self._hyperparams['rep_speed'])
                cost = distance*self._hyperparams['rep_cost']

                self.repositions.append({'start': start, 'destination': destination, 'distance': distance, 'duration': duration, 'cost': cost})

        return

    def dump_data(self, ex_data_dir):
        # time & location dictionary
        time_space_info = {"time_detail": self.time_info, "time_seq": self.time_seq, "location_detail": self.location_info, "location_seq": self.location_seq}
        with open(ex_data_dir + 'time_space_info.json', 'w') as f:
            json.dump(time_space_info, f, indent=4, ensure_ascii=False)  # , encoding="utf-8"
        f.close()

        # init resources
        with open(ex_data_dir + 'init_resource.json', 'w') as f:
            json.dump(self.init_resource_info, f, indent=4, ensure_ascii=False)  # , encoding="utf-8"
        f.close()

        # task orders
        with open(ex_data_dir + 'tasks.json', 'w') as f:
            json.dump(self.tasks_info, f, indent=4, ensure_ascii=False)  # , encoding="utf-8"
        f.close()

        # possible relocations
        with open(ex_data_dir + 'repositions.json', 'w') as f:
            json.dump(self.repositions, f, indent=4, ensure_ascii=False)  # , encoding="utf-8"
        f.close()

        return

    def load_data(self, ex_data_dir):
        with open(ex_data_dir + 'time_space_info.json', 'r') as f:
            #time_space_info = json.load(f, encoding="utf-8")
            time_space_info = yaml.load(f)

        with open(ex_data_dir + 'init_resource.json', 'r') as f:
            #init_resource = json.load(f, encoding="utf-8")
            init_resource=yaml.load(f)

        with open(ex_data_dir + 'tasks.json', 'r') as f:
            #tasks = json.load(f, encoding="utf-8")
            tasks = yaml.load(f)

        with open(ex_data_dir + 'repositions.json', 'r') as f:
            #relocations = json.load(f, encoding="utf-8")
            repositions = yaml.load(f)

        return time_space_info, init_resource, tasks, repositions

    def next_batch(self, batch_size):
        return


    def cal_location_distance(self, location_id1, location_id2):
        location1 = self.location_info[location_id1]
        location2 = self.location_info[location_id2]

        route_vec = np.array(location2) - np.array(location1)

        if self._hyperparams['location_shape'] == "rectangular":
            if self._hyperparams['distance_cal_type'] == 'euc':
                return np.linalg.norm(route_vec, 2)
            else:
                return np.linalg.norm(route_vec, 1)

        return -1.0

    def cal_duration(self, distance, speed):
        return np.round(distance/speed)


