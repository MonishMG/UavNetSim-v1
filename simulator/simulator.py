import math
import random
import numpy as np
import matplotlib.pyplot as plt
from phy.channel import Channel
from entities.drone import Drone
from entities.obstacle import SphericalObstacle, CubeObstacle
from simulator.metrics import Metrics
from mobility import start_coords
from path_planning.astar import astar
from utils import config
from utils.util_function import grid_map
from allocation.central_controller import CentralController
from visualization.static_drawing import scatter_plot, scatter_plot_with_obstacles


class Simulator:
    """
    Description: simulation environment

    Attributes:
        env: simpy environment
        total_simulation_time: discrete time steps, in nanosecond
        n_drones: number of the drones
        channel_states: a dictionary, used to describe the channel usage
        channel: wireless channel
        metrics: Metrics class, used to record the network performance
        drones: a list, contains all drone instances

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2025/7/8
    """

    def __init__(self,
                 seed,
                 env,
                 channel_states,
                 n_drones,
                 total_simulation_time=config.SIM_TIME):

        self.env = env
        self.seed = seed
        self.total_simulation_time = total_simulation_time  # total simulation time (ns)

        self.n_drones = n_drones  # total number of drones in the simulation
        self.channel_states = channel_states
        self.channel = Channel(self.env)

        self.metrics = Metrics(self)  # use to record the network performance

        # NOTE: if distributed optimization is adopted, remember to comment this to speed up simulation
        # self.central_controller = CentralController(self)

        start_position = start_coords.get_random_start_point_3d(seed)
        # start_position = start_coords.get_customized_start_point_3d()

        self.drones = []
        print('Seed is: ', self.seed)
        for i in range(n_drones):
            if config.HETEROGENEOUS:
                speed = random.randint(5, 60)
            else:
                speed = 10

            print('UAV: ', i, ' initial location is at: ', start_position[i], ' speed is: ', speed)
            drone = Drone(env=env,
                          node_id=i,
                          coords=start_position[i],
                          speed=speed,
                          inbox=self.channel.create_inbox_for_receiver(i),
                          simulator=self)

            self.drones.append(drone)

        # scatter_plot_with_spherical_obstacles(self)
        scatter_plot(self)

        self.env.process(self.show_performance())
        self.env.process(self.show_time())

        # Start time-varying velocity update process (no-op when profile is "constant")
        if config.VELOCITY_PROFILE != "constant":
            self._velocity_rng = random.Random(seed + 77)
            self._velocity_target = 10.0  # initial target for "random" profile
            self.env.process(self._velocity_profile_update())

        # Install simulated attack (simulation-only network degradation)
        if config.ATTACK_ENABLED and config.ATTACK_TYPE != "none":
            self._setup_attack()

    def show_time(self):
        while True:
            print('At time: ', self.env.now / 1e6, ' s.')

            # the simulation process is displayed every 0.5s
            yield self.env.timeout(0.5*1e6)

    def show_performance(self):
        yield self.env.timeout(self.total_simulation_time - 1)

        scatter_plot(self)

        self.metrics.print_metrics()

    # ------------------------------------------------------------------ #
    #  Time-varying UAV velocity                                           #
    # ------------------------------------------------------------------ #

    def _get_target_speed(self, frac):
        """Return the desired speed (m/s) for the given fractional simulation time [0, 1]."""
        profile = config.VELOCITY_PROFILE
        if profile == "step":
            if frac < 0.25:
                return 5.0
            elif frac < 0.50:
                return 10.0
            elif frac < 0.75:
                return 20.0
            else:
                return 30.0
        elif profile == "linear":
            return 5.0 + 25.0 * frac          # 5 m/s → 30 m/s
        elif profile == "sinusoidal":
            return 15.0 + 10.0 * math.sin(2.0 * math.pi * frac * 3.0)
        elif profile == "random":
            # Change target speed with ~1 % probability each update tick (~0.1 s)
            if self._velocity_rng.random() < 0.01:
                self._velocity_target = self._velocity_rng.uniform(5.0, 30.0)
            return self._velocity_target
        return 10.0  # fallback

    @staticmethod
    def _update_drone_speed(drone, new_speed):
        """Update a drone's speed, velocity_mean, and scale its velocity vector."""
        if new_speed <= 0:
            return
        drone.speed = new_speed
        drone.velocity_mean = new_speed          # drives Gauss-Markov convergence
        cur_vel = drone.velocity
        cur_mag = (cur_vel[0] ** 2 + cur_vel[1] ** 2 + cur_vel[2] ** 2) ** 0.5
        if cur_mag > 0:
            scale = new_speed / cur_mag
            drone.velocity = [v * scale for v in cur_vel]

    def _velocity_profile_update(self):
        """SimPy process: update all drone speeds every 0.1 s according to the profile."""
        update_interval_us = 1e5  # 0.1 s in microseconds
        while True:
            yield self.env.timeout(update_interval_us)
            frac = self.env.now / self.total_simulation_time
            new_speed = self._get_target_speed(frac)
            for drone in self.drones:
                self._update_drone_speed(drone, new_speed)

    # ------------------------------------------------------------------ #
    #  Simulated attack (simulation-only network degradation)              #
    # ------------------------------------------------------------------ #

    def _setup_attack(self):
        """
        Wrap channel.unicast_put to simulate network degradation during the
        configured attack window.  This is purely a research/visualisation aid;
        no real exploit or offensive code is involved.
        """
        original_unicast_put = self.channel.unicast_put
        env = self.env
        attack_rng = random.Random(self.seed + 42)

        attack_type = config.ATTACK_TYPE
        start_us = config.ATTACK_START_TIME_S * 1e6
        end_us = config.ATTACK_END_TIME_S * 1e6

        if attack_type in ("packet_drop", "route_disruption"):
            drop_prob = config.ATTACK_DROP_PROBABILITY
            if attack_type == "route_disruption":
                drop_prob = min(drop_prob + 0.2, 0.95)

            def attacked_unicast_put(value, dst_id):
                if start_us <= env.now <= end_us and attack_rng.random() < drop_prob:
                    return  # silently drop — simulated packet loss only
                original_unicast_put(value, dst_id)

        elif attack_type == "delay_injection":
            delay_us = config.ATTACK_DELAY_US

            def attacked_unicast_put(value, dst_id):
                if start_us <= env.now <= end_us:
                    def _delayed():
                        yield env.timeout(delay_us)
                        original_unicast_put(value, dst_id)
                    env.process(_delayed())
                else:
                    original_unicast_put(value, dst_id)

        else:
            return  # unknown attack type — do nothing

        self.channel.unicast_put = attacked_unicast_put
