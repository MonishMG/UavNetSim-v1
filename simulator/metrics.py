import numpy as np
from collections import defaultdict
from openpyxl import load_workbook
from utils import config


class Metrics:
    """
    Tools for statistics of network performance

    1. Packet Delivery Ratio (PDR): is the ratio of number of packets received at the destinations to the number
       of packets sent from the sources
    2. Average end-to-end (E2E) delay: is the time a packet takes to route from a source to its destination through
       the network. It is the time the data packet reaches the destination minus the time the data packet was generated
       in the source node
    3. Routing Load: is calculated as the ratio between the numbers of control Packets transmitted
       to the number of packets actually received. NRL can reflect the average number of control packets required to
       successfully transmit a data packet and reflect the efficiency of the routing protocol
    4. Throughput: it can be defined as a measure of how fast the data is sent from its source to its intended
       destination without loss. In our simulation, each time the destination receives a data packet, the throughput is
       calculated and finally averaged
    5. Hop count: used to record the number of router output ports through which the packet should pass.

    References:
        [1] Rani. N, Sharma. P, Sharma. P., "Performance Comparison of Various Routing Protocols in Different Mobility
            Models," in arXiv preprint arXiv:1209.5507, 2012.
        [2] Gulati M K, Kumar K. "Performance Comparison of Mobile Ad Hoc Network Routing Protocols," International
            Journal of Computer Networks & Communications. vol. 6, no. 2, pp. 127, 2014.

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2025/4/22
    """

    def __init__(self, simulator):
        self.simulator = simulator

        self.control_packet_num = 0

        self.datapacket_generated = set()  # all data packets generated
        self.datapacket_arrived = set()  # all data packets that arrives the destination
        self.datapacket_generated_num = 0

        self.delivery_time = []
        self.deliver_time_dict = defaultdict()

        self.throughput = []
        self.throughput_dict = defaultdict()

        self.hop_cnt = []
        self.hop_cnt_dict = defaultdict()

        self.mac_delay = []

        self.collision_num = 0

        self.throughput_velocity_samples = []

    def calculate_metrics(self, received_packet):
        """Calculate the corresponding metrics when the destination receives a data packet successfully"""
        latency = self.simulator.env.now - received_packet.creation_time  # in us

        self.deliver_time_dict[received_packet.packet_id] = latency
        throughput_bps = received_packet.packet_length / (latency / 1e6)
        self.throughput_dict[received_packet.packet_id] = throughput_bps
        self.hop_cnt_dict[received_packet.packet_id] = received_packet.get_current_ttl()
        self.datapacket_arrived.add(received_packet.packet_id)

        # Collect per-packet throughput vs. velocity sample
        has_src = hasattr(received_packet, 'src_drone')
        has_dst = hasattr(received_packet, 'dst_drone')

        src_drone = received_packet.src_drone if has_src else None
        dst_drone = received_packet.dst_drone if has_dst else None

        src_speed = src_drone.speed if src_drone is not None else None
        dst_speed = dst_drone.speed if dst_drone is not None else None

        avg_src_dst_speed = (src_speed + dst_speed) / 2.0 if (src_speed is not None and dst_speed is not None) else None

        avg_network_speed = np.mean([d.speed for d in self.simulator.drones]) if self.simulator.drones else None

        self.throughput_velocity_samples.append({
            'packet_id': received_packet.packet_id,
            'time_s': self.simulator.env.now / 1e6,
            'throughput_bps': throughput_bps,
            'throughput_kbps': throughput_bps / 1e3,
            'src_id': src_drone.identifier if src_drone is not None else None,
            'dst_id': dst_drone.identifier if dst_drone is not None else None,
            'src_speed_mps': src_speed,
            'dst_speed_mps': dst_speed,
            'avg_src_dst_speed_mps': avg_src_dst_speed,
            'avg_network_speed_mps': avg_network_speed,
            'attack_enabled': config.ATTACK_ENABLED,
            'attack_type': config.ATTACK_TYPE,
        })

    def print_metrics(self):
        # calculate the average end-to-end delay
        e2e_delay = np.mean(list(self.deliver_time_dict.values())) / 1e3

        # calculate the packet delivery ratio
        pdr = len(self.datapacket_arrived) / self.datapacket_generated_num * 100  # in %

        # calculate the throughput
        throughput = np.mean(list(self.throughput_dict.values())) / 1e3

        # calculate the hop count
        hop_cnt = np.mean(list(self.hop_cnt_dict.values()))

        # calculate the routing load
        rl = self.control_packet_num / len(self.datapacket_arrived)

        # channel access delay
        average_mac_delay = np.mean(self.mac_delay)

        print('Totally send: ', self.datapacket_generated_num, ' data packets')
        print('Packet delivery ratio is: ', pdr, '%')
        print('Average end-to-end delay is: ', e2e_delay, 'ms')
        print('Routing load is: ', rl)
        print('Average throughput is: ', throughput, 'Kbps')
        print('Average hop count is: ', hop_cnt)
        print('Collision num is: ', self.collision_num)
        print('Average mac delay is: ', average_mac_delay, 'ms')

    def compute_time_windowed_throughput(self, window_size_s=None):
        """
        Group throughput_velocity_samples into fixed-width time windows and
        compute per-window averages.

        Returns a list of dicts, one per window that contained at least one
        received packet.  Each dict contains:
            window_id, window_start_s, window_end_s, time_s (centre),
            avg_network_velocity_mps, avg_src_dst_velocity_mps,
            throughput_kbps, packet_count, attack_enabled, attack_type.
        """
        if window_size_s is None:
            window_size_s = config.THROUGHPUT_WINDOW_SIZE_S

        samples = self.throughput_velocity_samples
        if not samples:
            return []

        min_t = min(s['time_s'] for s in samples)
        max_t = max(s['time_s'] for s in samples)

        windows = []
        window_id = 0
        t = min_t
        while t <= max_t:
            w_start = t
            w_end = t + window_size_s
            w_samples = [s for s in samples if w_start <= s['time_s'] < w_end]
            if w_samples:
                net_speeds = [s['avg_network_speed_mps'] for s in w_samples
                              if s.get('avg_network_speed_mps') is not None]
                sd_speeds = [s['avg_src_dst_speed_mps'] for s in w_samples
                             if s.get('avg_src_dst_speed_mps') is not None]
                windows.append({
                    'window_id': window_id,
                    'window_start_s': w_start,
                    'window_end_s': w_end,
                    'time_s': (w_start + w_end) / 2.0,
                    'avg_network_velocity_mps': float(np.mean(net_speeds)) if net_speeds else None,
                    'avg_src_dst_velocity_mps': float(np.mean(sd_speeds)) if sd_speeds else None,
                    'throughput_kbps': float(np.mean([s['throughput_kbps'] for s in w_samples])),
                    'packet_count': len(w_samples),
                    'attack_enabled': w_samples[0].get('attack_enabled', False),
                    'attack_type': w_samples[0].get('attack_type', 'none'),
                })
            window_id += 1
            t += window_size_s

        return windows
