import csv
import glob as _glob
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.widgets import Slider, Button, TextBox
from utils import config
import io
import matplotlib.patheffects as path_effects

# Add 3D arrow class definition that handles arrows in 3D view
class Arrow3D(FancyArrowPatch):
    """
    Class for drawing arrows in 3D view
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        # Calculate average z value as depth
        avg_z = np.mean(zs)
        return avg_z
        
    def draw(self, renderer):
        FancyArrowPatch.draw(self, renderer)

class SimulationVisualizer:
    """
    Visualize UAV network simulation process, including movement trajectories and communication status
    """
    
    def __init__(self, simulator, output_dir="vis_results", vis_frame_interval=50000):
        """
        Initialize visualizer
        
        Parameters:
            simulator: simulator instance
            output_dir: output directory
            vis_frame_interval: interval for visualization frames (microseconds)
        """
        self.simulator = simulator
        self.output_dir = output_dir
        
        # Store vis_frame_interval in microseconds
        self.vis_frame_interval = vis_frame_interval
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage structures
        self.drone_positions = {i: [] for i in range(self.simulator.n_drones)}
        self.timestamps = []
        
        # Comm events tracking
        self.comm_events = []  # Store tuples (src_id, dst_id, packet_id, packet_type, timestamp)
        
        # Assign a fixed color to each UAV
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.simulator.n_drones))
        
        # Color mapping for communication types
        self.comm_colors = {
            "DATA": "blue",
            "ACK": "green",
            "HELLO": "orange"
        }
        
        # Setup communication tracking
        self._setup_communication_tracking()
        
        # Reference for interactive elements
        self.interactive_fig = None
        self.interactive_slider = None
        self.frame_times = []
    
    def _setup_communication_tracking(self):
        """Setup tracking for communication events"""
        # Save the original unicast_put method
        original_unicast_put = self.simulator.channel.unicast_put
        
        # Rewrite unicast_put method to track communications
        def tracked_unicast_put(message, dst_drone_id):
            # Call the original method
            result = original_unicast_put(message, dst_drone_id)
            
            # Record communication event
            packet, _, src_drone_id, _, _ = message
            
            # Add packet type differentiation
            packet_id = packet.packet_id
            
            # Identify packet type based on ID range
            if packet_id >= 20000:
                packet_type = "ACK"
            elif packet_id >= 10000:
                packet_type = "HELLO"
            else:
                packet_type = "DATA"
            
            self.track_communication(src_drone_id, dst_drone_id, packet_id, packet_type)
            
            return result
        
        # Replace the method
        self.simulator.channel.unicast_put = tracked_unicast_put
    
    def track_drone_positions(self):
        """
        Record current drone positions
        """
        current_time = self.simulator.env.now / 1e6  # Convert to seconds
        self.timestamps.append(current_time)
        
        for i, drone in enumerate(self.simulator.drones):
            position = drone.coords  # This already contains (x, y, z) coordinates
            self.drone_positions[i].append(position)
    
    def track_communication(self, src_id, dst_id, packet_id, packet_type="DATA"):
        """
        Record communication event
        """
        current_time = self.simulator.env.now / 1e6  # Convert to seconds
        # Record complete communication event information
        self.comm_events.append((src_id, dst_id, packet_id, packet_type, current_time))
    
    def _draw_visualization_frame(self, fig, current_time):
        """
        Draw visualization elements on two side-by-side axes
        
        Parameters:
            fig: matplotlib figure to draw on
            current_time: current simulation time (seconds)
        """
        fig.suptitle(f"UAV Network Simulation at t={int(current_time*1e6)}μs", fontsize=14)
        
        # Create left and right subplots for DATA and ACK only
        ax_data = fig.add_subplot(121, projection='3d')
        ax_ack = fig.add_subplot(122, projection='3d')
        
        # Set titles for subplots
        ax_data.set_title("DATA Packets")
        ax_ack.set_title("ACK Packets")
        
        # Set axis labels and limits for both subplots
        for ax in [ax_data, ax_ack]:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_xlim(0, config.MAP_LENGTH)
            ax.set_ylim(0, config.MAP_WIDTH)
            ax.set_zlim(0, config.MAP_HEIGHT)
            ax.grid(True)
        
        # Get drone positions at current time
        drone_positions = self._get_drone_positions(current_time)
        
        # Draw drones on both subplots
        for ax in [ax_data, ax_ack]:
            self._draw_drones(ax, drone_positions)
        
        # Draw communication links
        display_window = self.vis_frame_interval / 1e6  # Convert to seconds
        recent_comms = [e for e in self.comm_events 
                      if current_time - display_window <= e[4] <= current_time]
        
        # Get only the latest communication events for each src-dst pair
        latest_data_comms = self._get_latest_comms(recent_comms, "DATA")
        latest_ack_comms = self._get_latest_comms(recent_comms, "ACK")
        
        # Draw DATA packet links on left subplot
        self._draw_data_links(ax_data, latest_data_comms, drone_positions)
        
        # Draw ACK packet links on right subplot
        self._draw_ack_links(ax_ack, latest_ack_comms, drone_positions)
        
        # Add legends
        data_legend = [Line2D([0], [0], color=self.comm_colors["DATA"], lw=2, label="DATA Packets")]
        ax_data.legend(handles=data_legend, loc='upper right')
        
        ack_legend = [Line2D([0], [0], color=self.comm_colors["ACK"], lw=2, label="ACK Packets")]
        ax_ack.legend(handles=ack_legend, loc='upper right')

    def _get_latest_comms(self, comms, packet_type):
        """
        Get only the latest communication for each src-dst pair
        
        Parameters:
            comms: List of communication events
            packet_type: Type of packet (DATA, ACK, HELLO)
            
        Returns:
            List of latest communication events for each src-dst pair
        """
        # Filter by packet type
        type_comms = [e for e in comms if e[3] == packet_type]
        
        # Dictionary to store latest comm for each src-dst pair
        latest_comms_dict = {}
        
        # For each src-dst pair, keep only the comm with the latest timestamp
        for comm in type_comms:
            src_id, dst_id = comm[0], comm[1]
            pair_key = (src_id, dst_id)
            
            # If this is the first comm for this pair, or has a later timestamp
            if pair_key not in latest_comms_dict or comm[4] > latest_comms_dict[pair_key][4]:
                latest_comms_dict[pair_key] = comm
        
        # Return the values (latest comms)
        return list(latest_comms_dict.values())

    def create_animations(self):
        """Create GIF animation of the simulation"""
        import io
        
        if not self.timestamps:
            print("No timestamps available for animation")
            return
        
        try:
            print("Creating animation GIF...")
            animation_frames = []
            
            # Calculate frames based on vis_frame_interval
            min_time = min(self.timestamps)
            max_time = max(self.timestamps)
            frame_interval_sec = self.vis_frame_interval / 1e6  # Convert microseconds to seconds
            
            # Create frame times at regular intervals based on vis_frame_interval
            self.frame_times = []
            current_time = min_time
            while current_time <= max_time:
                self.frame_times.append(current_time)
                current_time += frame_interval_sec
            
            n_frames = len(self.frame_times)
            print(f"Generating {n_frames} frames with interval of {frame_interval_sec} seconds")
            
            for i, time_point in enumerate(self.frame_times):
                print(f"Generating frame {i+1}/{n_frames}", end="\r")
                
                # Create a new figure for this frame
                fig = plt.figure(figsize=(18, 6))
                
                # Draw visualization elements
                self._draw_visualization_frame(fig, time_point)
                
                # Save the figure to a BytesIO buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close(fig)
                
                # Reset buffer position and open image
                buf.seek(0)
                img = Image.open(buf)
                # Convert to RGB mode to ensure compatibility
                img = img.convert('RGB')
                # Create a copy of the image to ensure it's fully loaded
                img_copy = img.copy()
                animation_frames.append(img_copy)
                buf.close()
            
            print("\nSaving animation...")
            
            # Save the animation
            animation_file = os.path.join(self.output_dir, "uav_network_simulation.gif")
            if animation_frames:
                # Save with explicit parameters
                animation_frames[0].save(
                    animation_file,
                    format='GIF',
                    save_all=True,
                    append_images=animation_frames[1:],
                    duration=50,  # ms per frame
                    loop=1,  # Loop indefinitely
                    optimize=True,
                    quality=70,    # Reduce quality slightly (0-100)
                    # Reduce colors if needed
                    colors=128     # Maximum number of colors
                )
                print(f"Animation saved to {animation_file}")
            else:
                print("No frames were generated for the animation")
                
        except Exception as e:
            print(f"Error creating animation: {e}")
            print("Continuing with interactive visualization...")
    
    def run_visualization(self):
        """
        Run visualization process
        """
        # Use vis_frame_interval directly (it's already in microseconds)
        tracking_interval_us = self.vis_frame_interval
        
        # Start tracking drone positions
        def track_positions():
            while True:
                self.track_drone_positions()
                yield self.simulator.env.timeout(tracking_interval_us)
        
        # Register tracking process
        self.simulator.env.process(track_positions())
    
    def finalize(self):
        """
        Finalize visualization, generate frames and animations
        """
        print("Finalizing visualization...")
        
        # Create animation
        self.create_animations()
        
        # Create interactive visualization
        self.create_interactive_visualization()

        # Create throughput vs. UAV velocity plot (original per-packet version)
        self.create_throughput_velocity_plot()

        # New analysis plots and CSV exports
        self._save_throughput_velocity_csvs()
        self.create_throughput_vs_velocity_time_windowed_plot()
        self.create_throughput_and_velocity_over_time_plot()
        self.create_attack_impact_throughput_vs_velocity_plot()

        # Summary print
        print("\n--- Throughput / Velocity Analysis Summary ---")
        print(f"  Velocity profile : {config.VELOCITY_PROFILE}")
        print(f"  Dynamic velocity : {'enabled' if config.VELOCITY_PROFILE != 'constant' else 'disabled (constant 10 m/s)'}")
        print(f"  Attack simulation: {'enabled  (type: ' + config.ATTACK_TYPE + ')' if config.ATTACK_ENABLED else 'disabled'}")
        print(f"  Output directory : {self.output_dir}")
        print("  New files saved  :")
        for fname in [
            "throughput_velocity_raw_samples.csv",
            "throughput_velocity_time_windowed.csv",
            "throughput_vs_velocity_time_windowed.png",
            "throughput_and_velocity_over_time.png",
            "attack_impact_throughput_vs_velocity.png",
        ]:
            print(f"    {os.path.join(self.output_dir, fname)}")
        print("----------------------------------------------\n")
        
        print("Visualization complete. Output saved to:", self.output_dir)

    def create_throughput_velocity_plot(self):
        """Generate a scatter plot of throughput vs. UAV velocity and save raw data as CSV."""
        samples = []
        if hasattr(self.simulator, 'metrics'):
            samples = self.simulator.metrics.throughput_velocity_samples

        if not samples:
            print("No throughput-velocity samples available; skipping throughput_vs_uav_velocity plot.")
            return

        # Filter samples that have a valid avg_src_dst_speed_mps
        valid = [s for s in samples if s.get('avg_src_dst_speed_mps') is not None]
        if not valid:
            print("No valid speed data in throughput-velocity samples; skipping plot.")
            return

        velocities = np.array([s['avg_src_dst_speed_mps'] for s in valid])
        throughputs = np.array([s['throughput_kbps'] for s in valid])

        # Compute grouped mean throughput per (rounded) velocity
        unique_velocities = sorted(set(velocities))
        mean_throughputs = []
        for v in unique_velocities:
            mask = velocities == v
            mean_throughputs.append(np.mean(throughputs[mask]))

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(velocities, throughputs, alpha=0.5, s=30, label='Per-packet samples', color='steelblue')
        ax.plot(unique_velocities, mean_throughputs, color='darkorange', linewidth=2,
                marker='o', markersize=5, label='Mean throughput per velocity')

        ax.set_title('Throughput vs UAV Velocity', fontsize=14)
        ax.set_xlabel('UAV Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Throughput (Kbps)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, 'throughput_vs_uav_velocity.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved throughput vs. UAV velocity plot to: {plot_path}")

        # Save raw data as CSV
        csv_path = os.path.join(self.output_dir, 'throughput_vs_uav_velocity.csv')
        fieldnames = [
            'packet_id', 'time_s', 'throughput_bps', 'throughput_kbps',
            'src_id', 'dst_id', 'src_speed_mps', 'dst_speed_mps',
            'avg_src_dst_speed_mps', 'avg_network_speed_mps',
        ]
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                writer.writerow({k: s.get(k, '') for k in fieldnames})
        print(f"Saved throughput-velocity data to: {csv_path}")

    # ------------------------------------------------------------------ #
    #  New analysis: time-windowed throughput / velocity + attack plots    #
    # ------------------------------------------------------------------ #

    def _save_throughput_velocity_csvs(self):
        """Save raw samples and time-windowed aggregates as CSV files."""
        samples = []
        if hasattr(self.simulator, 'metrics'):
            samples = self.simulator.metrics.throughput_velocity_samples

        # --- raw samples ---
        raw_path = os.path.join(self.output_dir, 'throughput_velocity_raw_samples.csv')
        raw_fields = [
            'time_s', 'avg_network_speed_mps', 'avg_src_dst_speed_mps',
            'throughput_bps', 'throughput_kbps',
            'packet_id', 'src_id', 'dst_id',
            'attack_enabled', 'attack_type',
        ]
        with open(raw_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=raw_fields)
            writer.writeheader()
            for s in samples:
                writer.writerow({k: s.get(k, '') for k in raw_fields})

        # --- windowed aggregates ---
        windowed = []
        if hasattr(self.simulator, 'metrics'):
            windowed = self.simulator.metrics.compute_time_windowed_throughput()

        win_path = os.path.join(self.output_dir, 'throughput_velocity_time_windowed.csv')
        win_fields = [
            'window_id', 'window_start_s', 'window_end_s', 'time_s',
            'avg_network_velocity_mps', 'avg_src_dst_velocity_mps',
            'throughput_kbps', 'packet_count',
            'attack_enabled', 'attack_type',
        ]
        with open(win_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=win_fields)
            writer.writeheader()
            for w in windowed:
                writer.writerow({k: w.get(k, '') for k in win_fields})

    def create_throughput_vs_velocity_time_windowed_plot(self):
        """
        Plot A: Throughput (Kbps) vs. Average UAV Velocity (m/s) using
        time-windowed samples as scatter points, plus a trend line grouped
        by velocity bins.

        Output: throughput_vs_velocity_time_windowed.png
        """
        windowed = []
        if hasattr(self.simulator, 'metrics'):
            windowed = self.simulator.metrics.compute_time_windowed_throughput()

        if not windowed:
            print("No windowed data; skipping throughput_vs_velocity_time_windowed.png")
            return

        valid = [w for w in windowed if w.get('avg_network_velocity_mps') is not None]
        if not valid:
            print("No velocity data in windowed samples; skipping plot.")
            return

        velocities = np.array([w['avg_network_velocity_mps'] for w in valid])
        throughputs = np.array([w['throughput_kbps'] for w in valid])

        # Bin velocities into 2 m/s buckets for the trend line
        bin_size = 2.0
        v_min, v_max = velocities.min(), velocities.max()
        bins = np.arange(v_min, v_max + bin_size, bin_size)
        bin_centers, bin_means = [], []
        for b in bins:
            mask = (velocities >= b) & (velocities < b + bin_size)
            if mask.sum() > 0:
                bin_centers.append(b + bin_size / 2)
                bin_means.append(throughputs[mask].mean())

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(velocities, throughputs, alpha=0.6, s=40,
                   label='1-s window samples', color='steelblue')
        if bin_centers:
            ax.plot(bin_centers, bin_means, color='darkorange', linewidth=2,
                    marker='o', markersize=5, label='Mean per 2 m/s velocity bin')

        ax.set_title('Throughput vs UAV Velocity\n(Time-Varying Mobility)', fontsize=13)
        ax.set_xlabel('Average UAV Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Average Throughput (Kbps)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'throughput_vs_velocity_time_windowed.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def create_throughput_and_velocity_over_time_plot(self):
        """
        Plot B: Throughput (Kbps) and Average UAV Velocity (m/s) both over
        simulation time, sharing the x-axis.

        Output: throughput_and_velocity_over_time.png
        """
        windowed = []
        if hasattr(self.simulator, 'metrics'):
            windowed = self.simulator.metrics.compute_time_windowed_throughput()

        if not windowed:
            print("No windowed data; skipping throughput_and_velocity_over_time.png")
            return

        times = np.array([w['time_s'] for w in windowed])
        throughputs = np.array([w['throughput_kbps'] for w in windowed])
        velocities = np.array([
            w['avg_network_velocity_mps'] if w.get('avg_network_velocity_mps') is not None else float('nan')
            for w in windowed
        ])

        fig, ax1 = plt.subplots(figsize=(11, 5))
        color_tp = 'steelblue'
        color_vel = 'darkorange'

        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Throughput (Kbps)', color=color_tp, fontsize=12)
        ax1.plot(times, throughputs, color=color_tp, linewidth=1.5,
                 label='Throughput (Kbps)')
        ax1.tick_params(axis='y', labelcolor=color_tp)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Average UAV Velocity (m/s)', color=color_vel, fontsize=12)
        ax2.plot(times, velocities, color=color_vel, linewidth=1.5,
                 linestyle='--', label='Avg Velocity (m/s)')
        ax2.tick_params(axis='y', labelcolor=color_vel)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

        ax1.set_title('Throughput and UAV Velocity Over Time', fontsize=13)
        ax1.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'throughput_and_velocity_over_time.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def create_attack_impact_throughput_vs_velocity_plot(self):
        """
        Plot C: Baseline vs. attack-condition throughput vs. UAV velocity.

        * Uses the current simulation's windowed data.
        * If an existing CSV from the complementary run (attack vs. no-attack)
          is present in the output directory it will be overlaid automatically.

        Output: attack_impact_throughput_vs_velocity.png
        """
        windowed = []
        if hasattr(self.simulator, 'metrics'):
            windowed = self.simulator.metrics.compute_time_windowed_throughput()

        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_anything = False

        def _plot_series(w_list, label, color, marker='o'):
            valid = [w for w in w_list if w.get('avg_network_velocity_mps') is not None]
            if not valid:
                return
            velocities = np.array([w['avg_network_velocity_mps'] for w in valid])
            throughputs = np.array([w['throughput_kbps'] for w in valid])
            ax.scatter(velocities, throughputs, alpha=0.35, s=25, color=color)
            # Trend line via 2 m/s bins
            bin_size = 2.0
            v_min, v_max = velocities.min(), velocities.max()
            bins = np.arange(v_min, v_max + bin_size, bin_size)
            bx, by = [], []
            for b in bins:
                mask = (velocities >= b) & (velocities < b + bin_size)
                if mask.sum() > 0:
                    bx.append(b + bin_size / 2)
                    by.append(throughputs[mask].mean())
            if bx:
                ax.plot(bx, by, color=color, linewidth=2, marker=marker,
                        markersize=5, label=label)

        # --- current run ---
        is_attack = config.ATTACK_ENABLED and config.ATTACK_TYPE != "none"
        cur_label = (f"Attack ({config.ATTACK_TYPE})" if is_attack else "Baseline")
        cur_color = 'crimson' if is_attack else 'steelblue'
        if windowed:
            _plot_series(windowed, cur_label, cur_color)
            plotted_anything = True

        # --- look for a complementary CSV from a previous run ---
        csv_path = os.path.join(self.output_dir, 'throughput_velocity_time_windowed.csv')
        # The CSV we just wrote belongs to the current run; look for the *other* one
        # (if the user saves outputs with different names via ATTACK_ENABLED toggle).
        # Convention: a run saves <attack_type>_throughput_velocity_time_windowed.csv
        other_csv = None
        if is_attack:
            candidate = os.path.join(self.output_dir,
                                     'baseline_throughput_velocity_time_windowed.csv')
        else:
            # Try to find any attack CSV from a previous run
            candidates = _glob.glob(
                os.path.join(self.output_dir, '*_throughput_velocity_time_windowed.csv'))
            candidate = candidates[0] if candidates else None

        if candidate and os.path.isfile(candidate):
            other_csv = candidate

        if other_csv:
            try:
                other_windowed = []
                with open(other_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            w = {
                                'avg_network_velocity_mps': float(row['avg_network_velocity_mps'])
                                if row.get('avg_network_velocity_mps') else None,
                                'throughput_kbps': float(row['throughput_kbps'])
                                if row.get('throughput_kbps') else None,
                                'attack_enabled': row.get('attack_enabled', 'False') == 'True',
                                'attack_type': row.get('attack_type', 'none'),
                            }
                            if w['throughput_kbps'] is not None:
                                other_windowed.append(w)
                        except (ValueError, KeyError):
                            pass
                if other_windowed:
                    other_is_attack = other_windowed[0].get('attack_enabled', False)
                    other_label = (
                        f"Attack ({other_windowed[0].get('attack_type', '?')})"
                        if other_is_attack else "Baseline"
                    )
                    other_color = 'crimson' if other_is_attack else 'steelblue'
                    _plot_series(other_windowed, other_label, other_color,
                                 marker='^')
                    plotted_anything = True
            except Exception as exc:
                print(f"Could not load complementary CSV {other_csv}: {exc}")

        if not plotted_anything:
            print("No data for attack_impact_throughput_vs_velocity.png; skipping.")
            plt.close(fig)
            return

        title = ("Attack Impact: Throughput vs UAV Velocity\n"
                 "(Simulated network degradation — research only)")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Average UAV Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Average Throughput (Kbps)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        path = os.path.join(self.output_dir, 'attack_impact_throughput_vs_velocity.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

        # Also save a named copy so future complementary runs can load it
        named_copy = os.path.join(
            self.output_dir,
            ('baseline' if not is_attack else config.ATTACK_TYPE)
            + '_throughput_velocity_time_windowed.csv'
        )
        src_csv = os.path.join(self.output_dir, 'throughput_velocity_time_windowed.csv')
        if os.path.isfile(src_csv) and not os.path.isfile(named_copy):
            shutil.copy2(src_csv, named_copy)

    def create_interactive_visualization(self):
        """Create an interactive visualization with a slider for time navigation"""
        if not self.timestamps:
            print("No timestamps available for interactive visualization")
            return
        
        print("Creating interactive visualization...")
        
        # Make sure frame_times is populated
        if not self.frame_times:
            min_time = min(self.timestamps)
            max_time = max(self.timestamps)
            frame_interval_sec = self.vis_frame_interval / 1e6
            
            current_time = min_time
            while current_time <= max_time:
                self.frame_times.append(current_time)
                current_time += frame_interval_sec
        
        # Convert frame times to microseconds for the slider
        frame_times_us = [t * 1e6 for t in self.frame_times]
        
        # Create figure with fixed subplots - this is key to solving the error
        fig = plt.figure(figsize=(15, 7))
        plt.subplots_adjust(bottom=0.15)  # Make room for controls
        
        # Create the subplots once and keep them
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.2)
        ax_data = fig.add_subplot(gs[0, 0], projection='3d')
        ax_ack = fig.add_subplot(gs[0, 1], projection='3d')
        
        # Add slider axes
        slider_ax = plt.axes([0.2, 0.05, 0.65, 0.03])
        time_slider = Slider(
            slider_ax, 'Time (μs)', 
            min(frame_times_us), max(frame_times_us),
            valinit=frame_times_us[0],
            valstep=frame_times_us  # Discrete steps based on frame times
        )
        
        # Add text box and button for direct time input
        text_ax = plt.axes([0.2, 0.01, 0.2, 0.03])
        time_text = TextBox(text_ax, 'Go to time (μs): ', initial='')
        
        button_ax = plt.axes([0.45, 0.01, 0.1, 0.03])
        goto_button = Button(button_ax, 'Go')
        
        def update_plot(current_time):
            # Clear existing content on axes
            ax_data.clear()
            ax_ack.clear()
            
            # Set titles
            ax_data.set_title("DATA Packets")
            ax_ack.set_title("ACK Packets")
            
            # Update main figure title
            fig.suptitle(f"UAV Network Simulation at t={int(current_time*1e6)}μs", fontsize=14)
            
            # Set axis properties for both subplots
            for ax in [ax_data, ax_ack]:
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_xlim(0, config.MAP_LENGTH)
                ax.set_ylim(0, config.MAP_WIDTH)
                ax.set_zlim(0, config.MAP_HEIGHT)
                ax.grid(True)
            
            # Get drone positions
            drone_positions = self._get_drone_positions(current_time)
            
            # Draw drones on both subplots
            self._draw_drones(ax_data, drone_positions)
            self._draw_drones(ax_ack, drone_positions)
            
            # Get recent communications
            display_window = self.vis_frame_interval / 1e6
            recent_comms = [e for e in self.comm_events 
                            if current_time - display_window <= e[4] <= current_time]
            
            # Filter by packet type and get latest only
            latest_data_comms = self._get_latest_comms(recent_comms, "DATA")
            latest_ack_comms = self._get_latest_comms(recent_comms, "ACK")
            
            # Draw communication links
            self._draw_data_links(ax_data, latest_data_comms, drone_positions)
            self._draw_ack_links(ax_ack, latest_ack_comms, drone_positions)
            
            # Add legends
            data_legend = [Line2D([0], [0], color=self.comm_colors["DATA"], lw=2, label="DATA Packets")]
            ax_data.legend(handles=data_legend, loc='upper right')
            
            ack_legend = [Line2D([0], [0], color=self.comm_colors["ACK"], lw=2, label="ACK Packets")]
            ax_ack.legend(handles=ack_legend, loc='upper right')
        
        def update(val):
            try:
                # Get current time from slider
                current_time_us = time_slider.val
                current_time = current_time_us / 1e6  # Convert to seconds
                
                # Update plot with new time
                update_plot(current_time)
                
                # Redraw
                fig.canvas.draw_idle()
            except Exception as e:
                print(f"Error updating plot: {e}")
        
        def goto_time(event):
            try:
                # Get time from text box
                time_us = float(time_text.text)
                
                # Find the closest frame time
                closest_time_us = min(frame_times_us, key=lambda x: abs(x - time_us))
                
                # Update slider to trigger update
                time_slider.set_val(closest_time_us)
                
                # Update textbox to show actual time used
                time_text.set_val(str(int(closest_time_us)))
            except ValueError:
                print("Invalid time format. Please enter a number.")
            except Exception as e:
                print(f"Error going to time: {e}")
        
        # Connect the update function to the slider
        time_slider.on_changed(update)
        
        # Connect the goto function to the button
        goto_button.on_clicked(goto_time)
        
        # Initial plot
        update_plot(self.frame_times[0])
        
        # Save reference to interactive elements
        self.interactive_fig = fig
        self.interactive_slider = time_slider
        
        # Show the interactive visualization
        plt.show()
        
        print("Interactive visualization created. Close the plot window to continue.")

    def _get_drone_positions(self, current_time):
        """Get drone positions at a specific time"""
        drone_positions = {}
        for drone_id in range(len(self.drone_positions)):
            positions = self.drone_positions[drone_id]
            timestamps = self.timestamps
            
            if positions and timestamps:
                # Find closest timestamp
                closest_idx = min(range(len(timestamps)), 
                               key=lambda i: abs(timestamps[i] - current_time))
                
                # Get position at closest timestamp
                if 0 <= closest_idx < len(positions):
                    drone_positions[drone_id] = positions[closest_idx]
        return drone_positions

    def _draw_drones(self, ax, drone_positions):
        """Draw drones on the given axis with embedded ID numbers"""
        for drone_id, position in drone_positions.items():
            color = self.colors[drone_id]
            
            # Use smaller marker size for drone representation
            ax.scatter(position[0], position[1], position[2], 
                    color=color, s=150, alpha=0.7, edgecolors='black')
            
            # Add ID text with outline for better visibility
            # Set high zorder to ensure text appears above other elements
            text = ax.text(position[0], position[1], position[2], 
                     f"{drone_id}", ha='center', va='center', 
                     color='white', fontweight='bold', fontsize=10,
                     path_effects=[path_effects.withStroke(linewidth=2, foreground='black')],
                     zorder=100)  # Ensure text is displayed on top layer

    def _draw_data_links(self, ax, data_comms, drone_positions):
        """Draw DATA packet links on the given axis with smaller packet ID boxes"""
        for src_id, dst_id, packet_id, _, _ in data_comms:
            if src_id in drone_positions and dst_id in drone_positions:
                start_pos = drone_positions[src_id]
                end_pos = drone_positions[dst_id]
                
                # Draw an arrow for DATA packet
                arrow = Arrow3D([start_pos[0], end_pos[0]], 
                              [start_pos[1], end_pos[1]], 
                              [start_pos[2], end_pos[2]],
                              mutation_scale=15, 
                              lw=2, 
                              arrowstyle="-|>", 
                              color=self.comm_colors["DATA"])
                
                ax.add_artist(arrow)
                
                # Add more visible packet ID at midpoint
                mid_x, mid_y, mid_z = [(start_pos[i] + end_pos[i]) / 2 for i in range(3)]
                
                # Draw a smaller, more compact background for the packet ID
                ax.text(mid_x, mid_y, mid_z, str(packet_id), 
                      ha='center', va='center', fontsize=7, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', 
                                alpha=0.8, edgecolor=self.comm_colors["DATA"], linewidth=1.5),
                      zorder=99)  # Display above other elements but below drone IDs

    def _draw_ack_links(self, ax, ack_comms, drone_positions):
        """Draw ACK packet links on the given axis with smaller packet ID boxes"""
        for src_id, dst_id, packet_id, _, _ in ack_comms:
            if src_id in drone_positions and dst_id in drone_positions:
                start_pos = drone_positions[src_id]
                end_pos = drone_positions[dst_id]
                
                # Draw a straight line for ACK packet
                ax.plot([start_pos[0], end_pos[0]], 
                       [start_pos[1], end_pos[1]], 
                       [start_pos[2], end_pos[2]],
                       color=self.comm_colors["ACK"], 
                       linewidth=2)
                
                # Add more visible packet ID at midpoint
                mid_x, mid_y, mid_z = [(start_pos[i] + end_pos[i]) / 2 for i in range(3)]
                
                # Draw a smaller, more compact background for the ACK packet ID
                ax.text(mid_x, mid_y, mid_z, str(packet_id), 
                       ha='center', va='center', fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', 
                                alpha=0.8, edgecolor=self.comm_colors["ACK"], linewidth=1.5),
                       zorder=99)  # Display above other elements but below drone IDs