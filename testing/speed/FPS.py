import time
import matplotlib.pyplot as plt
import os
from datetime import datetime


class FPSMeter:
    def __init__(self):
        self.frame_times = []

    def record_frame_time(self):
        # Call this method at the end of processing each frame
        self.frame_times.append(time.time())

    def plot_graph(self):
        # Calculate FPS from frame times
        if len(self.frame_times) < 2:
            print("Not enough data to plot FPS.")
            return

        time_diffs = [t2 - t1 for t1, t2 in zip(self.frame_times[:-1], self.frame_times[1:])]
        fps_list = [1 / t if t > 0 else 0 for t in time_diffs]

        plt.figure()
        plt.plot(fps_list)
        plt.xlabel('Frame Number')
        plt.ylabel('FPS')
        plt.title('FPS')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f"{timestamp}.png"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(os.path.join(script_dir, "fps_plots"), plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"FPS plot saved to {plot_path}")
