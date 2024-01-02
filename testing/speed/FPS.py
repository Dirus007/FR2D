import time
import matplotlib.pyplot as plt
import os


class FPSMeter:
    def __init__(self):
        self.frame_times = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            end_time = time.time()
            self.frame_times.append(end_time - self.start_time)
            self.start_time = None

    def plot_fps(self, plot_filename='fps_plot.png'):
        fps_list = [1 / t if t > 0 else 0 for t in self.frame_times]

        plt.figure()
        plt.plot(fps_list)
        plt.xlabel('Frame Number')
        plt.ylabel('FPS')
        plt.title('FPS Over Time')
        plot_filename = f""
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"FPS plot saved to {plot_path}")
