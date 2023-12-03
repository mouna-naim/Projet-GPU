
#!/usr/bin/env python

import time
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from matplotlib.widgets import Button

np.set_printoptions(edgeitems=10, linewidth=512, formatter=dict(float=lambda x: "%.3g" % x))

def read_mesh(filename):
        mesh = np.genfromtxt(filename, delimiter=',')
        return mesh

def plot_single_mesh(args, filename):
        mesh = read_mesh(filename)
        fig, ax = plt.subplots()
        im = ax.imshow(mesh)
        if args.colorbar:
                cbar = ax.figure.colorbar(im, format="{x:.2e}", ax=ax)
        ax.set_title(filename)
        fig.tight_layout()
        if args.delay <= 0:
                plt.show()
        else:
                plt.pause(args.delay)
                plt.close()

def plot_mesh_list(args, filename_list):
        class Mesh_sequence:
                def __init__(self, filename_list, delay):
                        self.filename_list = filename_list
                        self.i = 0
                        self.meshes = dict()

                        mesh = self.get_mesh()

                        self.fig, self.ax = plt.subplots()
                        self.im = self.ax.imshow(mesh)
                        if args.colorbar:
                                self.cbar = self.ax.figure.colorbar(self.im, format="{x:.2e}", ax=self.ax)
                        else:
                                self.cbar = None
                        self.ax.set_title(self.filename_list[self.i])
                        self.fig.tight_layout()

                        self.fig.subplots_adjust(bottom=0.2)

                        self.ax_previous = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
                        self.ax_next     = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])

                        self.button_previous = Button(self.ax_previous, 'Previous')
                        self.button_previous.on_clicked(self.previous_mesh)

                        self.button_next = Button(self.ax_next, 'Next')
                        self.button_next.on_clicked(self.next_mesh)

                        if delay > 0:
                                self.timer = self.fig.canvas.new_timer(interval = delay * 1e3)
                                self.timer.add_callback(self.next_mesh)
                        else:
                                self.timer = None

                def get_mesh(self):
                        if self.i in self.meshes:
                                mesh = self.meshes[self.i]
                        else:
                                filename = self.filename_list[self.i]
                                mesh = read_mesh(filename)
                                self.meshes[self.i] = mesh
                        return mesh

                def next_mesh(self, event=None):
                        self.i += 1
                        self.i = self.i % len(self.filename_list)
                        mesh = self.get_mesh()
                        if self.cbar is not None:
                                self.cbar.remove()
                        self.im.remove()
                        self.im = self.ax.imshow(mesh)
                        if self.cbar is not None:
                                self.cbar = self.ax.figure.colorbar(self.im, format="{x:.2e}", ax=self.ax)
                        self.ax.set_title(self.filename_list[self.i])
                        plt.draw()

                def previous_mesh(self, event=None):
                        self.i -= 1
                        self.i = self.i % len(self.filename_list)
                        mesh = self.get_mesh()
                        if self.cbar is not None:
                                self.cbar.remove()
                        self.im.remove()
                        self.im = self.ax.imshow(mesh)
                        if self.cbar is not None:
                                self.cbar = self.ax.figure.colorbar(self.im, format="{x:.2e}", ax=self.ax)
                        self.ax.set_title(self.filename_list[self.i])
                        plt.draw()

                def show(self):
                        if self.timer is not None:
                                self.timer.start()
                        plt.show()

        print("plot_mesh_list")
        sequence = Mesh_sequence(filename_list, args.delay)
        sequence.show()


def display_single_mesh(args, filename):
        mesh = read_mesh(filename)
        print(f"{filename}:")
        print(mesh)

def display_mesh_list(args, filename_list):
        for filename in filename_list:
                display_single_mesh(args, filename)
                print()

def main():
        argparser = argparse.ArgumentParser(description="disp_mesh")
        argparser.add_argument('-p', '--plot', help='plot mesh as heatmap', action='store_true')
        argparser.add_argument('-c', '--colorbar', help='plot a colorbar next to the mesh', action='store_true')
        argparser.add_argument('-d', '--delay', help='set pause delay between mesh plots in a sequence', action='store', type = float, default=0)

        (args, filename_list) = argparser.parse_known_args()

        if len(filename_list) < 1:
                return
        
        if args.plot:
                if len(filename_list) == 1:
                        plot_single_mesh(args, filename_list[0])
                else:
                        plot_mesh_list(args, filename_list)
        else:
                display_mesh_list(args, filename_list)

main()
