"""GUI."""

from matplotlib import pyplot as plt
import numpy as np
import tkinter as tk


class CalibrationGUI():
    
    def __init__(self,
                 n_levels=5,
                 step_size=4,
                 ball_size=20,
                 canvas_width=800,
                 canvas_height=800,
                 buffer=10):
        self._n_levels = n_levels
        self._step_size = step_size
        self._ball_size = ball_size
        self._canvas_width=canvas_width
        self._canvas_height=canvas_height
        self._buffer = buffer
        
        self.root = tk.Tk()
        self.canvas = tk.Canvas(
            self.root, width=canvas_width, height=canvas_height)
        self.canvas.pack()
        
        self.target = self.canvas.create_oval(
            self._buffer,
            self._canvas_height - self._ball_size - self._buffer,
            self._ball_size + self._buffer,
            self._canvas_height - self._buffer,
            fill='red',
        )
        self.agent = self.canvas.create_oval(
            -3 * self._buffer,
            self._canvas_height - self._ball_size + 3 * self._buffer,
            self._ball_size - 3 * self._buffer,
            self._canvas_height + 3 * self._buffer,
            fill='green',
        )
        
        # Setup target deltas
        row_steps = (
            (self._canvas_width - 2 * (self._ball_size + self._buffer)) //
            self._step_size
        )
        column_steps = (
            (self._canvas_height - 2 * (self._buffer + self._ball_size)) //
            (self._step_size * (self._n_levels - 1))
        )
        deltas = []
        moving_right = 1
        for level in range(self._n_levels):
            # Add horizontal leg
            h_delta = moving_right * self._step_size
            for _ in range(row_steps):
                deltas.append((h_delta, 0))
            moving_right *= -1
            
            if level < self._n_levels - 1:
                # Add vertical leg
                for _ in range(column_steps):
                    deltas.append((0, -self._step_size))
        self._deltas = deltas
        self.reset()
        
    def _coords_to_pos(self, coords):
        w = float(coords[0]) / self._canvas_width
        w = 2 * w - 1
        h = 1. - float(coords[1]) / self._canvas_width
        return np.array([w, h])
    
    def _pos_to_coords(self, pos):
        w = self._canvas_width * 0.5 * (1 + pos[0])
        h = self._canvas_height * (1. - pos[1])
        return (w, h)
    
    def _move_object(self, obj, target_coords):
        current_coords = self.canvas.coords(obj)
        delta = (
            target_coords[0] - current_coords[0],
            target_coords[1] - current_coords[1],
        )
        self.canvas.move(obj, delta[0], delta[1])
        
    def set_callback(self, callback):
        self._callback = callback
        
    def reset(self):
        self._step_index = 0
        target_coords = (
            self._buffer,
            self._canvas_height - self._ball_size - self._buffer,
        )
        self._move_object(self.target, target_coords)
        
    def step(self):
        
        if self._step_index == 0:
            plt.pause(2)
        
        delta = self._deltas[self._step_index]
        self._step_index += 1
        
        finished = self._step_index >= len(self._deltas)
        self.canvas.move(self.target, delta[0], delta[1])
        target = self._coords_to_pos(self.canvas.coords(self.target))
        agent_pos = self._callback(target, finished)
        if agent_pos is not None:
            agent_coords = self._pos_to_coords(agent_pos)
            self._move_object(self.agent, agent_coords)
        
        if finished:
            self.reset()
            
        self.root.after(10, self.step)
        
    @property
    def n_features(self):
        return 2
        

if __name__ == '__main__':
    gui = CalibrationGUI()
    callback = lambda target, finished: None
    gui.set_callback(callback)
    gui.root.after(10, gui.step)
    gui.root.mainloop()
    