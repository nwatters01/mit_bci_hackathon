"""GUI."""

from matplotlib import pyplot as plt
import numpy as np
import tkinter as tk


class CalibrationGUI():
    
    def __init__(self,
                 n_levels=5,
                 step_size=3,
                 ball_size=20,
                 canvas_width=500,
                 canvas_height=500,
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
                
        self._step_index = 0
        
    def set_callback(self, callback):
        self._callback = callback
        
    def step(self):
        
        if self._step_index == 0:
            plt.pause(0.1)
        
        delta = self._deltas[self._step_index]
        self._step_index += 1
        
        finished = self._step_index >= len(self._deltas)
        self.canvas.move(self.target, delta[0], delta[1])
        target = self.canvas.coords(self.target)
        target = np.array([
            float(target[0]) / self._canvas_width,
            float(target[1]) / self._canvas_width,
        ])
        
        self._agent_pos = self._callback(target, finished)
        self._agent_pos = [
            self._canvas_width * self._agent_pos[0],
            self._canvas_height * self._agent_pos[1],
        ]
        
        if self._agent_pos is not None:
            try:
                a = self.canvas.coords(self.agent)
            except:
                return
            d = (self._agent_pos[0] - a[0], self._agent_pos[1] - a[1])
            self.canvas.move(self.agent, d[0], d[1])
        
        if not finished:
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
    