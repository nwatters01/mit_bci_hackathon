"""GUI."""

import logging
import math
import numpy as np
import sys
import time
import tkinter as tk

from PIL import Image
from PIL import ImageTk

_WINDOW_ASPECT_RATIO = 1.  # height/width for the gui window


class MouseFrame():
    """Tkinter frame for mouse interaction."""
    def __init__(self, canvas, canvas_half_width, name='mouse'):
        """Constructor."""
        self.name = name
        canvas.bind('<Motion>', self._mouse_move)
        self._canvas_half_width = canvas_half_width
        self._mouse_coords = np.array([0.5, 0.5])

    def _mouse_move(self, event):
        """Place the self._mouse_coords (x, y) coordinates of a mouse event."""
        centered_event_coords = (
            np.array([event.x, event.y], dtype=float) - self._canvas_half_width)
        centered_event_coords = np.clip(
            centered_event_coords,
            -self._canvas_half_width,
            self._canvas_half_width,
        )
        self._mouse_coords = 0.5 * (
            1 + centered_event_coords.astype(float) / self._canvas_half_width)

    @property
    def action(self):
        """Return the mouse's position as an action in [0, 1] x [0, 1]."""
        return np.array([self._mouse_coords[0], 1. - self._mouse_coords[1]])
    
    
class KeyboardFrame(tk.Frame):
    """Tkinter frame for keyboard interaction."""

    def __init__(self, root, canvas_half_width=100, name='keyboard'):
        """Constructor."""
        super(KeyboardFrame, self).__init__(root)
        self.name = name
        self._current_key = 4  # Do-nothing action

        # Create a canvas
        self.canvas = tk.Canvas(
            width=2 * canvas_half_width,
            height=2 * canvas_half_width)

        # Add bindings for key presses and releases
        root.bind('<KeyPress>', self._key_press)
        root.bind('<KeyRelease>', self._key_release)

    def _get_action_from_event(self, event):
        if event.keysym == 'Left':
            return 0
        elif event.keysym == 'Right':
            return 1
        elif event.keysym == 'Down':
            return 2
        elif event.keysym == 'Up':
            return 3
        else:
            return None

    def _key_press(self, event):
        self._current_key = self._get_action_from_event(event)

    def _key_release(self, event):
        if self._get_action_from_event(event) == self._current_key:
            self._current_key = None

    @property
    def action(self):
        if self._current_key is not None:
            return self._current_key
        else:
            return 4  # Do-nothing action


class GUI():
    """GUI."""

    def __init__(self, env, render_size, fps=60):
        """Constructor."""
        self._env = env
        self._ms_per_step = 1000. / fps
        self._canvas_half_width = render_size / 2

        # Create root Tk window and fix its size
        self.root = tk.Tk()
        frame_width = str(render_size)
        frame_height = str(int(_WINDOW_ASPECT_RATIO * render_size))
        self.root.geometry(frame_width + 'x' + frame_height)

        # Bind escape key to exit
        def _close(_):
            self._env.agent.snapshot()
            sys.exit()
        self.root.bind('<Escape>', _close)

        ########################################################################
        # Create the environment display and pack it into the top of the window.
        ########################################################################

        image = env.reset()
        self._env_canvas = tk.Canvas(
            self.root, width=render_size, height=render_size)
        self._env_canvas.pack(side=tk.TOP)
        img = ImageTk.PhotoImage(image=Image.fromarray(image))
        self._env_canvas.img = img
        self.image_on_canvas = self._env_canvas.create_image(
            0, 0, anchor="nw", image=self._env_canvas.img)
        self._textbox_id = self._env_canvas.create_text(
            int(self._canvas_half_width), 20, fill='white',
            font=('Helvetica 20'),
        )

        ########################################################################
        # Create the gui frame and pack it into the bottom of the window.
        ########################################################################

        logging.info('Grid action space, use arrow keys.')
        keyboard_frame = KeyboardFrame(
            self.root,
            canvas_half_width=self._canvas_half_width,
        )
        keyboard_frame.canvas.pack(side=tk.BOTTOM)
        mouse_frame = MouseFrame(
            self.root,
            canvas_half_width=self._canvas_half_width,
        )
        self.gui_frames = [keyboard_frame, mouse_frame]

        ########################################################################
        # Start run loop, automatically running the environment.
        ########################################################################

        self.root.after(math.floor(self._ms_per_step), self.step)
        self.root.mainloop()

    def render(self, observation):
        """Render the environment display and reward plot."""
        # Set the image in the environment display to the new observation
        self._env_canvas.img = ImageTk.PhotoImage(Image.fromarray(
            observation))
        self._env_canvas.itemconfig(
            self.image_on_canvas, image=self._env_canvas.img)

    def step(self):
        """Take an action in the environment and render."""
        step_start_time = time.time()
        action = {x.name: x.action for x in self.gui_frames}
        observation = self._env.step(action)
        
        # Display new observation
        self.render(observation)
        self._env_canvas.itemconfigure(self._textbox_id, text=self._env.title)

        # Recurse to step again after self._ms_per_step milliseconds
        step_end_time = time.time()
        delay = (step_end_time - step_start_time) * 1000  # convert to ms
        self.root.after(math.floor(self._ms_per_step - delay), self.step)