import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sympy.physics.mechanics import actuator


class Environment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()

        # Action space: [Source X, Source Y, Action Type, Target X, Target Y]
        # X and Y are 0-7 (for an 8x8 grid). Action Type is 0 (Move) or 1 (Spawn Unit) or 2 (capture city)
        self.turn_count = None
        self.captured_cities = 0
        self.state = None
        self.action_shape = (8, 8, 3, 8, 8)
        self.action_space = spaces.Discrete(math.prod(self.action_shape))

        # Observation space: 2 layers of 8x8 grids.
        # Layer 0: Cities (1 = Player City, 2 = Neutral City)
        # Layer 1: Units (1 = Player Unit)
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 8, 8), dtype=np.uint8)

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize an empty 2-layer 8x8 grid
        self.state = np.zeros((2, 8, 8), dtype=np.uint8)

        # Randomize unique city locations!
        # Choose 4 distinct spots from 64 tiles
        locations = self.np_random.choice(64, 4, replace=False)
        city_coords = [np.unravel_index(loc, (8, 8)) for loc in locations]

        # First location is the player's starting city and unit
        px, py = city_coords[0]
        self.state[0, px, py] = 1 # Player city
        self.state[1, px, py] = 1 # Player unit

        # The other 3 locations are neutral cities (using 2 for neutral)
        for nx, ny in city_coords[1:]:
            self.state[0, nx, ny] = 2

        self.turn_count = 0
        self.captured_cities = 0

        # Return observation and an empty info dictionary
        return self.state, {}

    def step(self, action):
        dims = np.unravel_index(action, self.action_shape)
        src_x, src_y, act_type, tgt_x, tgt_y = [int(d) for d in dims]
        reward = 0.0
        terminated = False
        truncated = False

        # 1. ACTION: MOVE
        if act_type == 0:
            # Check if we have a unit at the source coordinates
            if self.state[1, src_x, src_y] == 1:
                # Polytopia movement constraint: 1 tile in any direction
                if abs(src_x - tgt_x) <= 1 and abs(src_y - tgt_y) <= 1:
                    # Check if target tile is empty of units
                    if self.state[1, tgt_x, tgt_y] == 0:
                        self.state[1, src_x, src_y] = 0  # Remove from old position
                        self.state[1, tgt_x, tgt_y] = 1  # Move to new position
                        reward += 0.005 # Small reward for successfully moving

        # 2. ACTION: SPAWN UNIT
        elif act_type == 1:
            # Check if the source is our city
            if self.state[0, src_x, src_y] == 1:
                # Check if target is empty of units
                if self.state[1, src_x, src_y] == 0:
                    self.state[1, src_x, src_y] = 1 # Create new unit

        # 3. ACTION: CAPTURE CITY
        elif act_type == 2:
            # Check if we have unit in position
            if self.state[1, src_x, src_y] == 1:
                # Check if city is not ours
                if self.state[0, src_x, src_y] == 2:
                    # Claim city
                    self.state[0, src_x, src_y] = 1
                    reward += 10.0
                    self.captured_cities += 1

        if self.captured_cities == 3:
            terminated = True

        # Keep track of time to avoid infinite games during training
        reward -= 0.02
        self.turn_count += 1
        if self.turn_count > 100:
            truncated = True

        return self.state, reward, terminated, truncated, {}

    def get_valid_mask(self):
        mask = np.zeros(self.action_shape, dtype=bool)
        full_dst = np.ones((8, 8))

        # vaya merdazo tengo aqui
        for i in range(self.action_shape[0]):
            for j in range(self.action_shape[1]):
                # 1. ACTION: MOVE
                if self.state[1, i, j] == 1:
                    for x in range(-1, 2):
                        cx = max(min(i + x, 7), 0)
                        for y in range(-1, 2):
                            cy = max(min(j + y, 7), 0)
                            if not (x == 0 and y == 0):
                                mask[i, j, 0, cx, cy] = 1
                                # a tomar lo por puto culo
                    if self.state[0, i, j] == 2:
                        mask[i, j, 2] = full_dst

                # 2. ACTION: SPAWN UNIT
                if self.state[0, i, j] == 1:
                    # Check if target is empty of units
                    if self.state[1, i, j] == 0:
                        mask[i, j, 1] = full_dst

        return mask.flatten()