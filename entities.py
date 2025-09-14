# at top
import numpy as np
from utils import NUM_ENEMIES

class Entities:
    def __init__(self, world, num_enemies=NUM_ENEMIES, min_goal_md=None):
        taken = set()

        # place hero first
        self.hero = world.random_free_cell(taken); taken.add(self.hero)

        # pick goal far from hero
        if min_goal_md is None:
            min_goal_md = world.n // 2  # ≥32 on a 64x64
        hr, hc = self.hero
        free_rc = np.argwhere(world.grid == 0)  # FREE
        free = [tuple(rc) for rc in map(tuple, free_rc) if tuple(rc) not in taken]
        manh = lambda rc: abs(rc[0]-hr) + abs(rc[1]-hc)
        goal = max(free, key=manh)
        if manh(goal) < min_goal_md:
            corners = [(0,0),(0,world.n-1),(world.n-1,0),(world.n-1,world.n-1)]
            corners = [c for c in corners if world.grid[c] == 0 and c not in taken]
            if corners: goal = max(corners, key=manh)
        self.goal = goal; taken.add(self.goal)

        # enemies
        self.enemies = []
        for _ in range(num_enemies):
            rc = world.random_free_cell(taken); taken.add(rc)
            self.enemies.append(rc)

