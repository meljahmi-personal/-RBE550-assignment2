# entities.py
from utils import NUM_ENEMIES

class Entities:
    def __init__(self, world, num_enemies=NUM_ENEMIES):
        taken = set()
        self.goal = world.random_free_cell(taken); taken.add(self.goal)
        self.hero = world.random_free_cell(taken); taken.add(self.hero)
        self.enemies = []
        for _ in range(num_enemies):
            rc = world.random_free_cell(taken); taken.add(rc)
            self.enemies.append(rc)

