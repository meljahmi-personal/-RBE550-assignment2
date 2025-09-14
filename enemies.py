# enemies.py
from collections import defaultdict
from utils import FREE, JUNK, sign

class EnemyModel:
    @staticmethod
    def _step_toward(hero_rc, enemy_rc):
        (hr, hc), (er, ec) = hero_rc, enemy_rc
        dr, dc = hr - er, hc - ec
        if abs(dr) >= abs(dc):
            nr, nc = er + sign(dr), ec
        else:
            nr, nc = er, ec + sign(dc)
        return (nr, nc)

    @staticmethod
    def step_enemies(world, hero, enemies):
        """
        Enemies greedily move 1 step toward hero.
        If an intended move is OOB/blocked/occupied/colliding -> enemy breaks (current cell -> JUNK).
        If any enemy steps onto hero -> lose.
        """
        n = world.n
        hero_r, hero_c = hero

        intents = []
        occupied = set(enemies)
        for (r, c) in enemies:
            nr, nc = EnemyModel._step_toward(hero, (r, c))
            intents.append(((r, c), (nr, nc)))

        dest_counts = defaultdict(int)
        for _, dest in intents:
            dest_counts[dest] += 1

        new_enemies = []
        for (r, c), (nr, nc) in intents:
            bad = False
            if not (0 <= nr < n and 0 <= nc < n): bad = True
            elif world.grid[nr, nc] != FREE:      bad = True
            elif (nr, nc) in occupied:            bad = True
            elif dest_counts[(nr, nc)] > 1:       bad = True

            if bad:
                world.grid[r, c] = JUNK
                continue

            if (nr, nc) == (hero_r, hero_c):
                return [], True
            new_enemies.append((nr, nc))

        return new_enemies, False

