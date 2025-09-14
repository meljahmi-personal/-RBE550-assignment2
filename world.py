# world.py
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import FREE, OBST, JUNK, GRID_SIZE, TARGET_FILL, rng, SHOW_FIGURES

TETROMINOES = {
    "I": [(0,0),(1,0),(2,0),(3,0)],
    "O": [(0,0),(0,1),(1,0),(1,1)],
    "T": [(0,0),(0,1),(0,2),(1,1)],
    "L": [(0,0),(1,0),(2,0),(2,1)],
    "S": [(0,1),(0,2),(1,0),(1,1)]
}

class GridWorld:
    def __init__(self, n=GRID_SIZE, target_fill=TARGET_FILL):
        self.n = n
        self.target_fill = target_fill
        self.grid = np.zeros((n, n), dtype=int)

    # ----- obstacle gen -----
    @staticmethod
    def _rot(shape):  return [(-c, r) for (r, c) in shape]
    @staticmethod
    def _flip(shape): return [(r, -c) for (r, c) in shape]

    def _place_tetromino(self, shape, row, col):
        coords = []
        for dr, dc in shape:
            r, c = row + dr, col + dc
            if not (0 <= r < self.n and 0 <= c < self.n): return False
            if self.grid[r, c] == OBST: return False
            coords.append((r, c))
        for r, c in coords:
            self.grid[r, c] = OBST
        return True

    def generate_obstacles(self):
        placed = 0
        target = int(self.n * self.n * self.target_fill)
        while placed < target:
            shape = random.choice(list(TETROMINOES.values()))
            for _ in range(random.randint(0,3)):
                shape = self._rot(shape)
            if random.random() < 0.5:
                shape = self._flip(shape)
            r = random.randint(0, self.n-1)
            c = random.randint(0, self.n-1)
            if self._place_tetromino(shape, r, c):
                placed = int(np.sum(self.grid == OBST))

    # ----- sampling / stats -----
    def random_free_cell(self, taken=None):
        if taken is None: taken = set()
        free_rc = np.argwhere(self.grid == FREE)
        mask = [tuple(rc) not in taken for rc in map(tuple, free_rc)]
        free_rc = free_rc[mask]
        if len(free_rc) == 0:
            raise RuntimeError("No free cells left to place.")
        r, c = free_rc[rng.integers(len(free_rc))]
        return int(r), int(c)

    def count_junk(self): 
        return int(np.sum(self.grid == JUNK))

    def stats_str(self):
        total = self.grid.size
        obst  = np.count_nonzero(self.grid == OBST)
        junk  = np.count_nonzero(self.grid == JUNK)
        free  = np.count_nonzero(self.grid == FREE)
        return (f"free={free}, obst={obst}, junk={junk}, "
                f"blocked={(obst+junk)}/{total} = {(obst+junk)/total:.4f}")

    # ----- rendering -----
    def draw(self, goal, hero, enemies, path=None, save_path="flatland.png", overlay=None, show_path=False):
        n = self.n
        fig, ax = plt.subplots(figsize=(9,9), dpi=120, facecolor="white")
        ax.set_facecolor("white")

        # Render blocked mask (obstacles + junk = 1, free = 0)
        blocked = (self.grid != FREE).astype(int)
        ax.imshow(blocked, cmap="gray_r", origin="upper", vmin=0, vmax=1)

        # Gridlines
        ax.set_xticks(np.arange(-.5, n, 1))
        ax.set_yticks(np.arange(-.5, n, 1))
        ax.grid(which="both", color="#cccccc", linestyle="-", linewidth=0.3)
        ax.set_xticklabels([]); ax.set_yticklabels([])

        def centers(rc_list):
            rr = [r + 0.5 for (r, c) in rc_list]
            cc = [c + 0.5 for (r, c) in rc_list]
            return cc, rr

        gx, gy = centers([goal])
        hx, hy = centers([hero])

        # Draw goal + hero (overlap-aware)
        if hero == goal:
            # Goal first
            ax.scatter(gx, gy, marker='s', s=140, facecolors='tab:green',
                       edgecolors='black', linewidths=0.8, label='Goal', zorder=3)
            # Hero as hollow ring on top so both are visible
            ax.scatter(hx, hy, marker='o', s=200, facecolors='none',
                       edgecolors='tab:blue', linewidths=2.2, label='Hero (at goal)', zorder=4)
        else:
            ax.scatter(gx, gy, marker='s', s=140, facecolors='tab:green',
                       edgecolors='black', linewidths=0.8, label='Goal',  zorder=3)
            ax.scatter(hx, hy, marker='o', s=140, facecolors='tab:blue',
                       edgecolors='black', linewidths=0.8, label='Hero',  zorder=3)

        # Enemies
        if enemies:
            ex, ey = centers(enemies)
            ax.scatter(ex, ey, marker='^', s=120, facecolors='tab:red',
                       edgecolors='black', linewidths=0.6, label='Enemy', zorder=3)

        # Optional path overlay (off by default)
        if show_path and path and len(path) >= 2:
            xs = [c + 0.5 for (r, c) in path]
            ys = [r + 0.5 for (r, c) in path]
            ax.plot(xs, ys, linewidth=1.8, alpha=0.9, zorder=2)

        density = np.mean(self.grid != FREE)
        ax.set_title(f"{n}×{n} World (~20% tetromino obstacles)  Density={density:.2f}")
        leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
        leg.get_frame().set_alpha(0.9)

        if overlay:
            ax.text(0.02, 0.98, overlay, transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                    edgecolor="black", alpha=0.85))

        plt.tight_layout(pad=0.2)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

        if SHOW_FIGURES:
            plt.show(block=False)
            plt.pause(0.5)
        plt.close()

