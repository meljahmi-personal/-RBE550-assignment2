import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

GRID_SIZE = 64
TARGET_FILL = 0.20  # 20% obstacles

# Tetromino base shapes (as lists of (r, c) offsets)
TETROMINOES = {
    "I": [(0,0),(1,0),(2,0),(3,0)],
    "O": [(0,0),(0,1),(1,0),(1,1)],
    "T": [(0,0),(0,1),(0,2),(1,1)],
    "L": [(0,0),(1,0),(2,0),(2,1)],
    "S": [(0,1),(0,2),(1,0),(1,1)]
}

# --- agent/constants ---
FREE, OBST = 0, 1
NUM_ENEMIES = 10
JUNK = 2  # permanent obstacle created by enemy crashes


rng = np.random.default_rng()  # pass a seed if you want reproducibility, e.g., np.random.default_rng(42)

def random_free_cell(grid, taken=None):
    """Pick a random FREE (0) cell not already taken."""
    if taken is None:
        taken = set()
    free_rc = np.argwhere(grid == FREE)
    # filter out already taken cells
    mask = [tuple(rc) not in taken for rc in map(tuple, free_rc)]
    free_rc = free_rc[mask]
    if len(free_rc) == 0:
        raise RuntimeError("No free cells left to place.")
    r, c = free_rc[rng.integers(len(free_rc))]
    return int(r), int(c)


def place_agents(grid):
    """Return (goal, hero, enemies) placed on FREE cells with no overlaps."""
    taken = set()
    goal = random_free_cell(grid, taken); taken.add(goal)
    hero = random_free_cell(grid, taken); taken.add(hero)
    enemies = []
    for _ in range(NUM_ENEMIES):
        rc = random_free_cell(grid, taken)
        taken.add(rc)
        enemies.append(rc)
    return goal, hero, enemies


def step_toward(hero_rc, enemy_rc):
    """One 4-neighbor greedy step toward hero (ignores obstacles per spec)."""
    (hr, hc), (er, ec) = hero_rc, enemy_rc
    dr, dc = hr - er, hc - ec
    if abs(dr) >= abs(dc):
        nr = er + (1 if dr > 0 else -1 if dr < 0 else 0)
        nc = ec
    else:
        nr = er
        nc = ec + (1 if dc > 0 else -1 if dc < 0 else 0)
    return (nr, nc)


def in_bounds(r, c, n):
    return 0 <= r < n and 0 <= c < n


def update_enemies(grid, hero, enemies):
    """
    Simultaneous enemy update:
      - Each enemy takes 1 greedy step toward hero (ignoring map).
      - If intended target is OOB or blocked (OBST/JUNK) -> crash:
            enemy's CURRENT cell becomes JUNK; enemy removed.
      - If any intended target == hero -> hero destroyed (game over).
      - If >=2 enemies intend the SAME free target -> collision:
            that target cell becomes JUNK; all those enemies removed.
      - Remaining enemies move to their unique targets.
    Returns: (new_grid, new_enemies, status) with status in {"ok","hero_caught"}.
    """
    n = grid.shape[0]

    # 1) compute intents
    intents = [step_toward(hero, e) for e in enemies]

    # 2) hero contact?
    if any(t == hero for t in intents):
        return grid, enemies, "hero_caught"

    # 3) crashes into wall/blocked
    will_crash = set()
    for i, (r, c) in enumerate(intents):
        if not in_bounds(r, c, n) or grid[r, c] in (OBST, JUNK):
            will_crash.add(i)

    # 4) collisions: multiple enemies to same free target
    bucket = defaultdict(list)
    for i, t in enumerate(intents):
        if i not in will_crash:
            bucket[t].append(i)
    collision_targets = {t for t, idxs in bucket.items() if len(idxs) >= 2}
    colliders = {i for t, idxs in bucket.items() if t in collision_targets for i in idxs}

    # 5) apply crashes -> JUNK at CURRENT enemy cells
    new_grid = grid.copy()
    survivors = []
    for i, (er, ec) in enumerate(enemies):
        if i in will_crash:
            new_grid[er, ec] = JUNK
        else:
            survivors.append(i)

    # 6) apply enemy-enemy collisions -> JUNK at TARGET cell
    still = []
    for i in survivors:
        if i in colliders:
            tr, tc = intents[i]
            new_grid[tr, tc] = JUNK
        else:
            still.append(i)

    # 7) move remaining enemies
    new_enemies = []
    for i in still:
        tr, tc = intents[i]
        new_enemies.append((tr, tc))

    return new_grid, new_enemies, "ok"


def rotate(shape):
    return [(-c, r) for (r, c) in shape]


def flip(shape):
    return [(r, -c) for (r, c) in shape]


def place_tetromino(grid, shape, row, col):
    coords = []
    for dr, dc in shape:
        r, c = row + dr, col + dc
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
            return False  # out of bounds
        if grid[r, c] == 1:
            return False  # overlap
        coords.append((r, c))
    for r, c in coords:
        grid[r, c] = 1
    return True


def generate_obstacles():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    obstacle_count = 0
    target_cells = int(GRID_SIZE * GRID_SIZE * TARGET_FILL)

    while obstacle_count < target_cells:
        shape = random.choice(list(TETROMINOES.values()))
        # Random rotations & flips
        for _ in range(random.randint(0,3)):
            shape = rotate(shape)
        if random.random() < 0.5:
            shape = flip(shape)
        row, col = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if place_tetromino(grid, shape, row, col):
            obstacle_count = np.sum(grid)
    return grid


def show_grid(grid):
    GRID_SIZE = grid.shape[0]
    fig, ax = plt.subplots(figsize=(8,8), dpi=120)

    # Display obstacles
    blocked = (grid != FREE).astype(int)  # OBST or JUNK -> 1 (black), FREE -> 0 (white)
    ax.imshow(blocked, cmap="gray_r", origin="upper", vmin=0, vmax=1)

    # Gridlines every cell
    ax.set_xticks(np.arange(-.5, GRID_SIZE, 1))
    ax.set_yticks(np.arange(-.5, GRID_SIZE, 1))
    ax.grid(which="both", color="black", linestyle="-", linewidth=0.2)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    density = np.sum(grid)/(GRID_SIZE*GRID_SIZE)
    ax.set_title(f"{GRID_SIZE}x{GRID_SIZE} Obstacle Map with Gridlines\nDensity={density:.2f}")
    
    plt.savefig("obstacle_map.png", dpi=300, bbox_inches="tight")
    
    plt.show(block=False)
    plt.pause(3)   # keeps the window open for 3 seconds
    plt.close()


def draw_world(grid, goal, hero, enemies, save_path="flatland_map.png"):
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(9,9), dpi=120, facecolor="white")
    ax.set_facecolor("white")

    blocked = (grid != FREE).astype(int)
    ax.imshow(blocked, cmap="gray_r", origin="upper", vmin=0, vmax=1)

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
    ex, ey = centers(enemies)

    ax.scatter(gx, gy, marker='s', s=140, facecolors='tab:green', edgecolors='black', linewidths=0.8, label='Goal',  zorder=3)
    ax.scatter(hx, hy, marker='o', s=140, facecolors='tab:blue',  edgecolors='black', linewidths=0.8, label='Hero',  zorder=3)
    ax.scatter(ex, ey, marker='^', s=120, facecolors='tab:red',   edgecolors='black', linewidths=0.6, label='Enemy', zorder=3)

    density = np.sum(blocked) / (n*n)
    ax.set_title(f"{n}×{n} World (~20% tetromino obstacles)\nDensity={density:.2f}")

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(block=False); plt.pause(3); plt.close()


def world_stats(grid):
    total = grid.size
    obst  = np.count_nonzero(grid == OBST)
    junk  = np.count_nonzero(grid == JUNK)
    free  = np.count_nonzero(grid == FREE)
    print(f"free={free}, obst={obst}, junk={junk}, "
          f"blocked={(obst+junk)}/{total} = {(obst+junk)/total:.4f}")


# --- Main ---
def main():
    grid = generate_obstacles()
    goal, hero, enemies = place_agents(grid)
    draw_world(grid, goal, hero, enemies, save_path="flatland_step0.png")

    status = "ok"
    for t in range(5):
        grid, enemies, status = update_enemies(grid, hero, enemies)
        print(f"t={t+1}, enemies={len(enemies)}, status={status}")
        world_stats(grid)
        draw_world(grid, goal, hero, enemies, save_path=f"flatland_step{t+1}.png")
        if status != "ok":
            break
        blocked_ratio = np.mean(grid != FREE)
        print(f"Blocked ratio = {blocked_ratio:.2f}")  # should be ~0.20 at t=0, then rise slowly




if __name__ == "__main__":
    main()





