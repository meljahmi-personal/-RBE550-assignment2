import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import csv
import json
from datetime import datetime
from collections import deque
import csv




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

NEI_ORDER = [(-1,0),(0,1),(1,0),(0,-1)]  # up, right, down, left (deterministic)'

# --- agent/constants ---
FREE, OBST = 0, 1
NUM_ENEMIES = 10
JUNK = 2  # permanent obstacle created by enemy crashes
SEED = 42  # change per run to reproduce
rng = np.random.default_rng(SEED)
random.seed(SEED)


"""Helpers"""
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def count_junk(grid):
    return int(np.sum(grid == JUNK))


def export_config_json(outdir, cfg: dict):
    with open(os.path.join(outdir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def export_log_csv(outdir, rows, header):
    with open(os.path.join(outdir, "run_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def neighbors4_all(r, c, n):
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr, cc = r+dr, c+dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc


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


def proof_run(out_name="proof_run_001", max_steps=500, render_every=None):
    # Output folder with timestamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ensure_dir(f"{out_name}_{stamp}")

    # --- Build world
    grid = generate_obstacles()
    goal, hero, enemies = place_agents(grid)

    # --- Save START frame (compute path first)
    start_path, _ = bfs_path(grid, hero, goal, set(enemies), keepout=1)
    draw_world(grid, goal, hero, enemies,
               path=start_path,
               save_path=os.path.join(outdir, "frame_start.png"))

    # --- Simulate with logging
    log_rows = []
    header = ["t", "hero_r", "hero_c", "enemy_count", "junk_cells",
              "status", "bfs_expanded", "path_len"]

    status = "running"
    t = 0

    # compute initial path + expansions
    path0, meta0 = bfs_path(grid, hero, goal, set(enemies), keepout=1)
    plen0 = len(path0) if path0 else 0
    exp0 = int(meta0.get("expanded", 0))

    log_rows.append([t, hero[0], hero[1], len(enemies),
                     count_junk(grid), status, exp0, plen0])


    while t < max_steps and status == "running":
        # one simulation step (hero BFS + enemies move)
        hero, enemies, status = sim_step(grid, hero, goal, enemies)
        t += 1
       
        path_t, meta_t = bfs_path(grid, hero, goal, set(enemies), keepout=1)
        plen_t = len(path_t) if path_t else 0
        exp_t = int(meta_t.get("expanded", 0))

        log_rows.append([t, hero[0], hero[1], len(enemies),
                         count_junk(grid), status, exp_t, plen_t])

        # Optional periodic snapshots with the CURRENT path
        if render_every and (t % render_every == 0):
            cur_path, _ = bfs_path(grid, hero, goal, set(enemies), keepout=1)
            draw_world(grid, goal, hero, enemies,
                       path=cur_path,
                       save_path=os.path.join(outdir, f"frame_{t:04d}.png"))

    # --- Save FINAL frame (compute path for the terminal state)
    final_path, _ = bfs_path(grid, hero, goal, set(enemies), keepout=1)
    draw_world(grid, goal, hero, enemies,
               path=final_path,
               save_path=os.path.join(outdir, "frame_final.png"))

    # --- Export logs + config
    export_log_csv(outdir, log_rows, header)
    cfg = {
        "grid_size": int(grid.shape[0]),
        "target_fill": float(TARGET_FILL),
        "num_enemies": int(NUM_ENEMIES),
        "seed": SEED,
        "max_steps": max_steps,
        "render_every": render_every,
        "result": status,
        "steps_taken": t
    }
    export_config_json(outdir, cfg)
    
    # --- Optional GIF from saved frames ---
    try:
        import imageio, glob
        frames = sorted(glob.glob(os.path.join(outdir, "frame_*.png")))
        if len(frames) >= 2:
            imgs = [imageio.v2.imread(f) for f in frames]
            imageio.mimsave(os.path.join(outdir, "run.gif"), imgs, duration=0.06)  # ~16 FPS
            print("[gif] wrote", os.path.join(outdir, "run.gif"))
    except Exception as e:
        print("[gif] skipped:", e)


    print(f"[proof] result={status} steps={t} dir={outdir}")
    return outdir, status, t



def bfs_path(grid, start, goal, enemy_set, keepout=0):
    """
    Breadth-First Search (BFS) on a 4-connected grid.
    Returns:
      path : list of (r,c) from start to goal (or None if no path)
      meta : dict with {"expanded": count}
    Blocks: obstacles, junk, enemies, and (optionally) cells adjacent to enemies (keepout).
    """
    n = grid.shape[0]
    if start == goal:
        return [start], {"expanded": 0}

    blocked = set(map(tuple, np.argwhere(grid != FREE)))
    blocked.discard(start)
    blocked |= set(enemy_set)

    if keepout > 0:
        for er, ec in enemy_set:
            for dr, dc in NEI_ORDER:
                rr, cc = er+dr, ec+dc
                if 0 <= rr < n and 0 <= cc < n:
                    blocked.add((rr, cc))

    q = deque([start])
    parent = {start: None}
    expanded = 0

    while q:
        r, c = q.popleft(); expanded += 1
        for rr, cc in neighbors4_all(r, c, n):
            nxt = (rr, cc)
            if nxt in parent:
                continue
            if nxt in blocked and nxt != goal:
                continue
            parent[nxt] = (r, c)
            if nxt == goal:
                path = [nxt]
                while path[-1] is not None:
                    path.append(parent[path[-1]])
                path.pop(); path.reverse()
                return path, {"expanded": expanded}
            q.append(nxt)

    return None, {"expanded": expanded}


def sign(x): 
    return (x > 0) - (x < 0)


def step_enemies(grid, hero, enemies):
    """
    Enemies move 1 cell toward hero (Manhattan).
    If an enemy's move would hit boundary/obstacle/another enemy -> it BREAKS:
      - current cell becomes junk (grid=1), enemy removed.
    If an enemy moves into hero -> returns game_over=True.
    """
    n = grid.shape[0]
    hero_r, hero_c = hero
    # First compute intended moves
    intents = []
    occupied = set(enemies)  # current enemy cells to detect collisions
    for (r, c) in enemies:
        dr = sign(hero_r - r)
        dc = sign(hero_c - c)
        # prefer the axis with larger distance; tie-break deterministic
        if abs(hero_r - r) >= abs(hero_c - c):
            nr, nc = r + dr, c
            alt = (r, c + dc)
        else:
            nr, nc = r, c + dc
            alt = (r + dr, c)
        intents.append(((r, c), (nr, nc), alt))

    new_enemies = []
    dest_counts = {}
    # Count primary destinations
    for _, dest, _ in intents:
        dest_counts[dest] = dest_counts.get(dest, 0) + 1

    # Resolve each enemy
    for (r, c), (nr, nc), alt in intents:
        # Check primary dest
        bad = False
        if not (0 <= nr < n and 0 <= nc < n): 
            bad = True
        elif grid[nr, nc] != FREE:  # blocks OBST (1) and JUNK (2)
            bad = True
        elif (nr, nc) in occupied: 
            bad = True
        elif dest_counts[(nr, nc)] > 1: 
            bad = True  # head-on into same cell

        if bad:
            # BREAKS -> current cell becomes junk, enemy removed
            grid[r, c] = JUNK
            continue

        # Move succeeds
        if (nr, nc) == (hero_r, hero_c):
            return [], True  # hero destroyed
        new_enemies.append((nr, nc))

    return new_enemies, False


def sim_step(grid, hero, goal, enemies):
    enemy_set = set(enemies)

    # --- HERO move (BFS replan each step)
    path, meta = bfs_path(grid, hero, goal, enemy_set, keepout=1)  # set keepout=0 if no buffer
    # print(f"[BFS] expanded={meta['expanded']}")  # uncomment if  live log

    if path and len(path) >= 2:
        hero_next = path[1]
    else:
        hero_next = hero  # no path -> stay 

    if hero_next in enemy_set:
        return hero, enemies, "lose"

    hero = hero_next
    if hero == goal:
        return hero, enemies, "win"

    # ENEMIES move exactly as before
    enemies, game_over = step_enemies(grid, hero, enemies)
    if game_over:
        return hero, enemies, "lose"

    if hero == goal:
        return hero, enemies, "win"

    return hero, enemies, "running"



def simulate(grid, goal, hero, enemies, max_steps=500, render_every=None):
    """
    Run the simulation loop until hero wins, loses, or max_steps reached.
    If render_every is set (e.g., 10), save a snapshot frame every N steps.
    """
    for step in range(1, max_steps+1):
        hero, enemies, status = sim_step(grid, hero, goal, enemies)

        if render_every and (step % render_every == 0 or status != "running"):
            draw_world(grid, goal, hero, enemies, save_path=f"frame_{step:04d}.png")

        if status != "running":
            return status, step
    return "running", max_steps



def draw_path(ax, path):
    if not path or len(path) < 2: 
        return
    xs = [c+0.5 for r,c in path]
    ys = [r+0.5 for r,c in path]
    ax.plot(xs, ys, linewidth=1.5, alpha=0.9)



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



def draw_world(grid, goal, hero, enemies, path=None, save_path="flatland_map.png", overlay=None):
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
    if enemies:
        ex, ey = centers(enemies)

    ax.scatter(gx, gy, marker='s', s=140, facecolors='tab:green', edgecolors='black', linewidths=0.8, label='Goal',  zorder=3)
    ax.scatter(hx, hy, marker='o', s=140, facecolors='tab:blue',  edgecolors='black', linewidths=0.8, label='Hero',  zorder=3)
    if enemies:
        ax.scatter(ex, ey, marker='^', s=120, facecolors='tab:red', edgecolors='black', linewidths=0.6, label='Enemy', zorder=3)

    if path and len(path) >= 2:
        xs = [c + 0.5 for (r, c) in path]
        ys = [r + 0.5 for (r, c) in path]
        ax.plot(xs, ys, linewidth=1.8, alpha=0.9, zorder=2)

    density = np.sum(blocked) / (n*n)
    ax.set_title(f"{n}×{n} World (~20% tetromino obstacles)  Density={density:.2f}")

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_alpha(0.9)

    # --- overlay box (top-left) ---
    if overlay:
        ax.text(
            0.02, 0.98, overlay,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85)
        )

    plt.tight_layout(pad=0.2)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    # in draw_world at the end
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def world_stats(grid):
    total = grid.size
    obst  = np.count_nonzero(grid == OBST)
    junk  = np.count_nonzero(grid == JUNK)
    free  = np.count_nonzero(grid == FREE)
    print(f"free={free}, obst={obst}, junk={junk}, "
          f"blocked={(obst+junk)}/{total} = {(obst+junk)/total:.4f}")



def load_run_csv(outdir):
    path = os.path.join(outdir, "run_log.csv")
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # convert to ints where useful
            row["t"] = int(row["t"])
            row["enemy_count"] = int(row["enemy_count"])
            row["junk_cells"] = int(row["junk_cells"])
            # these two exist if you added the earlier BFS-logging patch
            row["bfs_expanded"] = int(row.get("bfs_expanded", 0) or 0)
            row["path_len"] = int(row.get("path_len", 0) or 0)
            rows.append(row)
    return rows

def plot_from_outdir(outdir):
    data = load_run_csv(outdir)
    t = [d["t"] for d in data]
    enemies = [d["enemy_count"] for d in data]
    junk = [d["junk_cells"] for d in data]
    exps = [d["bfs_expanded"] for d in data]
    plen = [d["path_len"] for d in data]

    # 1) BFS expansions per step (if you added that column)
    if any(exps):
        plt.figure()
        plt.plot(t, exps)
        plt.xlabel("time step")
        plt.ylabel("BFS nodes expanded")
        plt.title("BFS expansions vs time")
        plt.grid(True, linewidth=0.3)
        plt.savefig(os.path.join(outdir, "plot_expansions.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # 2) Enemies & junk over time
    plt.figure()
    plt.plot(t, enemies, label="enemies")
    plt.plot(t, junk, label="junk")
    plt.xlabel("time step")
    plt.ylabel("count")
    plt.title("Enemies and Junk vs time")
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.savefig(os.path.join(outdir, "plot_enemies_junk.png"), dpi=200, bbox_inches="tight")
    plt.close()

def plot_latest_run():
    # pick the most recent flatland_proof_* folder
    runs = sorted(glob.glob("flatland_proof_*"))
    if not runs:
        print("No flatland_proof_* folder found.")
        return
    outdir = runs[-1]
    print("Plotting from:", outdir)
    plot_from_outdir(outdir)

# call this if you want to auto-plot the most recent run:
# plot_latest_run()


# --- Main ---
def main():
    grid = generate_obstacles()
    goal, hero, enemies = place_agents(grid)

    # initial snapshot (with path)
    path, _ = bfs_path(grid, hero, goal, set(enemies), keepout=1)
    draw_world(grid, goal, hero, enemies, path=path, save_path="flatland_step0.png")

    # run a few steps with consistent updates
    for t in range(5):
        hero, enemies, status = sim_step(grid, hero, goal, enemies)
        print(f"t={t+1}, enemies={len(enemies)}, status={status}")
        world_stats(grid)

        # recompute path for the CURRENT state and draw
        path, _ = bfs_path(grid, hero, goal, set(enemies), keepout=1)
        draw_world(grid, goal, hero, enemies, path=path, save_path=f"flatland_step{t+1}.png")

        if status != "running":
            break

        blocked_ratio = np.mean(grid != FREE)
        print(f"Blocked ratio = {blocked_ratio:.2f}")


def quick_summary(outdir):
    import csv, os
    with open(os.path.join(outdir, "run_log.csv")) as f:
        r = list(csv.DictReader(f))
    steps = int(r[-1]["t"])
    enemies_end = int(r[-1]["enemy_count"])
    junk_end = int(r[-1]["junk_cells"])
    exps = [int(x.get("bfs_expanded", 0) or 0) for x in r]
    avg_exp = sum(exps) / len(exps)
    print(f"steps={steps}, enemies_end={enemies_end}, junk_end={junk_end}, avg_bfs_expanded={avg_exp:.1f}")


if __name__ == "__main__":

    main()
    outdir, status, steps = proof_run(out_name="flatland_proof", max_steps=500, render_every=50)
    print("Artifacts written to:", outdir)

    # --- make plots for this run ---
    plot_from_outdir(outdir)
    
    quick_summary(outdir)





