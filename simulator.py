# simulator.py
import os, glob
from datetime import datetime

from utils import GRID_SIZE, TARGET_FILL, NUM_ENEMIES, SEED
from utils import ensure_dir
from world import GridWorld
from entities import Entities
from enemies import EnemyModel
from planner import Planner
from logger import Logger

class Simulator:
    def __init__(self, keepout=1, render_every=None):
        self.keepout = keepout
        self.render_every = render_every

    def sim_step(self, world: GridWorld, ent: Entities, goal):
        enemy_set = set(ent.enemies)
        path, meta = Planner.bfs_path(world, ent.hero, goal, enemy_set, keepout=self.keepout)
        hero_next = path[1] if path and len(path) >= 2 else ent.hero

        if hero_next in enemy_set:
            return ent.hero, ent.enemies, "lose", meta, len(path) if path else 0

        ent.hero = hero_next
        if ent.hero == goal:
            return ent.hero, ent.enemies, "win", meta, len(path) if path else 0

        ent.enemies, game_over = EnemyModel.step_enemies(world, ent.hero, ent.enemies)
        if game_over:
            return ent.hero, ent.enemies, "lose", meta, len(path) if path else 0

        if ent.hero == goal:
            return ent.hero, ent.enemies, "win", meta, len(path) if path else 0

        return ent.hero, ent.enemies, "running", meta, len(path) if path else 0

    def proof_run(self, out_name="flatland_proof", max_steps=500):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = ensure_dir(f"{out_name}_{stamp}")

        world = GridWorld(GRID_SIZE, TARGET_FILL)
        world.generate_obstacles()
        ent = Entities(world, NUM_ENEMIES)

        start_path, _ = Planner.bfs_path(world, ent.hero, ent.goal, set(ent.enemies), keepout=self.keepout)
        world.draw(ent.goal, ent.hero, ent.enemies, path=start_path, save_path=os.path.join(outdir, "frame_start.png"))

        header = ["t","hero_r","hero_c","enemy_count","junk_cells","status","bfs_expanded","path_len"]
        rows = []
        p0, m0 = Planner.bfs_path(world, ent.hero, ent.goal, set(ent.enemies), keepout=self.keepout)
        rows.append([0, ent.hero[0], ent.hero[1], len(ent.enemies), world.count_junk(), "running",
                     int(m0.get("expanded",0)), len(p0) if p0 else 0])

        status = "running"; t = 0
        while t < max_steps and status == "running":
            ent.hero, ent.enemies, status, meta, plen = self.sim_step(world, ent, ent.goal)
            t += 1
            rows.append([t, ent.hero[0], ent.hero[1], len(ent.enemies), world.count_junk(),
                         status, int(meta.get("expanded",0)), int(plen)])

            if self.render_every and (t % self.render_every == 0):
                cur_path, _ = Planner.bfs_path(world, ent.hero, ent.goal, set(ent.enemies), keepout=self.keepout)
                world.draw(ent.goal, ent.hero, ent.enemies, path=cur_path,
                           save_path=os.path.join(outdir, f"frame_{t:04d}.png"))

        final_path, _ = Planner.bfs_path(world, ent.hero, ent.goal, set(ent.enemies), keepout=self.keepout)
        world.draw(ent.goal, ent.hero, ent.enemies, path=final_path, save_path=os.path.join(outdir, "frame_final.png"))

        Logger.export_log_csv(outdir, rows, header)
        Logger.export_config_json(outdir, {
            "grid_size": int(world.n),
            "target_fill": float(world.target_fill),
            "num_enemies": int(NUM_ENEMIES),
            "seed": SEED,
            "max_steps": max_steps,
            "render_every": self.render_every,
            "result": status,
            "steps_taken": t
        })

        # optional gif
        try:
            import imageio
            frames = sorted(glob.glob(os.path.join(outdir, "frame_*.png")))
            if len(frames) >= 2:
                imgs = [imageio.v2.imread(f) for f in frames]
                imageio.mimsave(os.path.join(outdir, "run.gif"), imgs, duration=0.06)
                print("[gif] wrote", os.path.join(outdir, "run.gif"))
        except Exception as e:
            print("[gif] skipped:", e)

        print(f"[proof] result={status} steps={t} dir={outdir}")
        return outdir, status, t

# convenience
def quick_summary(outdir):
    r = Logger.load_run_csv(outdir)
    steps = int(r[-1]["t"])
    enemies_end = int(r[-1]["enemy_count"])
    junk_end = int(r[-1]["junk_cells"])
    exps = [int(x.get("bfs_expanded", 0) or 0) for x in r]
    avg_exp = sum(exps) / len(exps) if exps else 0.0
    print(f"steps={steps}, enemies_end={enemies_end}, junk_end={junk_end}, avg_bfs_expanded={avg_exp:.1f}")

def plot_from_outdir(outdir):
    Logger.plots(outdir)

def plot_latest_run():
    runs = sorted(glob.glob("flatland_proof_*"))
    if not runs:
        print("No flatland_proof_* folder found.")
        return
    outdir = runs[-1]
    print("Plotting from:", outdir)
    plot_from_outdir(outdir)

