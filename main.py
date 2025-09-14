# main.py
import numpy as np
from world import GridWorld
from entities import Entities
from planner import Planner
from simulator import Simulator, plot_from_outdir, quick_summary
import utils   # this brings in SHOW_FIGURES
import argparse


def main_demo_steps():
    world = GridWorld()
    world.generate_obstacles()
    ent = Entities(world)

    # initial snapshot with path
    path, _ = Planner.bfs_path(world, ent.hero, ent.goal, set(ent.enemies), keepout=1)
    world.draw(ent.goal, ent.hero, ent.enemies, path=path, save_path="flatland_step0.png")

    sim = Simulator(keepout=1, render_every=None)
    for t in range(5):
        ent.hero, ent.enemies, status, _, _ = sim.sim_step(world, ent, ent.goal)
        print(f"t={t+1}, enemies={len(ent.enemies)}, status={status}")
        print(world.stats_str())
        path, _ = Planner.bfs_path(world, ent.hero, ent.goal, set(ent.enemies), keepout=1)
        world.draw(ent.goal, ent.hero, ent.enemies, path=path, save_path=f"flatland_step{t+1}.png")
        if status != "running":
            break
        print(f"Blocked ratio = {np.mean(world.grid != 0):.2f}")
        
        

import argparse
import utils   # where SHOW_FIGURES lives

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively while running")
    parser.add_argument("--steps", type=int, default=500,
                        help="Maximum steps for proof run (default=500)")
    parser.add_argument("--render-every", type=int, default=50,
                        help="Save snapshot every N steps (default=50)")
    args = parser.parse_args()

    # update flag in utils
    utils.SHOW_FIGURES = args.show

    # --- small demo
    main_demo_steps()

    # --- full proof run + plots + summary
    sim = Simulator(keepout=1, render_every=args.render_every)
    outdir, status, steps = sim.proof_run(out_name="flatland_proof",
                                          max_steps=args.steps)
    print("Artifacts written to:", outdir)
    plot_from_outdir(outdir)
    quick_summary(outdir)

