# logger.py
import os, csv, json, matplotlib.pyplot as plt

class Logger:
    @staticmethod
    def export_config_json(outdir, cfg: dict):
        with open(os.path.join(outdir, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @staticmethod
    def export_log_csv(outdir, rows, header):
        with open(os.path.join(outdir, "run_log.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    @staticmethod
    def load_run_csv(outdir):
        rows = []
        with open(os.path.join(outdir, "run_log.csv"), newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                row["t"] = int(row["t"])
                row["enemy_count"] = int(row["enemy_count"])
                row["junk_cells"] = int(row["junk_cells"])
                row["bfs_expanded"] = int(row.get("bfs_expanded", 0) or 0)
                row["path_len"] = int(row.get("path_len", 0) or 0)
                rows.append(row)
        return rows

    @staticmethod
    def plots(outdir):
        data = Logger.load_run_csv(outdir)
        t = [d["t"] for d in data]
        enemies = [d["enemy_count"] for d in data]
        junk = [d["junk_cells"] for d in data]
        exps = [d["bfs_expanded"] for d in data]

        if any(exps):
            plt.figure()
            plt.plot(t, exps)
            plt.xlabel("time step"); plt.ylabel("BFS nodes expanded")
            plt.title("BFS expansions vs time"); plt.grid(True, linewidth=0.3)
            plt.savefig(os.path.join(outdir, "plot_expansions.png"), dpi=200, bbox_inches="tight")
            plt.close()

        plt.figure()
        plt.plot(t, enemies, label="enemies")
        plt.plot(t, junk, label="junk")
        plt.xlabel("time step"); plt.ylabel("count")
        plt.title("Enemies and Junk vs time"); plt.grid(True, linewidth=0.3); plt.legend()
        plt.savefig(os.path.join(outdir, "plot_enemies_junk.png"), dpi=200, bbox_inches="tight")
        plt.close()

