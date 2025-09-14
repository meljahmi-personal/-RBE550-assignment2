# RBE-550 HW2: Flatland Motion Planning Simulation

This repository contains the implementation and report artifacts for **RBE-550 HW2 (Flatland)**.  
The project simulates a hero robot navigating a 64×64 world with tetromino-shaped obstacles,  
10 enemies that chase greedily, and a BFS planner guiding the hero to its goal.

---

## Project Structure
```
src/
 ├── world.py        # GridWorld environment and rendering
 ├── entities.py     # Hero, goal, enemies
 ├── enemies.py      # Enemy behavior
 ├── planner.py      # BFS planner
 ├── simulator.py    # Simulation loop, proof runs, GIF export
 ├── main.py         # Entry point (demo and proof runs)
 ├── utils.py        # Constants and helper utilities
 ├── logger.py       # Logging to CSV + JSON
 └── ...
```

---

## Setup Instructions

### 1. Clone or copy the files
If working on a new machine:
```bash
git clone <your_repo_url> RBE550_HW2
cd RBE550_HW2/src
```
Or copy all the provided `*.py` files into a `src/` folder.

### 2. Create and activate a virtual environment
Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python33 -m venv venv
.env\Scripts\Activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:
```bash
pip install numpy matplotlib imageio
```

---

## Running the Simulation

### Quick demo (4 steps, shows hero/enemies moving)
```bash
python3 main.py --steps 4 --show
```

### Full proof run (200 steps, render every 4 frames)
```bash
python3 main.py --steps 200 --render-every 4 --show
```

---

## Outputs

When running `proof_run`, results are saved in a timestamped folder:
```
flatland_proof_YYYYMMDD_HHMMSS/
 ├── frame_start.png      # Initial state
 ├── frame_final.png      # Final state (hero on goal)
 ├── frame_00XX.png       # Intermediate frames
 ├── run.gif              # Animated run
 ├── run_log.csv          # Per-step log (hero, enemies, BFS expansions, etc.)
 ├── run_config.json      # Config and summary (steps, result, seed, etc.)
 ├── plot_expansions.png  # BFS expansions vs. time
 └── plot_enemies_junk.png# Enemies vs junk over time
```

Example terminal output:
```
[gif] wrote flatland_proof_20250914_021445/run.gif
[proof] result=win steps=98 dir=flatland_proof_20250914_021445
Artifacts written to: flatland_proof_20250914_021445
steps=98, enemies_end=0, junk_end=10, avg_bfs_expanded=1990.0
```

---

## Notes for Report
- Use **frame_start.png**, a mid-run frame, and **frame_final.png** as figures.  
- Include **plot_expansions.png** and **plot_enemies_junk.png** for analysis.  
- Reference **run.gif** as the animation proof of the hero defeating all enemies and reaching the goal.

---

## Requirements
- Python 3.8+ (tested with 3.10/3.11)
- numpy
- matplotlib
- imageio

## GitHub repository:
---

https://github.com/meljahmi-personal/-RBE550-assignment2.git

---

