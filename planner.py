# planner.py
import numpy as np
from collections import deque
from utils import FREE, NEI_ORDER

class Planner:
    @staticmethod
    def bfs_path(world, start, goal, enemy_set, keepout=0):
        """
        4-connected BFS on FREE cells; OBST/JUNK/enemies are blocked.
        keepout>0 also blocks 4-neighbors of enemies.
        Returns (path, {"expanded": k})
        """
        n = world.n
        if start == goal:
            return [start], {"expanded": 0}

        blocked = set(map(tuple, np.argwhere(world.grid != FREE)))
        blocked.discard(start)
        blocked |= set(enemy_set)

        if keepout > 0:
            for er, ec in enemy_set:
                for dr, dc in NEI_ORDER:
                    rr, cc = er+dr, ec+dc
                    if 0 <= rr < n and 0 <= cc < n:
                        blocked.add((rr, cc))

        q = deque([start]); parent = {start: None}; expanded = 0
        while q:
            r, c = q.popleft(); expanded += 1
            for dr, dc in NEI_ORDER:
                rr, cc = r+dr, c+dc
                if not (0 <= rr < n and 0 <= cc < n): continue
                nxt = (rr, cc)
                if nxt in parent: continue
                if nxt in blocked and nxt != goal: continue
                parent[nxt] = (r, c)
                if nxt == goal:
                    path = [nxt]
                    while path[-1] is not None:
                        path.append(parent[path[-1]])
                    path.pop(); path.reverse()
                    return path, {"expanded": expanded}
                q.append(nxt)
        return None, {"expanded": expanded}

