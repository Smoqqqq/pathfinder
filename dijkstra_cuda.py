from numba import jit
from numba.typed import List
import heapq
import numpy as np

def findPath(grid, start, end):
    x_max, y_max = grid.shape
    dxy = List()
    dxy.append((0, 1))
    dxy.append((1, 0))
    dxy.append((0, -1))  # Add movement option: up
    dxy.append((-1, 0))  # Add movement option: left

    distances = np.full_like(grid, np.inf)
    distances[start[0], start[1]] = grid[start[0], start[1]]
    visited = np.zeros(grid.shape, dtype=bool)
    transitions = np.zeros((x_max, y_max, 2), dtype=np.int32)

    queue = List()
    queue.append((grid[start[0], start[1]], (start[0], start[1])))

    return dijkstra_run(grid, distances, queue, visited, transitions, dxy, start, end)


@jit(nopython=True)
def dijkstra_run(grid, distances, queue, visited, transitions, dxy, start, end):
    x_max, y_max = grid.shape

    while queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(queue)

        if (cur_x, cur_y) == end:
            break

        if visited[cur_x, cur_y]:
            continue

        for dx, dy in dxy:
            x, y = cur_x + dx, cur_y + dy
            if x < 0 or x >= x_max or y < 0 or y >= y_max:
                continue
            if not visited[x, y] and grid[x][y] == 'city':
                if grid[x, y] + distances[cur_x, cur_y] < distances[x, y]:
                    distances[x, y] = grid[x, y] + distances[cur_x, cur_y]
                    heuristic = distances[x, y]
                    heapq.heappush(queue, (heuristic, (x, y)))
                    transitions[x, y, 0] = cur_x
                    transitions[x, y, 1] = cur_y

        visited[cur_x, cur_y] = True

    # Retrieve the path
    path = List()
    cur_x, cur_y = end
    path.append((cur_x, cur_y))
    while (cur_x, cur_y) != (start[0], start[1]):
        cur_x, cur_y = transitions[cur_x, cur_y]
        path.append((cur_x, cur_y))
    return path[::-1]