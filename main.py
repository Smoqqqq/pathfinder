import random
from position import Position
from gridPrinter import GridPrinter
# from dijkstra import Dijkstra
from dijkstra_cuda import findPath
import time
import numpy as np

TABLE_SIZE = 30
START_POS = np.int64(0)
END_POS = np.int64(TABLE_SIZE-1)

grid = []

for i in range(0, TABLE_SIZE):
    grid.insert(i, [])
    for j in range(0, TABLE_SIZE):
        if (random.randint(0, 10) == 0) or (i == START_POS and j == START_POS) or (i == END_POS and j == END_POS):
            grid[i].insert(j, 'city')
            continue
        
        grid[i].insert(j, 'void')
        
# GridPrinter().display(grid)

start = time.time()
path = findPath(np.array(grid), (0, 0), (TABLE_SIZE-1, TABLE_SIZE-1))
print(path)

for position in path:
    print(position)
    # Highlight the neighbor as visited
    grid[position.x][position.y] = 'highlight_city'
    
end = time.time()

# if None != shortest_distance:
#     print("Shortest distance:", shortest_distance)
#     if TABLE_SIZE < 100:
#         GridPrinter().display(grid)
# else:
#     print('No path found')

print("Temps écoulé: ", str(round(end - start, 2)), " secondes")