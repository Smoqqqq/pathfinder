import random
from position import Position
from gridPrinter import GridPrinter
from dijkstra import Dijkstra
import time

TABLE_SIZE = 3000
START_POS = 0;
END_POS = TABLE_SIZE-1

grid = []

for i in range(0, TABLE_SIZE):
    grid.insert(i, [])
    for j in range(0, TABLE_SIZE):
        if (random.randint(0, 10) == 0) or (i == START_POS and j == START_POS) or (i == END_POS and j == END_POS):
            grid[i].insert(j, 'city')
            continue
        
        grid[i].insert(j, 'void')

dijkstra = Dijkstra()
# GridPrinter().display(grid)

start = time.time()
shortest_distance = dijkstra.findPath(grid, Position(START_POS, START_POS), Position(END_POS, END_POS))
end = time.time()

if None != shortest_distance:
    print("Shortest distance:", shortest_distance)
    if TABLE_SIZE < 100:
        GridPrinter().display(dijkstra.grid)
else:
    print('No path found')

print("Temps écoulé: ", str(round(end - start, 2)), " secondes")