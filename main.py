import random
from position import Position
from gridPrinter import GridPrinter
from dijkstra import Dijkstra

TABLE_SIZE = 30
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
GridPrinter().display(grid)

shortest_distance = dijkstra.findPath(grid, Position(START_POS, START_POS), Position(END_POS, END_POS))

if None != shortest_distance:
    print("Shortest distance:", shortest_distance)
else:
    print('No path found')

GridPrinter().display(dijkstra.grid)