from position import Position
import heapq
import time
import numpy as np
import random
from gridPrinter import GridPrinter

class BgColor:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_COLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Neighbour:
    def __init__(self, position: Position, distance: int) -> None:
        self.position = position
        self.distance = distance
        
class GridPrinter:
    def __init__(self) -> None:
        self.paddSize = 5
    
    def display(self, grid: dict) -> None:
        buffer = ''
        
        if len(grid) > 50:
            self.paddSize = 3
            
        if len(grid) > 100:
            self.paddSize = 1
        
        if len(grid) < 100:
            buffer = '\n' + ''.ljust(self.paddSize, ' ')
            for i in range(0, len(grid)):
                buffer = buffer + str(i).ljust(self.paddSize, ' ')
            
        print(buffer)
        
        for i in range(0, len(grid)):
            buffer = ''
            for j in range(0, len(grid)):
                if grid[i][j] == 'city':
                    buffer = buffer + str('◼').ljust(self.paddSize, ' ')
                elif grid[i][j] == 'highlight_city':
                    buffer = buffer + BgColor.GREEN + str('◼').ljust(self.paddSize, ' ') + BgColor.END_COLOR
                elif grid[i][j] == 'void':
                    buffer = buffer + str('.').ljust(self.paddSize, ' ')
                else:
                    buffer = buffer + BgColor.GREEN + str(grid[i][j]).ljust(self.paddSize, ' ') + BgColor.END_COLOR
            
            if len(grid) < 100:
                buffer = buffer + '\n'
            
            print(str(i).ljust(self.paddSize, ' ') + buffer)
                
        print('\n◼ ville')
        print('_ vide\n')

class Dijkstra:
    def __init__(self) -> None:
        self.grid = []
    
    def findPath(self, grid: list, start: Position, end: Position) -> None:
        self.grid = grid

        # Initialize a dictionary to store distances from start to each position
        distances = {start: 0}
        # Initialize a dictionary to store the previous position for each position in the shortest path
        previous = {}
        # Initialize a priority queue to explore positions with the smallest distance first
        queue = [(0, start)]  # (distance, position)
        # Initialize a set to keep track of visited positions
        visited = set()

        # Dijkstra's algorithm
        while queue:
            current_distance, current_position = heapq.heappop(queue)
            
            # If we reached the end position, reconstruct the shortest path and print it
            if current_position == end:
                path = [end]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                path.reverse()
                print("Shortest path:")
                
                for position in path:
                    print(position)
                    # Highlight the neighbor as visited
                    self.grid[position.x][position.y] = 'highlight_city'
                    
                return current_distance

            # Mark the current position as visited
            visited.add(current_position)

            # Explore neighbors of the current position
            for neighbour in self.findNeighbours(current_position):
                # Skip visited neighbors
                if neighbour.position in visited:
                    continue

                # Calculate the distance from the start to the neighbor via the current position
                distance_to_neighbour = current_distance + neighbour.distance

                # If the new distance is shorter than the previous distance to the neighbor, update it
                if distance_to_neighbour < distances.get(neighbour.position, float('inf')):
                    distances[neighbour.position] = distance_to_neighbour
                    previous[neighbour.position] = current_position
                    heapq.heappush(queue, (distance_to_neighbour, neighbour.position))
                    
    def findNeighbours(self, position: Position):
        neighbours = []

        # Define the bounds of the grid
        min_x = max(0, position.x - 4)
        max_x = min(len(self.grid) - 1, position.x + 4)
        min_y = max(0, position.y - 4)
        max_y = min(len(self.grid[0]) - 1, position.y + 4)

        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                if self.grid[i][j] == 'city':
                    # Exclude the current position
                    if i != position.x or j != position.y:
                        neighbours.append(Neighbour(Position(i, j), abs(position.x - i) + abs(position.y - j)))

        return neighbours


TABLE_SIZE = 20
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
dijkstra = Dijkstra()
shortest_distance = dijkstra.findPath(grid, Position(0, 0), Position(TABLE_SIZE-1, TABLE_SIZE-1))
    
end = time.time()

if None != shortest_distance:
    print("Shortest distance:", shortest_distance)
else:
    print('No path found')

if TABLE_SIZE < 100:
    GridPrinter().display(grid)

print("Temps écoulé: ", str(round(end - start, 2)), " secondes")