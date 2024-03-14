#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define TABLE_SIZE 200
#define MAX_REACH 4

typedef struct {
    int x;
    int y;
} Position;

typedef struct {
    Position position;
    double cost;
} QueueNode;

char grid[TABLE_SIZE][TABLE_SIZE];
int grid_size = TABLE_SIZE;

void displayGrid(char grid[TABLE_SIZE][TABLE_SIZE]) {
    printf("\n");
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            printf("%c ", grid[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void findPath(char grid[TABLE_SIZE][TABLE_SIZE], Position start, Position end) {
    double distances[TABLE_SIZE][TABLE_SIZE];
    Position previous[TABLE_SIZE][TABLE_SIZE];

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            distances[i][j] = INFINITY;
        }
    }

    distances[start.x][start.y] = 0;

    bool visited[TABLE_SIZE][TABLE_SIZE] = {{false}};

    QueueNode queue[TABLE_SIZE * TABLE_SIZE];
    int queue_size = 0;

    queue[queue_size++] = (QueueNode){start, 0};

    while (queue_size > 0) {
        // Find the node with the minimum cost
        int min_index = 0;
        double min_cost = queue[0].cost;
        for (int i = 1; i < queue_size; i++) {
            if (queue[i].cost < min_cost) {
                min_index = i;
                min_cost = queue[i].cost;
            }
        }

        QueueNode cur_node = queue[min_index];
        queue[min_index] = queue[--queue_size]; // Remove the node from the queue

        Position cur_position = cur_node.position;

        if (cur_position.x == end.x && cur_position.y == end.y)
            break;

        if (visited[cur_position.x][cur_position.y])
            continue;

        visited[cur_position.x][cur_position.y] = true;

        for (int dx = -MAX_REACH; dx <= MAX_REACH; dx++) {
            for (int dy = -MAX_REACH; dy <= MAX_REACH; dy++) {
                int x = cur_position.x + dx;
                int y = cur_position.y + dy;

                if (x < 0 || x >= grid_size || y < 0 || y >= grid_size)
                    continue;

                if (grid[x][y] == 'c' && !visited[x][y]) {
                    double new_distance = cur_node.cost + sqrt(dx * dx + dy * dy);
                    if (new_distance < distances[x][y]) {
                        distances[x][y] = new_distance;
                        grid[x][y] = 'C'; // Marking the cell as visited
                        queue[queue_size++] = (QueueNode){(Position){x, y}, new_distance};
                        previous[x][y] = cur_position;
                    }
                }
            }
        }
    }

    // Retrieve the path
    Position path[TABLE_SIZE * TABLE_SIZE];
    int path_size = 0;
    Position cur_position = end;
    while (!(cur_position.x == start.x && cur_position.y == start.y)) {
        path[path_size++] = cur_position;
        cur_position = previous[cur_position.x][cur_position.y];
    }
    path[path_size++] = start;

    // Print the path
    printf("Shortest path:\n");
    for (int i = path_size - 1; i >= 0; i--) {
        printf("(%d, %d)\n", path[i].x, path[i].y);
        grid[path[i].x][path[i].y] = 'X'; // Marking the path on the grid
    }
}

int main() {
    srand(time(NULL));

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (rand() % 11 == 0 || (i == 0 && j == 0) || (i == TABLE_SIZE - 1 && j == TABLE_SIZE - 1))
                grid[i][j] = 'c';
            else
                grid[i][j] = '.';
        }
    }

    printf("Initial Grid:\n");
    displayGrid(grid);

    findPath(grid, (Position){0, 0}, (Position){TABLE_SIZE - 1, TABLE_SIZE - 1});

    printf("\nGrid with Shortest Path:\n");
    displayGrid(grid);

    return 0;
}
