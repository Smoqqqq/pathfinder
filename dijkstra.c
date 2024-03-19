#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define MAX_TABLE_SIZE 1000000
#define MAX_REACH 4

typedef struct {
    int x;
    int y;
} Position;

typedef struct {
    Position position;
    double cost;
} QueueNode;

char **grid;
int grid_size = MAX_TABLE_SIZE;

void displayGrid(char **grid, int grid_size) {
    printf("\n");
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (grid[i][j] == 'X')
                printf("\033[0;32mX \033[0m");
            else
                printf("%c ", grid[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int findPath(char **grid, int grid_size, Position start, Position end) {
    double **distances = (double **)malloc(grid_size * sizeof(double *));
    Position **previous = (Position **)malloc(grid_size * sizeof(Position *));
    bool **visited = (bool **)malloc(grid_size * sizeof(bool *));
    QueueNode *queue = (QueueNode *)malloc(grid_size * grid_size * sizeof(QueueNode));
    Position *path = (Position *)malloc(grid_size * grid_size * sizeof(Position));

    for (int i = 0; i < grid_size; i++) {
        distances[i] = (double *)malloc(grid_size * sizeof(double));
        previous[i] = (Position *)malloc(grid_size * sizeof(Position));
        visited[i] = (bool *)malloc(grid_size * sizeof(bool));
        for (int j = 0; j < grid_size; j++) {
            distances[i][j] = INFINITY;
            visited[i][j] = false;
        }
    }

    distances[start.x][start.y] = 0;

    int queue_size = 0;

    queue[queue_size++] = (QueueNode){start, 0};

    while (queue_size > 0) {
        int min_index = 0;
        double min_cost = queue[0].cost;
        for (int i = 1; i < queue_size; i++) {
            if (queue[i].cost < min_cost) {
                min_index = i;
                min_cost = queue[i].cost;
            }
        }

        QueueNode cur_node = queue[min_index];
        queue[min_index] = queue[--queue_size];

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
                        grid[x][y] = 'C';
                        queue[queue_size++] = (QueueNode){(Position){x, y}, new_distance};
                        previous[x][y] = cur_position;
                    }
                }
            }
        }
    }

    int path_size = 0;
    Position cur_position = end;
    while (!(cur_position.x == start.x && cur_position.y == start.y)) {
        path[path_size++] = cur_position;
        cur_position = previous[cur_position.x][cur_position.y];
    }
    path[path_size++] = start;

    if (path_size < 3 || path[0].x != end.x || path[0].y != end.y || path[path_size - 1].x != start.x || path[path_size - 1].y != start.y) {
        printf("No path found.\n");
        return false;
    }

    printf("Shortest path:\n");
    for (int i = path_size - 1; i >= 0; i--) {
        printf("(%d, %d), ", path[i].x, path[i].y);
        grid[path[i].x][path[i].y] = 'X';
    }
    printf("\n\n");

    for (int i = 0; i < grid_size; i++) {
        free(distances[i]);
        free(previous[i]);
        free(visited[i]);
    }
    free(distances);
    free(previous);
    free(visited);
    free(queue);
    free(path);

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <grid_size>\n", argv[0]);
        return 1;
    }

    int requested_size = atoi(argv[1]);
    if (requested_size <= 0 || requested_size > MAX_TABLE_SIZE) {
        printf("Invalid grid size. Please choose a value between 1 and %d\n", MAX_TABLE_SIZE);
        return 1;
    }

    grid_size = requested_size;

    srand(time(NULL));

    grid = (char **)malloc(grid_size * sizeof(char *));
    for (int i = 0; i < grid_size; i++) {
        grid[i] = (char *)malloc(grid_size * sizeof(char));
    }

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (rand() % 5 == 0 || (i == 0 && j == 0) || (i == grid_size - 1 && j == grid_size - 1))
                grid[i][j] = 'c';
            else
                grid[i][j] = '.';
        }
    }

    clock_t start_time = clock();
    int path_found = findPath(grid, grid_size, (Position){0, 0}, (Position){grid_size - 1, grid_size - 1});
    clock_t end_time = clock();

    if (path_found && grid_size < 100 && grid[0][0] == 'X' && grid[grid_size - 1][grid_size - 1] == 'X') {
        printf("\nGrid with Shortest Path:\n");
        displayGrid(grid, grid_size);
    }

    printf("execution time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    for (int i = 0; i < grid_size; i++) {
        free(grid[i]);
    }
    free(grid);

    return 0;
}
