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

__device__ double** allocateDeviceDoubleArray(int rows, int cols) {
    double **array;
    cudaMalloc(&array, rows * sizeof(double *));
    for (int i = 0; i < rows; ++i) {
        cudaMalloc(&(array[i]), cols * sizeof(double));
    }
    return array;
}

__device__ void freeDeviceDoubleArray(double** ptr, int rows) {
    for (int i = 0; i < rows; ++i) {
        cudaFree(ptr[i]);
    }
    cudaFree(ptr);
}

__device__ void freeDevicePositionArray(Position** ptr, int rows) {
    for (int i = 0; i < rows; ++i) {
        cudaFree(ptr[i]);
    }
    cudaFree(ptr);
}

__device__ void freeDeviceBoolArray(bool** ptr, int rows) {
    for (int i = 0; i < rows; ++i) {
        cudaFree(ptr[i]);
    }
    cudaFree(ptr);
}

__device__ Position** allocateDevicePositionArray(int rows, int cols) {
    Position **array;
    cudaMalloc(&array, rows * sizeof(Position *));
    for (int i = 0; i < rows; ++i) {
        cudaMalloc(&(array[i]), cols * sizeof(Position));
    }
    return array;
}

__device__ bool** allocateDeviceBoolArray(int rows, int cols) {
    bool **array;
    cudaMalloc(&array, rows * sizeof(bool *));
    for (int i = 0; i < rows; ++i) {
        cudaMalloc(&(array[i]), cols * sizeof(bool));
    }
    return array;
}

__device__ double deviceAbs(double val) {
    return val < 0 ? -val : val;
}

__global__ void findPathCUDA(char **grid, int grid_size, Position start, Position end, bool *path_found) {
    double **distances = allocateDeviceDoubleArray(grid_size, grid_size);
    Position **previous = allocateDevicePositionArray(grid_size, grid_size);
    bool **visited = allocateDeviceBoolArray(grid_size, grid_size);
    QueueNode *queue = (QueueNode *)malloc(grid_size * grid_size * sizeof(QueueNode));
    Position *path = (Position *)malloc(grid_size * grid_size * sizeof(Position));

    if (distances == NULL || previous == NULL || visited == NULL || queue == NULL || path == NULL) {
        *path_found = false;
        return;
    }

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            distances[i][j] = INFINITY;
            visited[i][j] = false;
        }
    }

    distances[start.x][start.y] = 0;

    int queue_size = 0;

    queue[queue_size].position = start;
    queue[queue_size].cost = 0;
    queue_size++;

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
                    double new_distance = cur_node.cost + sqrtf(dx * dx + dy * dy);
                    if (new_distance < distances[x][y]) {
                        distances[x][y] = new_distance;
                        queue[queue_size].position.x = x;
                        queue[queue_size].position.y = y;
                        queue[queue_size].cost = new_distance;
                        queue_size++;
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
        *path_found = false;
        return;
    }

    *path_found = true;

    // Mark the path on the grid
    for (int i = 0; i < path_size; ++i) {
        grid[path[i].x][path[i].y] = 'X';
    }
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

    int grid_size = requested_size;

    srand(time(NULL));

    char **grid_host = (char **)malloc(grid_size * sizeof(char *));
    for (int i = 0; i < grid_size; i++) {
        grid_host[i] = (char *)malloc(grid_size * sizeof(char));
    }

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (rand() % 5 == 0 || (i == 0 && j == 0) || (i == grid_size - 1 && j == grid_size - 1))
                grid_host[i][j] = 'c';
            else
                grid_host[i][j] = '.';
        }
    }

    printf("Grid before pathfinding:\n");
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            printf("%c ", grid_host[i][j]);
        }
        printf("\n");
    }

    char **grid_device;
    cudaMalloc(&grid_device, grid_size * sizeof(char *));
    for (int i = 0; i < grid_size; i++) {
        cudaMalloc(&(grid_device[i]), grid_size * sizeof(char));
        cudaMemcpy(grid_device[i], grid_host[i], grid_size * sizeof(char), cudaMemcpyHostToDevice);
    }

    clock_t start_time = clock();
    Position start = {0, 0};
    Position end = {grid_size - 1, grid_size - 1};

    bool path_found = false;
    bool *path_found_device;
    cudaMalloc(&path_found_device, sizeof(bool));

    dim3 blockSize(16, 16);
    dim3 gridSize((grid_size + blockSize.x - 1) / blockSize.x, (grid_size + blockSize.y - 1) / blockSize.y);
    findPathCUDA<<<gridSize, blockSize>>>(grid_device, grid_size, start, end, path_found_device);
    cudaDeviceSynchronize();
    cudaMemcpy(&path_found, path_found_device, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(path_found_device);

    if (!path_found) {
        printf("No path found.\n");
    } else {
        printf("Path found.\n");
    }

    for (int i = 0; i < grid_size; i++) {
        cudaFree(grid_device[i]);
    }
    cudaFree(grid_device);

    for (int i = 0; i < grid_size; i++) {
        free(grid_host[i]);
    }
    free(grid_host);

    clock_t end_time = clock();
    printf("Execution time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}
