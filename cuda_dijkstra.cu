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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= grid_size || y >= grid_size)
        return;

    int local_grid_size = grid_size;

    // Allocate memory for distances, previous, and visited arrays on device
    double **distances = allocateDeviceDoubleArray(local_grid_size, local_grid_size);
    Position **previous = allocateDevicePositionArray(local_grid_size, local_grid_size);
    bool **visited = allocateDeviceBoolArray(local_grid_size, local_grid_size);

    // Check if memory allocation failed
    if (distances == NULL || previous == NULL || visited == NULL) {
        *path_found = false;
        return;
    }

    // Initialize distances, previous, and visited arrays
    for (int i = 0; i < local_grid_size; i++) {
        for (int j = 0; j < local_grid_size; j++) {
            distances[i][j] = INFINITY;
            visited[i][j] = false;
        }
    }

    // Initialize distances for start position
    distances[start.x][start.y] = 0;

    // Initialize queue for breadth-first search
    QueueNode *queue = (QueueNode *)malloc(local_grid_size * local_grid_size * sizeof(QueueNode));
    if (queue == NULL) {
        *path_found = false;
        freeDeviceDoubleArray(distances, local_grid_size);
        freeDevicePositionArray(previous, local_grid_size);
        freeDeviceBoolArray(visited, local_grid_size);
        return;
    }
    int queue_size = 0;

    // Add start position to the queue
    QueueNode temp;
    temp.position = start;
    temp.cost = 0;
    queue[queue_size++] = temp;

    // Perform breadth-first search
    while (queue_size > 0) {
        // Dequeue the front node
        QueueNode cur_node = queue[0];
        for (int i = 0; i < queue_size - 1; i++) {
            queue[i] = queue[i + 1];
        }
        queue_size--;

        // Current position
        Position cur_position = cur_node.position;

        // Mark current position as visited
        visited[cur_position.x][cur_position.y] = true;

        // Explore neighboring positions
        for (int dx = -MAX_REACH; dx <= MAX_REACH; dx++) {
            for (int dy = -MAX_REACH; dy <= MAX_REACH; dy++) {
                int new_x = cur_position.x + dx;
                int new_y = cur_position.y + dy;

                // Check if new position is within bounds and not visited
                if (new_x >= 0 && new_x < local_grid_size && new_y >= 0 && new_y < local_grid_size &&
                    grid[new_x][new_y] == '.' && !visited[new_x][new_y]) {
                    double new_distance = cur_node.cost + sqrt((double)(dx * dx + dy * dy));
                    if (new_distance < distances[new_x][new_y]) {
                        distances[new_x][new_y] = new_distance;
                        QueueNode new_node;
                        new_node.position.x = new_x;
                        new_node.position.y = new_y;
                        new_node.cost = new_distance;
                        queue[queue_size++] = new_node;
                        previous[new_x][new_y] = cur_position;
                    }
                }
            }
        }
    }

    // Check if the end position is reachable
    *path_found = visited[end.x][end.y];

    // Free memory
    free(queue);
    freeDeviceDoubleArray(distances, local_grid_size);
    freeDevicePositionArray(previous, local_grid_size);
    freeDeviceBoolArray(visited, local_grid_size);
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
        char *row_device;
        cudaMalloc(&row_device, grid_size * sizeof(char));
        cudaMemcpy(&grid_device[i], &row_device, sizeof(char*), cudaMemcpyHostToDevice);
        cudaMemcpy(row_device, grid_host[i], grid_size * sizeof(char), cudaMemcpyHostToDevice);
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
    cudaMemcpy(&path_found, path_found_device, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(path_found_device);

    if (!path_found) {
        printf("No path found.\n");
    } else {
        char **grid_host_result = (char **)malloc(grid_size * sizeof(char *));
        for (int i = 0; i < grid_size; i++) {
            grid_host_result[i] = (char *)malloc(grid_size * sizeof(char));
            cudaMemcpy(grid_host_result[i], grid_device[i], grid_size * sizeof(char), cudaMemcpyDeviceToHost);
        }

        if (grid_size < 100 && grid_host_result[0][0] == 'X' && grid_host_result[grid_size - 1][grid_size - 1] == 'X') {
            printf("\nGrid with Shortest Path:\n");
            for (int i = 0; i < grid_size; i++) {
                for (int j = 0; j < grid_size; j++) {
                    if (grid_host_result[i][j] == 'X')
                        printf("\033[0;32mX \033[0m");
                    else
                        printf("%c ", grid_host_result[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }

        for (int i = 0; i < grid_size; i++) {
            free(grid_host_result[i]);
        }
        free(grid_host_result);
    }

    for (int i = 0; i < grid_size; i++) {
        free(grid_host[i]);
    }
    free(grid_host);
    cudaFree(grid_device);

    clock_t end_time = clock();
    printf("execution time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}
