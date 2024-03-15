#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define MAX_REACH 4
#define MAX_BLOCK_DIM 1024
#define MAX_GRID_DIM_X 65535
#define MAX_GRID_DIM_Y 65535

typedef struct {
    int x;
    int y;
} Position;

__device__ bool isValid(int x, int y, int grid_size) {
    return (x >= 0 && x < grid_size && y >= 0 && y < grid_size);
}

__global__ void findPath(char *grid, int grid_size, Position start, Position end, bool *visited, Position *previous, Position *path, int *path_size) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_x < grid_size && tid_y < grid_size) {
        int idx = tid_x * grid_size + tid_y;

        if (idx == start.x * grid_size + start.y) {
            visited[idx] = true;
            previous[idx].x = -1;  // Initialize with invalid value
            previous[idx].y = -1;
        }

        if (grid[idx] == 'c') {
            for (int dx = -MAX_REACH; dx <= MAX_REACH; dx++) {
                for (int dy = -MAX_REACH; dy <= MAX_REACH; dy++) {
                    int x = tid_x + dx;
                    int y = tid_y + dy;

                    if (isValid(x, y, grid_size)) {
                        int neighbor_idx = x * grid_size + y;

                        if (!visited[neighbor_idx] && (grid[neighbor_idx] == 'c' || grid[neighbor_idx] == 'C')) {
                            visited[neighbor_idx] = true;
                            previous[neighbor_idx].x = tid_x;
                            previous[neighbor_idx].y = tid_y;
                        }
                    }
                }
            }
        }

        if (idx == end.x * grid_size + end.y && visited[idx]) {
            // Trace back to find the path
            Position cur_position = end;
            int count = 0;

            while (!(cur_position.x == start.x && cur_position.y == start.y) && count < grid_size * grid_size) {
                path[count++] = cur_position;
                cur_position = previous[cur_position.x * grid_size + cur_position.y];
            }

            if (count < grid_size * grid_size) {
                path[count++] = start;
            }

            *path_size = count;
        }
    }
}

void displayGrid(char *grid, int grid_size, Position *path, int path_size) {
    printf("\n");
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            bool isPath = false;
            for (int k = 0; k < path_size; k++) {
                if (path[k].x == i && path[k].y == j) {
                    isPath = true;
                    break;
                }
            }
            if (isPath)
                printf("\033[0;32mX \033[0m");
            else if (grid[i * grid_size + j] == 'X')
                printf("\033[0;33mX \033[0m");
            else
                printf("%c ", grid[i * grid_size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <grid_size>\n", argv[0]);
        return 1;
    }

    int requested_size = atoi(argv[1]);
    if (requested_size <= 0) {
        printf("Invalid grid size.\n");
        return 1;
    }

    int grid_size = requested_size;

    srand(time(NULL));

    // Allocate memory for grid on the host
    char *host_grid = (char*)malloc(grid_size * grid_size * sizeof(char));
    if (host_grid == NULL) {
        fprintf(stderr, "Failed to allocate memory for grid on the host.\n");
        return 1;
    }

    // Initialize grid on the host
    for (int i = 0; i < grid_size * grid_size; i++) {
        host_grid[i] = (rand() % 5 == 0 || i == 0 || i == grid_size * grid_size - 1) ? 'c' : '.';
    }

    // Allocate memory for visited on the host
    bool *host_visited = (bool*)malloc(grid_size * grid_size * sizeof(bool));
    if (host_visited == NULL) {
        fprintf(stderr, "Failed to allocate memory for visited on the host.\n");
        free(host_grid);
        return 1;
    }

    // Initialize visited on the host
    for (int i = 0; i < grid_size * grid_size; i++) {
        host_visited[i] = false;
    }

    // Allocate memory for previous on the host
    Position *host_previous = (Position*)malloc(grid_size * grid_size * sizeof(Position));
    if (host_previous == NULL) {
        fprintf(stderr, "Failed to allocate memory for previous on the host.\n");
        free(host_grid);
        free(host_visited);
        return 1;
    }

    // Allocate memory for path on the host
    Position *host_path = (Position*)malloc(grid_size * grid_size * sizeof(Position));
    if (host_path == NULL) {
        fprintf(stderr, "Failed to allocate memory for path on the host.\n");
        free(host_grid);
        free(host_visited);
        free(host_previous);
        return 1;
    }

    // Allocate memory for path_size on the host
    int *host_path_size = (int*)malloc(sizeof(int));
    if (host_path_size == NULL) {
        fprintf(stderr, "Failed to allocate memory for path_size on the host.\n");
        free(host_grid);
        free(host_visited);
        free(host_previous);
        free(host_path);
        return 1;
    }
    *host_path_size = 0;

    // Transfer data from host to GPU
    char *grid;
    gpuErrorcheck(cudaMalloc(&grid, grid_size * grid_size * sizeof(char)));
    gpuErrorcheck(cudaMemcpy(grid, host_grid, grid_size * grid_size * sizeof(char), cudaMemcpyHostToDevice));

    bool *visited;
    gpuErrorcheck(cudaMalloc(&visited, grid_size * grid_size * sizeof(bool)));
    gpuErrorcheck(cudaMemcpy(visited, host_visited, grid_size * grid_size * sizeof(bool), cudaMemcpyHostToDevice));

    Position *previous;
    gpuErrorcheck(cudaMalloc(&previous, grid_size * grid_size * sizeof(Position)));
    gpuErrorcheck(cudaMemcpy(previous, host_previous, grid_size * grid_size * sizeof(Position), cudaMemcpyHostToDevice));

    Position *path;
    gpuErrorcheck(cudaMalloc(&path, grid_size * grid_size * sizeof(Position)));
    gpuErrorcheck(cudaMemcpy(path, host_path, grid_size * grid_size * sizeof(Position), cudaMemcpyHostToDevice));

    int *path_size;
    gpuErrorcheck(cudaMalloc(&path_size, sizeof(int)));
    gpuErrorcheck(cudaMemcpy(path_size, host_path_size, sizeof(int), cudaMemcpyHostToDevice));

    // Define start and end positions
    Position start = {0, 0};
    Position end = {grid_size - 1, grid_size - 1};

    // Calculate block and grid dimensions
    int block_dim_x = min(MAX_BLOCK_DIM, grid_size);
    int block_dim_y = min(MAX_BLOCK_DIM / block_dim_x, grid_size);
    dim3 blockSize(block_dim_x, block_dim_y);

    int grid_dim_x = min(MAX_GRID_DIM_X, (grid_size + blockSize.x - 1) / blockSize.x);
    int grid_dim_y = min(MAX_GRID_DIM_Y, (grid_size + blockSize.y - 1) / blockSize.y);
    dim3 gridSize(grid_dim_x, grid_dim_y);

    // Call the kernel
    findPath<<<gridSize, blockSize>>>(grid, grid_size, start, end, visited, previous, path, path_size);
    gpuErrorcheck(cudaPeekAtLastError());
    gpuErrorcheck(cudaDeviceSynchronize());

    // Transfer data from GPU to host
    gpuErrorcheck(cudaMemcpy(host_grid, grid, grid_size * grid_size * sizeof(char), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(host_visited, visited, grid_size * grid_size * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(host_previous, previous, grid_size * grid_size * sizeof(Position), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(host_path, path, grid_size * grid_size * sizeof(Position), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(host_path_size, path_size, sizeof(int), cudaMemcpyDeviceToHost));

    // Display the results
    printf("Path size: %d\n", *host_path_size);
    if (*host_path_size > 0) {
        printf("Path found:\n");
    } else {
        printf("No valid path\n");
    }
    displayGrid(host_grid, grid_size, host_path, *host_path_size);

    // Free host memory
    free(host_grid);
    free(host_visited);
    free(host_previous);
    free(host_path);
    free(host_path_size);

    // Free GPU memory
    gpuErrorcheck(cudaFree(grid));
    gpuErrorcheck(cudaFree(visited));
    gpuErrorcheck(cudaFree(previous));
    gpuErrorcheck(cudaFree(path));
    gpuErrorcheck(cudaFree(path_size));

    return 0;
}
