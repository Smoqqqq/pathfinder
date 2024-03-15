#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define MAX_REACH 4

typedef struct {
    int x;
    int y;
} Position;

__global__ void findPath(char *grid, int grid_size, Position start, Position end, double *distances, Position *previous, bool *visited, Position *path) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_x < grid_size && tid_y < grid_size) {
        for (int i = 0; i < grid_size; i++) {
            distances[tid_x * grid_size + i] = INFINITY;
            visited[tid_x * grid_size + i] = false;
        }

        distances[start.x * grid_size + start.y] = 0;

        int queue_size = 0;

        // Allocate memory using cudaMallocManaged
        __shared__ Position queue[1024];  // Assuming maximum block size of 1024

        queue[queue_size] = start;
        queue_size++;

        while (queue_size > 0) {
            int min_index = 0;
            double min_cost = distances[queue[0].x * grid_size + queue[0].y];
            for (int i = 1; i < queue_size; i++) {
                double cost = distances[queue[i].x * grid_size + queue[i].y];
                if (cost < min_cost) {
                    min_index = i;
                    min_cost = cost;
                }
            }

            Position cur_position = queue[min_index];
            queue[min_index] = queue[--queue_size];

            if (cur_position.x == end.x && cur_position.y == end.y)
                break;

            if (visited[cur_position.x * grid_size + cur_position.y])
                continue;

            visited[cur_position.x * grid_size + cur_position.y] = true;

            for (int dx = -MAX_REACH; dx <= MAX_REACH; dx++) {
                for (int dy = -MAX_REACH; dy <= MAX_REACH; dy++) {
                    int x = cur_position.x + dx;
                    int y = cur_position.y + dy;

                    if (x < 0 || x >= grid_size || y < 0 || y >= grid_size)
                        continue;

                    if (grid[x * grid_size + y] == 'c' && !visited[x * grid_size + y]) {
                        double new_distance = distances[cur_position.x * grid_size + cur_position.y] + sqrtf(dx * dx + dy * dy);
                        if (new_distance < distances[x * grid_size + y]) {
                            distances[x * grid_size + y] = new_distance;
                            grid[x * grid_size + y] = 'C';
                            queue[queue_size].x = x;
                            queue[queue_size].y = y;
                            queue_size++;
                            previous[x * grid_size + y] = cur_position;
                        }
                    }
                }
            }
        }

        int path_size = 0;
        Position cur_position = end;
        while (!(cur_position.x == start.x && cur_position.y == start.y)) {
            path[path_size++] = cur_position;
            cur_position = previous[cur_position.x * grid_size + cur_position.y];
        }
        path[path_size++] = start;

        if (path_size < 3 || path[0].x != end.x || path[0].y != end.y || path[path_size - 1].x != start.x || path[path_size - 1].y != start.y) {
            return;
        } 
        
        printf("Shortest path:\n");
        for (int i = path_size - 1; i >= 0; i--) {
            printf("(%d, %d), ", path[i].x, path[i].y);
            grid[path[i].x * grid_size + path[i].y] = 'X';
        }
        printf("\n\n");
    } else {
        printf("Invalid thread index: (%d, %d)\n", tid_x, tid_y);
    }
}

void displayGrid(char *grid, int grid_size) {
    printf("\n");
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (grid[i * grid_size + j] == 'X')
                printf("\033[0;32mX \033[0m");
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

    // Allocate memory for the grid
    char *grid;
    gpuErrorCheck(cudaMallocManaged(&grid, grid_size * grid_size * sizeof(char)));

    // Initialize the grid
    for (int i = 0; i < grid_size * grid_size; i++) {
        grid[i] = (rand() % 5 == 0 || i == 0 || i == grid_size * grid_size - 1) ? 'c' : '.';
    }

    double *distances;
    gpuErrorCheck(cudaMallocManaged(&distances, grid_size * grid_size * sizeof(double)));

    Position *previous;
    gpuErrorCheck(cudaMallocManaged(&previous, grid_size * grid_size * sizeof(Position)));

    bool *visited;
    gpuErrorCheck(cudaMallocManaged(&visited, grid_size * grid_size * sizeof(bool)));

    Position *path;
    gpuErrorCheck(cudaMallocManaged(&path, grid_size * grid_size * sizeof(Position)));

    Position start = {0, 0};
    Position end = {grid_size - 1, grid_size - 1};

    int block_dim = 32;  // Define an appropriate block dimension

    dim3 blockSize(block_dim, block_dim);
    dim3 gridSize((grid_size + block_dim - 1) / block_dim, (grid_size + block_dim - 1) / block_dim);

    clock_t start_time = clock();
    findPath<<<gridSize, blockSize>>>(grid, grid_size, start, end, distances, previous, visited, path);
    clock_t end_time = clock();
    gpuErrorCheck(cudaDeviceSynchronize());

    if (grid_size < 100000) {
        displayGrid(grid, grid_size);
    }

    printf("Execution time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    gpuErrorCheck(cudaFree(grid));
    gpuErrorCheck(cudaFree(distances));
    gpuErrorCheck(cudaFree(previous));
    gpuErrorCheck(cudaFree(visited));
    gpuErrorCheck(cudaFree(path));

    return 0;
}
