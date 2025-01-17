/*
 * 2D Heat Diffusion
 *
 * In this homework you will be implementing a finite difference 2D-Heat Diffusion Solver
 * in three different ways, in particular with and without using shared memory.
 * You will implement stencils of orders 2, 4 and 8.  A reference CPU implementation
 * has been provided.  You should keep all existing classes, method names, function names,
 * and variables as is.
 *
 * The simParams and Grid classes are provided for convenience. The simParams class will
 * load a file containing all the information needed for the simulation and calculate the
 * maximum stable CFL number.  The Grid will set up a grid with the appropriate boundary and
 * initial conditions.
 *
 * Some general notes about declaring N-dimensional arrays.
 * You may have seen / been taught to do this in the past:
 * int **A = (int **)malloc(numRows * sizeof(int *));
 * for (int r = 0; r < numRows; ++r)
 *     A[r] = (int *)malloc(numCols * sizeof(int));
 *
 * so that you can then access elements of A with the notation A[row][col], which involves dereferencing
 * two pointers.  This is a *really bad* way to represent 2D arrays for a couple of reasons.
 *
 * 1) For a NxN array, it does N+1 mallocs which is slow.  And on the gpu setting up this data
 *    structure is inconvenient.  But you should know how to do it.
 * 2) There is absolutely no guarantee that different rows are even remotely close in memory;
 *    subsequent rows could allocated on complete opposite sides of the address space
 *    which leads to terrible cache behavior.
 * 3) The double indirection leads to really high memory latency.  To access location A[i][j],
 *    first we have to make a trip to memory to fetch A[i], and once we get that pointer, we have to make another
 *    trip to memory to fetch (A[i])[j].  It would be far better if we only had to make one trip to
 *    memory.  This is especially important on the gpu.
 *
 * The *better way* - just allocate one 1-D array of size N*N.  Then just calculate the correct offset -
 * A[i][j] = *(A + i * numCols + j).  There is only one allocation, adjacent rows are as close as they can be
 * and we only make one trip to memory to fetch a value.  The grid implements this storage scheme
 * "under the hood" and overloads the () operator to allow the more familiar (x, y) notation.
 *
 * For the GPU code in this exercise you don't need to worry about trying to be fancy and overload an operator
 * or use some #define macro magic to mimic the same behavior - you can just do the raw addressing calculations.
 *
 * For the first part of the homework where you will implement the kernels without using shared memory
 * each thread should compute exactly one output.
 *
 * For the second part with shared memory - it is recommended that you use 1D blocks since the ideal
 * implementation will have each thread outputting more than 1 value and the addressing arithmetic
 * is actually easier with 1D blocks.
 */


#include <ostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <unistd.h>

#include "mp1-util.h"
#include "simParams.h"
#include "Grid.h"

#include "gpuStencil.cu"

#define PRINT_ERR

using std::setw;
using std::setprecision;
using std::cout;
using std::endl;

void updateBCsOnly(Grid& grid, Grid& prev, const simParams& params) {
    const int borderSize = params.order() / 2;

    const int gx = params.gx();
    const int gy = params.gy();

    const float dt = params.dt();
    const double dx = params.dx();
    const double dy = params.dy();
    const double a = 0.06/sqrt(dx*dy);
    const float scaling_factor = exp(-2 * a * a * dt);
    assert(scaling_factor > 0);

    const int upper_border_x = gx - borderSize;
    const int upper_border_y = gy - borderSize;

    for(int i = 0; i < gx; ++i) {
        for(int j = 0; j < borderSize; ++j) {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }

        for(int j = upper_border_y; j < gy; ++j) {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }
    }

    for(int j = borderSize; j < upper_border_y; ++j) {
        for(int i = 0; i < borderSize; ++i) {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }

        for(int i = upper_border_x; i < gx; ++i) {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }
    }

    /*
    // Testing that the boundary conditions were correctly applied
    for (int i = 0; i < gx; ++i)
      for (int j = 0; j < gy; ++j)
        if (i<borderSize || i >= upper_border_x || j<borderSize || j >= upper_border_y)
          assert(grid.hGrid_[i + gx * j] == prev.hGrid_[i + gx * j] * scaling_factor);
    */
}

void initGrid(Grid& grid, const simParams& params) {

    const int gx = params.gx();
    const int gy = params.gy();
    const double dx = params.dx();
    const double dy = params.dy();
    const double a = 0.06/sqrt(dx*dy);
    for(int i = 0; i < gx; ++i) {
        for(int j = 0; j < gy; ++j) {
            grid.hGrid_.at(i + gx * j) = sin(i * a * dx) * sin(j * a * dy);
        }
    }

    grid.toGPU();
}


template<int order>
inline float stencil(float* curr_grid, int gx, int x, int y, float xcfl,
                     float ycfl) {
    if(order == 2) {
        return curr_grid[x + gx * y] +
               xcfl * (curr_grid[x+1 + gx * y] + curr_grid[x-1 + gx * y] -
                       2 * curr_grid[x + gx *  y]) +
               ycfl * (curr_grid[x + gx *(y+1)] + curr_grid[x + gx *(y-1)] -
                       2 * curr_grid[x + gx * y]);
    } else if(order == 4) {
        return curr_grid[x + gx * y] +
               xcfl * (-curr_grid[x+2 + gx * y] + 16 * curr_grid[x+1 + gx * y] -
                       30 * curr_grid[x + gx * y] + 16 * curr_grid[x-1 + gx * y] -
                       curr_grid[x-2 + gx * y]) +
               ycfl * (-curr_grid[x +  gx * (y+2)] + 16 * curr_grid[x + gx * (y+1)] -
                       30 * curr_grid[x + gx * y] + 16 * curr_grid[x + gx * (y-1)] -
                       curr_grid[x + gx * (y-2)]);
    } else if(order == 8) {
        return curr_grid[x + gx * y] +
               xcfl * (-9*curr_grid[x+4 + gx * y] + 128 * curr_grid[x+3 + gx *  y] -
                       1008 * curr_grid[x+2 + gx * y] + 8064 * curr_grid[x+1 + gx *  y] -
                       14350 * curr_grid[x + gx *  y] + 8064 * curr_grid[x-1 + gx * y] -
                       1008 * curr_grid[x-2 + gx * y] + 128 * curr_grid[x-3 + gx *  y] -
                       9 * curr_grid[x-4 + gx * y]) +
               ycfl * (-9*curr_grid[x + gx * (y+4)] + 128 * curr_grid[x + gx * (y+3)] -
                       1008 * curr_grid[x + gx * (y+2)] + 8064 * curr_grid[x + gx * (y+1)] -
                       14350 * curr_grid[x + gx * y] + 8064 * curr_grid[x + gx * (y-1)] -
                       1008 * curr_grid[x + gx * (y-2)] + 128 * curr_grid[x + gx * (y-3)] -
                       9 * curr_grid[x + gx * (y-4)]);
    } else {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

double cpuComputation(Grid& curr_grid, const simParams& params) {
    Grid next_grid(curr_grid);

    event_pair timer;
    start_timer(&timer);

    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int borderSize = params.borderSize();

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        updateBCsOnly(curr_grid, next_grid, params);

        // apply stencil
        if(params.order() == 2) {
            for(int y = borderSize; y < ny + borderSize; ++y) {
                for(int x = borderSize; x < nx + borderSize; ++x) {
                    next_grid.hGrid_[x + gx *  y] = stencil<2>(curr_grid.hGrid_.data(), gx, x, y,
                                                    xcfl, ycfl);
                }
            }
        } else if(params.order() == 4) {
            for(int y = borderSize; y < ny + borderSize; ++y) {
                for(int x = borderSize; x < nx + borderSize; ++x) {
                    next_grid.hGrid_[x + gx *  y] = stencil<4>(curr_grid.hGrid_.data(), gx, x, y,
                                                    xcfl, ycfl);
                }
            }
        } else if(params.order() == 8) {
            for(int y = borderSize; y < ny + borderSize; ++y) {
                for(int x = borderSize; x < nx + borderSize; ++x) {
                    next_grid.hGrid_[x + gx *  y] = stencil<8>(curr_grid.hGrid_.data(), gx, x, y,
                                                    xcfl, ycfl);
                }
            }
        }

        Grid::swap(curr_grid, next_grid);
    }

    return stop_timer(&timer);
}

int checkErrors(const Grid& ref_grid, const Grid& gpu_grid,
                const simParams& params, std::string filename, std::vector<double>& errors) {
    //check that we got the same answer
    std::ofstream ofs(filename.c_str());
    int error = 0;
    double l2ref = 0;
    double linf = 0;
    double l2err = 0;

    for(int x = 0; x < params.gx(); ++x) {
        for(int y = 0; y < params.gy(); ++y) {
            const double ref = ref_grid.hGrid_[x + params.gx() * y];
            const double gpu = gpu_grid.hGrid_[x + params.gx() * y];
            if(!AlmostEqualUlps(ref, gpu, 512)) {
                ofs << "Mismatch at pos (" << x << ", " << y << ") cpu: "
                    << ref << " gpu: " << gpu << endl;
                ++error;
            }

            l2ref += ref * ref;
            l2err += (ref - gpu) * (ref - gpu);

            if (ref != 0) linf = max(abs(ref-gpu),linf);
        }
    }

    l2err = sqrt(l2err / params.gx() / params.gy());
    l2ref = sqrt(l2ref / params.gx() / params.gy());

#ifdef PRINT_ERR
    if(error) std::cerr << "There were " << error
                            << " total locations where there was a difference between the cpu and gpu" <<
                            endl;
#endif

    errors.push_back(l2ref);
    errors.push_back(linf);
    errors.push_back(l2err);

    ofs.close();

    return error;
}

void PrintErrors(const std::vector<double>& errorsg,
                 const std::vector<double>& errorsb, const std::vector<double>& errorss) {
    cout << endl;
    cout << setw(15) << " " << setw(15) << "L2Ref" << setw(15) << "LInf" << setw(
             15) << "L2Err" << endl;

    if(errorsg.size() > 0) {
        cout << setw(15) << "Global" << setw(15) << setprecision(6) << errorsg[0]
             << setw(15) << errorsg[1] << setw(15) << errorsg[2] << endl;
    }

    if(errorsb.size() > 0) {
        cout << setw(15) << "Block" << setw(15) << setprecision(6) << errorsb[0]
             << setw(15) << errorsb[1] << setw(15) << errorsb[2] << endl;
    }

    if(errorss.size() > 0) {
        cout << setw(15) << "Shared" << setw(15) << setprecision(6) << errorss[0]
             << setw(15) << errorss[1] << setw(15) << errorss[2] << endl;
    }

    cout << endl;
}

int main(int argc, char* argv[]) {
    bool doGlobal = false;
    bool doShared = false;
    bool doBlock = false;

    std::string helpString = "Usage:\n./heat [-gsb]"
                             "\n-g\tPerform the calculation using global memory"
                             "\n-s\tPerform the calculation using shared memory"
                             "\n-b\tPerform the calculation using block memory"
                             "\n\nBoth options can be passed\n";

    if(argc == 1) {
        std::cerr << helpString;
        exit(1);
    }

    {
        int opt;

        while((opt = getopt(argc, argv, "gsb")) != -1) {
            switch(opt) {
                case 'g':
                    doGlobal = true;
                    break;

                case 's':
                    doShared = true;
                    break;

                case 'b':
                    doBlock = true;
                    break;

                default:
                    std::cerr << helpString;
                    exit(1);
            };
        }
    }

    //load the parameters, setup the grid with the initial and boundary conditions
    simParams params("params.in");
    Grid grid(params.gx(), params.gy());
    initGrid(grid, params);

    //for debugging, you may want to uncomment this line
    //grid.saveStateToFile("init");
    //save our initial state, useful for making sure we got setup and BCs right

    cout << "Order: " << params.order() << ", "
         << params.nx() << "x" << params.ny() << ", "
         << params.iters() << " iterations" << endl;
    cout << setw(15) << " " << setw(15) << "time (ms)" << setw(
             15) << "GBytes/sec" << endl;

    //compute our reference solution
    double elapsed = cpuComputation(grid, params);

    //for debugging, you may want to uncomment the following line
    //grid.saveStateToFile("final_cpu");

    //Print statistics
    cout << setw(15) << "CPU" << setw(15) << setprecision(6) << elapsed
         << setw(15) << params.calcBytes() / (elapsed / 1E3) / 1E9 << endl;

    std::vector<double> errorsb, errorsg, errorss;

    // Use global memory
    if(doGlobal) {
        Grid gpuGrid(grid); // Set up a grid with same dimension as grid
        initGrid(gpuGrid, params); // Initialize the grid
        elapsed = gpuComputation(gpuGrid, params); // Calculation on the GPU
        cout << setw(15) << "Global" << setw(15) << setprecision(6) << elapsed
             << setw(15) << (params.calcBytes() / (elapsed / 1E3)) / 1E9 << endl;

        // Copy back the solution
        gpuGrid.fromGPU();
        // Check for errors
        checkErrors(grid, gpuGrid, params, "globalErrors.txt", errorsg);
        // for debugging, save data to file
        gpuGrid.saveStateToFile("./results/final_gpu_global.csv");
    }

    // This kernel iterates inside a large sub-domain
    if(doBlock) {
        Grid gpuGrid(grid);
        initGrid(gpuGrid, params);
        elapsed = gpuComputationLoop(gpuGrid, params);
        cout << setw(15) << "Block" << setw(15) << setprecision(6) << elapsed
             << setw(15) << (params.calcBytes() / (elapsed / 1E3)) / 1E9 << endl;
        gpuGrid.fromGPU();
        checkErrors(grid, gpuGrid, params, "globalErrors.txt", errorsb);
        gpuGrid.saveStateToFile("./results/final_gpu_block.csv");
    }

    // This kernel uses shared memory
    if(doShared) {
        Grid gpuGrid(grid);
        initGrid(gpuGrid, params);

        if(params.order() == 2) {
            elapsed = gpuComputationShared<2>(gpuGrid, params);
        } else if(params.order() == 4) {
            elapsed = gpuComputationShared<4>(gpuGrid, params);
        } else if(params.order() == 8) {
            elapsed = gpuComputationShared<8>(gpuGrid, params);
        }

        cout << setw(15) << "Shared" << setw(15) << setprecision(6) << elapsed
             << setw(15) << (params.calcBytes() / (elapsed / 1E3)) / 1E9 << endl;
        gpuGrid.fromGPU();
        checkErrors(grid, gpuGrid, params, "sharedErrors.txt", errorss);
        // gpuGrid.saveStateToFile("final_gpu_shared.csv");
    }

#ifdef PRINT_ERR
    PrintErrors(errorsg, errorsb, errorss);
#endif

    return 0;
}
