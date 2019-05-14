#include <math_constants.h>

#include "BC.h"
constexpr const int CUDA_BLOCK_SIZE = 256;

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencil(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    // TODO
    uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint row  = t_id/nx;
    uint col  = t_id%nx;
    uint id   = (1+gx)*order/2 + row*gx + col;

    if (t_id < nx*ny){
      switch(order){
        case 2:
          next[id] = Stencil<2>(curr+id, gx, xcfl, ycfl);
          break;
        case 4:
          next[id] = Stencil<4>(curr+id, gx, xcfl, ycfl);
          break;
        case 8:
          next[id] = Stencil<8>(curr+id, gx, xcfl, ycfl);
          break;
        default:
          printf("gpu order error");
      }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencil kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputation(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    uint total_threads = params.nx()*params.ny();
    uint num_blocks = total_threads/CUDA_BLOCK_SIZE+1;
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch(params.order()){
          case 2:
              gpuStencil<2><<<num_blocks, CUDA_BLOCK_SIZE>>>(next_grid.dGrid_,
                curr_grid.dGrid_, params.gx(), params.nx(), params.ny(),
                (float)params.xcfl(), (float)params.ycfl());
                break;
          case 4:
            gpuStencil<4><<<num_blocks, CUDA_BLOCK_SIZE>>>(next_grid.dGrid_,
              curr_grid.dGrid_, params.gx(), params.nx(), params.ny(),
              (float)params.xcfl(), (float)params.ycfl());
              break;
          case 8:
            gpuStencil<8><<<num_blocks, CUDA_BLOCK_SIZE>>>(next_grid.dGrid_,
              curr_grid.dGrid_, params.gx(), params.nx(), params.ny(),
              (float)params.xcfl(), (float)params.ycfl());
              break;
          default:
            printf("order error");
        }


        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencil");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilLoop(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    // TODO
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_id = (blockIdx.y * blockDim.y + threadIdx.y)*numYPerStep;

    if (x_id < nx){
      for (uint y = 0; y < numYPerStep; y++){
        if(y_id+y >= ny){break;}
        uint id = (y_id+y+order/2)*gx + order/2 + x_id;

        //printf("x %u, y %u, id %u\n", x_id, y_id, id);
        switch(order){
          case 2:
            next[id] = Stencil<2>(curr+id, gx, xcfl, ycfl);
            break;
          case 4:
            next[id] = Stencil<4>(curr+id, gx, xcfl, ycfl);
            break;
          case 8:
            next[id] = Stencil<8>(curr+id, gx, xcfl, ycfl);
            break;
          default:
            printf("gpu order error");
        }
      }
    }

}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilLoop kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationLoop(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    const uint steps_y = 8;
    uint t_per_y = 8;
    uint t_per_x = CUDA_BLOCK_SIZE/t_per_y;

    uint num_blocks_x = params.nx()/t_per_x+1;
    uint num_blocks_y = params.ny()/(steps_y*t_per_y)+1;

    dim3 threads(t_per_x, t_per_y);
    dim3 blocks(num_blocks_x, num_blocks_y);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch(params.order()){
          case 2:
              gpuStencilLoop<2, steps_y><<<blocks, threads>>>(next_grid.dGrid_,
                curr_grid.dGrid_, params.gx(), params.nx(), params.ny(),
                (float)params.xcfl(), (float)params.ycfl());
                break;
          case 4:
            gpuStencilLoop<4, steps_y><<<blocks, threads>>>(next_grid.dGrid_,
              curr_grid.dGrid_, params.gx(), params.nx(), params.ny(),
              (float)params.xcfl(), (float)params.ycfl());
              break;
          case 8:
            gpuStencilLoop<8, steps_y><<<blocks, threads>>>(next_grid.dGrid_,
              curr_grid.dGrid_, params.gx(), params.nx(), params.ny(),
              (float)params.xcfl(), (float)params.ycfl());
              break;
          default:
            printf("order error");
        }

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilLoop");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    dim3 threads(0, 0);
    dim3 blocks(0, 0);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuShared");
    return stop_timer(&timer);
}
