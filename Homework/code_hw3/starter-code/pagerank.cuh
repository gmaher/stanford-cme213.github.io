#ifndef _PAGERANK_CUH
#define _PAGERANK_CUH

#include "util.cuh"
#include "stdio.h"
/*
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = (1 / 2) A pi(t) + (1 / (2N))
 *
 * You may assume that num_nodes <= blockDim.x * 65535
 *
 */
__global__ void device_graph_propagate(
    const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes
) {
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_nodes){
      graph_nodes_out[i] = 0.f;
      for (uint j = graph_indices[i]; j < graph_indices[i+1]; j++){
        uint k = graph_edges[j];
        graph_nodes_out[i] += (num_nodes*inv_edges_per_node[k])*graph_nodes_in[k];
      }
      graph_nodes_out[i] += 1;
      graph_nodes_out[i] /= (2*num_nodes);
    }
}

/*
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
    const uint *h_graph_indices,
    const uint *h_graph_edges,
    const float *h_node_values_input,
    float  *h_gpu_node_values_output,
    const float *h_inv_edges_per_node,
    int nr_iterations,
    int num_nodes,
    int avg_edges
) {
    // TODO: allocate GPU memory
    const int num_bytes_alloc_indices  = (num_nodes+1) * sizeof(uint);
    const int num_bytes_alloc_float    = num_nodes * sizeof(float);
    const int num_bytes_alloc_edges    = num_nodes*avg_edges * sizeof(uint);

    float *d_node_values_input   = nullptr;
    float *d_node_values_output  = nullptr;
    uint *d_graph_indices        = nullptr;
    uint *d_graph_edges          = nullptr;
    float *d_inv_edges_per_node  = nullptr;

    cudaMalloc((void **) &d_node_values_input,  num_bytes_alloc_float);
    cudaMalloc((void **) &d_node_values_output, num_bytes_alloc_float);
    cudaMalloc((void **) &d_graph_indices,      num_bytes_alloc_indices);
    cudaMalloc((void **) &d_graph_edges,        num_bytes_alloc_edges);
    cudaMalloc((void **) &d_inv_edges_per_node, num_bytes_alloc_float);

    // TODO: check for allocation failure
    if (!d_node_values_input || !d_node_values_output
          || !d_graph_indices || !d_graph_edges || !d_inv_edges_per_node)
    {
        std::cerr << "Couldn't allocate memory!" << std::endl;
        return 1;
    }

    // TODO: copy data to the GPU
    cudaMemcpy(d_node_values_input, &h_node_values_input[0], num_bytes_alloc_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_indices, &h_graph_indices[0], num_bytes_alloc_indices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, &h_graph_edges[0], num_bytes_alloc_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_edges_per_node, &h_inv_edges_per_node[0], num_bytes_alloc_float, cudaMemcpyHostToDevice);

    event_pair timer;
    start_timer(&timer);

    const int block_size = 192;
    const int num_blocks = num_nodes/block_size+1;

    // TODO: launch your kernels the appropriate number of iterations
    for (int i = 0; i < nr_iterations; i++){
      device_graph_propagate<<<num_blocks, block_size>>>(
        d_graph_indices, d_graph_edges, d_node_values_input,
        d_node_values_output, d_inv_edges_per_node, num_nodes
      );

      device_graph_propagate<<<num_blocks, block_size>>>(
        d_graph_indices, d_graph_edges, d_node_values_output,
        d_node_values_input, d_inv_edges_per_node, num_nodes
      );
    }

    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO: copy final data back to the host for correctness checking
    cudaMemcpy(h_gpu_node_values_output, &d_node_values_input[0], num_bytes_alloc_float, cudaMemcpyDeviceToHost);


    // TODO: free the memory you allocated!
    cudaFree(d_node_values_input);
    cudaFree(d_node_values_output);
    cudaFree(d_graph_indices);
    cudaFree(d_graph_edges);
    cudaFree(d_inv_edges_per_node);

    return gpu_elapsed_time;
}

#endif
