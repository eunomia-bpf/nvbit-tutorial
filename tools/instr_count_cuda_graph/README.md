# NVBit Tutorial: Instruction Counting with CUDA Graphs Support

> Github repo: <https://github.com/eunomia-bpf/nvbit-tutorial>

**TL;DR:** Like `instr_count` but works with CUDA Graphs. Use this if your app uses cudaGraphLaunch, stream capture, or manual graph construction.

**Quick Start:**
```bash
LD_PRELOAD=./tools/instr_count_cuda_graph/instr_count_cuda_graph.so ./graph_app
# Works with both traditional and graph-based launches
```

**What are CUDA Graphs?**
CUDA Graphs let you record sequences of kernel launches and replay them efficiently. Think of it like "recording a macro" of GPU operations for faster re-execution.

```cpp
// Traditional (works with regular instr_count)
for (int i = 0; i < 100; i++) {
    kernel<<<blocks, threads>>>();
}

// CUDA Graph (needs instr_count_cuda_graph)
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
kernel<<<blocks, threads, 0, stream>>>();
cudaStreamEndCapture(stream, &graph);
for (int i = 0; i < 100; i++) {
    cudaGraphLaunch(graph, stream);  // Much faster!
}
```

## Overview

CUDA Graphs provide a way to define and optimize sequences of CUDA operations, improving performance by reducing launch overhead and enabling better scheduling. However, they present unique challenges for instrumentation tools:

1. Kernels can be launched directly or captured into a graph for later execution
2. The same graph can be executed multiple times with different inputs
3. Multiple kernels in a graph can execute concurrently

The `instr_count_cuda_graph` tool addresses these challenges by:
1. Supporting both direct kernel launches and graph-based launches
2. Tracking per-kernel instruction counts even for concurrent execution
3. Synchronizing results after graph execution
4. Handling both manual graph construction and stream capture

## Code Structure

The tool consists of two main components:

- `instr_count_cuda_graph.cu` – Host-side code that:
  - Maps CUDA functions to instruction counters
  - Handles different types of kernel launches
  - Processes CUDA Graph API calls
  - Reports per-kernel instruction counts
  
- `inject_funcs.cu` – Device-side code that:
  - Executes on the GPU for each instruction
  - Updates per-kernel counters atomically

## CUDA Graphs Overview

Before diving into the implementation, let's understand the three main ways kernels can be launched with CUDA Graphs:

1. **Standard kernel launch**: The traditional `cudaLaunchKernel` or `cuLaunchKernel` approach
2. **Stream capture**: Capturing a sequence of operations executed on a stream into a graph
3. **Manual graph construction**: Building a graph by explicitly adding kernel nodes

A complete instrumentation tool must handle all three scenarios.

## How It Works: Host Side (instr_count_cuda_graph.cu)

Let's examine the key elements of the host-side implementation:

### 1. Per-Kernel Counter Management

```cpp
/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory */
uint64_t tot_app_instrs = 0;

/* per kernel instruction counter, updated by the GPU */
std::unordered_map<CUfunction, kernel_info> kernel_map;
__managed__ uint64_t kernel_counter[MAX_NUM_KERNEL];
```

Unlike the basic instruction counter which uses a single counter, this tool:
1. Maps each kernel function to a unique ID
2. Allocates a separate counter for each kernel in managed memory
3. Tracks up to `MAX_NUM_KERNEL` different kernels

This approach allows multiple kernels to update their counters concurrently without interference.

### 2. Kernel Launch Instrumentation

```cpp
void try_to_instrument(CUfunction f, CUcontext ctx) {
    /* skip encountered kernels */
    if (kernel_map.find(f) != kernel_map.end()) {
        return;
    }
    
    /* stop instrumenting kernels if we run out of kernel_counters */
    if (kernel_id >= MAX_NUM_KERNEL) {
        /* keep record of total number of launched kernels */
        kernel_id++;
        return;
    }
    
    /* Associate this kernel with a unique counter */
    kernel_map[f].kernel_id = kernel_id++;
    
    /* Instrument the function */
    instrument_function_if_needed(ctx, f);
    
    /* Control whether instrumentation is active */
    if (active_from_start) {
        if (kernel_map[f].kernel_id >= start_grid_num &&
            kernel_map[f].kernel_id < end_grid_num) {
            active_region = true;
        } else {
            active_region = false;
        }
    }

    if (active_region) {
        nvbit_enable_instrumented(ctx, f, true);
    } else {
        nvbit_enable_instrumented(ctx, f, false);
    }
}
```

This function is called for any kernel that's about to be launched or added to a graph:
1. It checks if the kernel has already been seen and skips if it has
2. It assigns a unique ID to the kernel and maps it to a counter
3. It instruments the kernel code to count instructions
4. It enables or disables instrumentation based on configuration

### 3. CUDA Event Handling

```cpp
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    // ... lock mutex to prevent concurrent execution ...
    
    /* Handle different types of kernel launches and graph operations */
    switch (cbid) {
        case API_CUDA_cuLaunchKernel_ptsz:
        case API_CUDA_cuLaunchKernel:
        // ... other direct launch APIs ...
            {
                /* Extract kernel function and stream */
                CUfunction f = extract_function(params);
                CUstream hStream = extract_stream(params);
                
                if (!is_exit) {
                    /* Before launch: Instrument kernel */
                    try_to_instrument(f, ctx);
                } else {
                    /* Check if stream is capturing */
                    cudaStreamCaptureStatus streamStatus;
                    CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
                    
                    if (streamStatus != cudaStreamCaptureStatusActive) {
                        /* Regular launch: Wait for completion and print results */
                        CUDA_SAFECALL(cudaDeviceSynchronize());
                        print_kernel_stats(f, ctx, get_num_ctas(params));
                    } else {
                        /* Stream capture: Don't synchronize */
                        if (verbose >= 1) {
                            printf("kernel %s captured by cuda graph\n", 
                                   nvbit_get_func_name(ctx, f));
                        }
                    }
                }
            }
            break;
            
        case API_CUDA_cuGraphAddKernelNode:
            {
                /* Extract kernel from graph node */
                cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
                CUfunction f = p->nodeParams->func;
                
                if (!is_exit) {
                    /* Instrument kernel that's being added to graph */
                    try_to_instrument(f, ctx);
                }
            }
            break;
            
        case API_CUDA_cuGraphLaunch:
            {
                /* Handle graph launch completion */
                if (is_exit) {
                    cuGraphLaunch_params *p = (cuGraphLaunch_params *)params;
                    
                    /* Wait for all kernels in graph to complete */
                    CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
                    
                    /* Print results for all kernels in the graph */
                    for (const auto& kernel: kernel_map) {
                        print_kernel_stats(kernel.first, ctx, 0);
                    }
                }
            }
            break;
            
        // ... other cases ...
    }
}
```

This function handles all CUDA API calls related to kernel launches and graphs:

1. **Direct kernel launches**: For traditional launches, it instruments the kernel, waits for completion, and prints statistics.
2. **Stream capture**: It detects if a kernel is being captured into a graph and skips synchronization.
3. **Graph node creation**: It instruments kernels as they're added to graphs.
4. **Graph launch**: After a graph executes, it synchronizes and prints statistics for all kernels.

### 4. Result Reporting

```cpp
void print_kernel_stats(CUfunction f, CUcontext ctx, int num_ctas) {
    /* Add this kernel's count to the total */
    tot_app_instrs += kernel_counter[kernel_map[f].kernel_id];
    
    /* Print statistics */
    printf(
        "\nkernel %d - %s - #thread-blocks %d,  kernel "
        "instructions %ld, total instructions %ld\n",
        kernel_map[f].kernel_id, nvbit_get_func_name(ctx, f, mangled), num_ctas,
        kernel_counter[kernel_map[f].kernel_id], tot_app_instrs);
    
    /* Reset counter for this kernel */
    kernel_counter[kernel_map[f].kernel_id] = 0;
}
```

After a kernel or graph completes:
1. We retrieve the instruction count for the specific kernel
2. We add it to the running total
3. We print detailed statistics
4. We reset the kernel's counter for future executions

## How It Works: Device Side (inject_funcs.cu)

The device function is similar to the basic instruction counter but operates on a specified counter:

```cpp
extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                    int count_warp_level,
                                                    uint64_t pcounter) {
    /* Calculate active and predicated threads */
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    
    /* Only the first active thread updates the counter */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            /* Count at warp level if any thread is active */
            if (num_threads > 0) {
                atomicAdd((unsigned long long*)pcounter, 1);
            }
        } else {
            /* Count at thread level */
            atomicAdd((unsigned long long*)pcounter, num_threads);
        }
    }
}
```

Key aspects of this device function:
1. It takes a pointer to a specific kernel's counter
2. It counts instructions either at warp or thread level
3. It uses atomic operations to safely update the counter

## Understanding CUDA Graphs Scenarios

The tool handles three distinct scenarios with CUDA Graphs:

### 1. Traditional Kernel Launch

For kernels launched directly (not through graphs):
- Instrument the kernel before launch
- Synchronize after the kernel completes
- Read and print the instruction count
- Reset the counter

### 2. Stream Capture

When a kernel is launched during stream capture:
- Instrument the kernel but recognize it's not executing immediately
- Skip synchronization and counter reading
- Wait until the captured graph is launched later
- After graph launch, synchronize and read all counters

### 3. Manual Graph Construction

When a kernel is added to a graph manually:
- Instrument the kernel when it's added to the graph
- Don't take any action until the graph is launched
- After graph launch, synchronize and read all counters

## Building the Tool

The build process follows the standard pattern for NVBit tools:

1. Compile the host code:
   ```
   $(NVCC) -dc -c -std=c++11 $(INCLUDES) ... instr_count_cuda_graph.cu -o instr_count_cuda_graph.o
   ```

2. Compile the device function:
   ```
   $(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions ... inject_funcs.cu -o inject_funcs.o
   ```

3. Link into a shared library:
   ```
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o instr_count_cuda_graph.so
   ```

## Running the Tool

Launch your CUDA application with the tool preloaded:

```bash
LD_PRELOAD=./tools/instr_count_cuda_graph/instr_count_cuda_graph.so ./your_cuda_application
```

### Environment Variables

The tool supports the same environment variables as the basic instruction counter:

- `INSTR_BEGIN`/`INSTR_END`: Instruction range to instrument
- `START_GRID_NUM`/`END_GRID_NUM`: Kernel launch range to instrument
- `COUNT_WARP_LEVEL`: Count at warp or thread level (default: 1)
- `EXCLUDE_PRED_OFF`: Skip predicated-off instructions (default: 0)
- `ACTIVE_FROM_START`: Instrument from start or wait for profiler commands (default: 1)
- `MANGLED_NAMES`: Print mangled kernel names (default: 1)
- `TOOL_VERBOSE`: Enable verbose output (default: 0)

## Sample Output

Here's an example of what the output might look like for a CUDA Graphs application:

```
------------- NVBit (NVidia Binary Instrumentation Tool) Loaded --------------
[Environment variables and settings shown here]
----------------------------------------------------------------------------------------------------
kernel 0 - matrixMultiply - #thread-blocks 128, kernel instructions 35840, total instructions 35840
kernel 1 - vectorAdd - #thread-blocks 256, kernel instructions 65536, total instructions 101376
kernel 2 - reduceSum - #thread-blocks 32, kernel instructions 15360, total instructions 116736
```

## Benefits for CUDA Graphs Applications

This tool provides several advantages for applications using CUDA Graphs:

1. **Accurate profiling**: Get precise instruction counts even for graph-launched kernels
2. **Per-kernel breakdown**: See which kernels in a graph consume the most instructions
3. **Repeated execution analysis**: Measure instructions across multiple graph executions
4. **Optimization guidance**: Identify which kernels to focus on for performance improvement

## Common CUDA Graphs Patterns

The tool helps you analyze several common CUDA Graphs usage patterns:

### 1. Repetitive Execution

```cpp
// Create and capture a graph
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel1<<<blocks, threads, 0, stream>>>(data);
kernel2<<<blocks, threads, 0, stream>>>(data);
cudaStreamEndCapture(stream, &graph);

// Create executable graph
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Execute multiple times
for (int i = 0; i < 100; i++) {
    cudaGraphLaunch(graphExec, stream);
}
```

The tool will report the instruction count after each graph execution.

### 2. Complex Dependencies

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add nodes manually
cudaGraphNode_t node1, node2, node3;
cudaGraphAddKernelNode(&node1, graph, NULL, 0, &nodeParams1);
cudaGraphAddKernelNode(&node2, graph, &node1, 1, &nodeParams2);
cudaGraphAddKernelNode(&node3, graph, &node2, 1, &nodeParams3);

// Launch graph
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```

The tool instruments each kernel as it's added to the graph.

## Extension Ideas

This tool can be extended in several ways:

1. **Graph structure analysis**: Identify relationships between nodes in a graph
2. **Memory profiling**: Track memory accesses for graph-launched kernels
3. **Optimization suggestions**: Recommend how to restructure graphs for better performance
4. **Comparative analysis**: Compare instruction counts between graph and non-graph executions

## CUDA Graphs Best Practices

Based on the insights from this tool, here are some best practices for CUDA Graphs:

1. **Balance graph size**: Very large graphs may cause instrumentation overhead
2. **Counter limits**: Be aware of the `MAX_NUM_KERNEL` limit for unique kernels
3. **Synchronization**: The tool needs to synchronize after graph execution, which may affect timing measurements
4. **Stream awareness**: Different streams may have different capturing states

## Next Steps

After mastering instruction counting with CUDA Graphs, consider:

1. Building more sophisticated analysis tools for graph-based applications
2. Implementing graph structure visualization
3. Creating a tool that compares performance between traditional launches and graph launches
4. Exploring how different graph structures affect instruction counts

CUDA Graphs support represents an important evolution in NVBit tools, enabling them to work with modern, high-performance CUDA programming patterns.
