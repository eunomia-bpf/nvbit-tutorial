# NVBit Tutorial: Memory Tracing

This tutorial explores the `mem_trace` tool, which captures detailed information about memory accesses in CUDA kernels. It demonstrates how to use NVBit to track memory behavior and implement efficient GPU-to-CPU communication channels.

## Overview

Understanding memory access patterns is crucial for optimizing GPU code. The `mem_trace` tool:

1. Instruments all memory instructions in a CUDA kernel
2. Captures the addresses accessed by each warp
3. Collects contextual information (grid, block, warp IDs)
4. Sends this data efficiently from the GPU to the CPU
5. Prints or analyzes the memory trace

This enables developers to identify access patterns, detect coalescing issues, find memory divergence, and optimize data layouts.

## Code Structure

The tool consists of three main components:

- `mem_trace.cu` – Host-side code that:
  - Establishes a communication channel between GPU and CPU
  - Identifies and instruments memory instructions
  - Processes and prints memory trace data
  
- `inject_funcs.cu` – Device-side code that:
  - Captures memory addresses from all threads in a warp
  - Packages the data with execution context
  - Sends the data through the channel to the host
  
- `common.h` – Shared structure definition for:
  - The `mem_access_t` struct used to transfer memory access data

## How It Works: Communication Channel

One of the key challenges in GPU instrumentation is getting data from the GPU back to the CPU efficiently. The `mem_trace` tool uses a custom communication channel implementation:

```cpp
/* Channel used to communicate from GPU to CPU receiving thread */
ChannelDev* channel_dev;
ChannelHost channel_host;
```

- `ChannelDev` is a device-side interface to push data
- `ChannelHost` is a host-side interface to receive data
- A dedicated host thread continuously polls for and processes messages

## How It Works: Host Side (mem_trace.cu)

Let's explore the key elements of the host-side implementation:

### 1. Initialization and Context Setup

```cpp
void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    
    /* Allocate channel device object in managed memory */
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    
    /* Initialize the channel host with a callback function */
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                ctx_state->channel_dev, recv_thread_fun, ctx);
                                
    /* Register the receiver thread with NVBit */
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}
```

When a CUDA context is created, we:
1. Create a device channel in CUDA managed memory
2. Initialize a host channel with a callback function
3. Start a receiver thread that will process messages

### 2. Instrumentation Logic

```cpp
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // ... standard NVBit setup ...
    
    /* Iterate on all the static instructions in the function */
    for (auto instr : instrs) {
        /* Skip instructions that aren't memory operations or are constant memory */
        if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
            instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
            instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
            cnt++;
            continue;
        }
        
        /* Map the opcode to a numeric ID */
        if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
            int opcode_id = opcode_to_id_map.size();
            opcode_to_id_map[instr->getOpcode()] = opcode_id;
            id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
        }
        int opcode_id = opcode_to_id_map[instr->getOpcode()];
        
        /* Find memory reference operands */
        int mref_idx = 0;
        for (int i = 0; i < instr->getNumOperands(); i++) {
            const InstrType::operand_t* op = instr->getOperand(i);
            
            if (op->type == InstrType::OperandType::MREF) {
                /* Insert call to instrumentation function */
                nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                
                /* Add arguments */
                nvbit_add_call_arg_guard_pred_val(instr);  // predicate
                nvbit_add_call_arg_const_val32(instr, opcode_id);  // opcode ID
                nvbit_add_call_arg_mref_addr64(instr, mref_idx);  // memory address
                nvbit_add_call_arg_launch_val64(instr, 0);  // grid launch ID (filled at launch)
                nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);  // channel
                
                mref_idx++;
            }
        }
        cnt++;
    }
}
```

The key aspects of the instrumentation:

1. We filter for memory instructions (skipping non-memory and constant memory ops)
2. We map each opcode to a numeric ID for compact representation
3. For each memory reference operand in an instruction, we:
   - Insert a call to `instrument_mem`
   - Pass the predicate, opcode ID, memory address, grid ID, and channel pointer
4. The memory address is extracted with `nvbit_add_call_arg_mref_addr64`
5. The grid launch ID is filled in at launch time with `nvbit_add_call_arg_launch_val64`

### 3. Receiver Thread

```cpp
void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;
    CTXstate* ctx_state = ctx_state_map[ctx];
    ChannelHost* ch_host = &ctx_state->channel_host;
    
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);
    
    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        /* Receive data from the channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                /* Cast the received data to our structure */
                mem_access_t* ma = (mem_access_t*)&recv_buffer[num_processed_bytes];
                
                /* Format and print the memory access information */
                std::stringstream ss;
                ss << "CTX " << HEX(ctx) << " - grid_launch_id "
                   << ma->grid_launch_id << " - CTA " << ma->cta_id_x << ","
                   << ma->cta_id_y << "," << ma->cta_id_z << " - warp "
                   << ma->warp_id << " - " << id_to_opcode_map[ma->opcode_id]
                   << " - ";
                
                for (int i = 0; i < 32; i++) {
                    ss << HEX(ma->addrs[i]) << " ";
                }
                
                printf("MEMTRACE: %s\n", ss.str().c_str());
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }
    
    free(recv_buffer);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}
```

The receiver thread:
1. Continuously polls the channel for new data
2. Processes received bytes in chunks of `mem_access_t` structures
3. Formats and prints information about each memory access
4. Continues until signaled to stop

## How It Works: Device Side (inject_funcs.cu)

The device-side code captures memory access information:

```cpp
extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                      uint64_t addr,
                                                      uint64_t grid_launch_id,
                                                      uint64_t pchannel_dev) {
    /* If thread is predicated off, return */
    if (!pred) {
        return;
    }
    
    /* Get active threads in the warp */
    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    
    /* Create a memory access record */
    mem_access_t ma;
    
    /* Collect memory addresses from all threads using warp vote */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = __shfl_sync(active_mask, addr, i);
    }
    
    /* Fill in execution context */
    int4 cta = get_ctaid();
    ma.grid_launch_id = grid_launch_id;
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;
    
    /* Only the first active thread pushes data to the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}
```

Key aspects of the device function:

1. We check if the thread is predicated (should execute)
2. We use warp-level functions to:
   - Identify active threads with `__ballot_sync`
   - Collect addresses from all threads with `__shfl_sync`
3. We capture execution context (grid, CTA, warp IDs)
4. Only one thread per warp pushes data to avoid duplicates
5. We use the channel to send the data to the host

## The Memory Access Structure (common.h)

The shared structure definition:

```cpp
typedef struct {
    uint64_t grid_launch_id;
    int cta_id_x;
    int cta_id_y;
    int cta_id_z;
    int warp_id;
    int opcode_id;
    uint64_t addrs[32];
} mem_access_t;
```

This compact structure captures:
1. Which kernel invocation (grid_launch_id)
2. Which thread block (CTA coordinates)
3. Which warp within the block
4. What type of memory instruction (opcode_id)
5. The memory addresses accessed by all 32 threads in the warp

## Building the Tool

The build process follows the same pattern as other NVBit tools:

1. Compile the host code:
   ```
   $(NVCC) -dc -c -std=c++11 $(INCLUDES) ... mem_trace.cu -o mem_trace.o
   ```

2. Compile the device function:
   ```
   $(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions ... inject_funcs.cu -o inject_funcs.o
   ```

3. Link into a shared library:
   ```
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o mem_trace.so
   ```

## Running the Tool

Launch your CUDA application with the tool preloaded:

```bash
LD_PRELOAD=./tools/mem_trace/mem_trace.so ./your_cuda_application
```

### Environment Variables

The tool supports these environment variables:

- `INSTR_BEGIN`/`INSTR_END`: Instruction range to instrument
- `TOOL_VERBOSE`: Enable verbose output

## Sample Output

Here's an example of what the output looks like:

```
MEMTRACE: CTX 0x00007f8a0040a000 - grid_launch_id 0 - CTA 0,0,0 - warp 0 - LDG.E - 0x00007f8a01800000 0x00007f8a01800008 0x00007f8a01800010 0x00007f8a01800018 0x00007f8a01800020 0x00007f8a01800028 0x00007f8a01800030 0x00007f8a01800038 0x00007f8a01800040 0x00007f8a01800048 0x00007f8a01800050 0x00007f8a01800058 0x00007f8a01800060 0x00007f8a01800068 0x00007f8a01800070 0x00007f8a01800078 0x00007f8a01800080 0x00007f8a01800088 0x00007f8a01800090 0x00007f8a01800098 0x00007f8a018000a0 0x00007f8a018000a8 0x00007f8a018000b0 0x00007f8a018000b8 0x00007f8a018000c0 0x00007f8a018000c8 0x00007f8a018000d0 0x00007f8a018000d8 0x00007f8a018000e0 0x00007f8a018000e8 0x00007f8a018000f0 0x00007f8a018000f8
```

Each line represents one memory instruction executed by one warp and contains:
- Context information (CTX, grid_launch_id, CTA coordinates, warp_id)
- The operation type (LDG.E = global memory load)
- 32 memory addresses (one per thread in the warp)

## Analyzing Memory Access Patterns

The memory trace output can help identify several common performance issues:

### 1. Memory Coalescing

In the example above, the addresses form a contiguous sequence with 8-byte strides, indicating good coalescing. Poor coalescing would show scattered addresses that require multiple memory transactions.

### 2. Memory Divergence

Look for patterns where threads in the same warp access very different memory regions. This causes serialized memory accesses, reducing performance.

### 3. Bank Conflicts

For shared memory operations, check if addresses map to the same bank, which would cause serialization.

### 4. Access Patterns

Analyze whether accesses follow sequential, strided, or random patterns, which helps optimize data layouts.

## Performance Considerations

Memory tracing has significant overhead due to:
1. The additional function calls for every memory instruction
2. The collection and communication of addresses
3. The processing and printing of trace data

For large applications, consider:
- Limiting instrumentation to specific kernels or instructions
- Sampling memory accesses instead of capturing all of them
- Post-processing the trace to focus on patterns rather than individual accesses

## Channel Implementation

The `mem_trace` tool uses a custom channel implementation to efficiently transfer data from GPU to CPU:

1. **Producer-Consumer Model**: The GPU threads produce data, and a CPU thread consumes it
2. **Ring Buffer**: The channel uses a circular buffer with atomic operations for synchronization
3. **Batched Communication**: Data is transferred in batches to amortize overhead
4. **Asynchronous Processing**: The receiving thread runs concurrently with kernel execution

This approach is much more efficient than using GPU printf or cudaMemcpy for each memory access.

## CUDA Graphs Support

The tool includes special handling for CUDA graphs, which are a way to record and replay sequences of CUDA operations:

1. **Stream Capture**: Detects if a kernel is being captured rather than executed
2. **Graph Node Tracking**: Instruments kernels added to graphs manually
3. **Graph Launch**: Synchronizes and flushes the channel after graph execution

This ensures the tool works correctly with modern CUDA applications that use the graphs API.

## Extending the Tool

You can extend this tool in several ways:

1. **Custom Analysis**: Modify the receiver to analyze patterns instead of just printing
2. **Selective Instrumentation**: Add filters to focus on specific memory operations
3. **Visualization**: Output the trace in a format suitable for visual analysis tools
4. **Memory Hierarchy Analysis**: Add tracking of cache behavior or memory hierarchy effects

## Next Steps

After capturing memory traces, consider:

1. Using tools like `nvvp` or `nsight-compute` to correlate with hardware performance counters
2. Implementing data layout transformations based on identified access patterns
3. Creating a custom visualization tool to better understand complex access patterns
4. Combining with instruction mix analysis from `opcode_hist` for a complete performance picture

Memory access patterns are often the key to GPU performance optimization, and this tool provides the visibility needed to understand and improve them.
