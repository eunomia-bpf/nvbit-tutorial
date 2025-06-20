# NVBit Tutorial: Basic Block Instruction Counting

This tool demonstrates how to count GPU instructions efficiently by instrumenting at the basic block level rather than instrumenting every instruction. It significantly reduces the overhead of dynamic instruction counting while maintaining accuracy.

## Overview

A basic block is a sequence of instructions with a single entry point (the first instruction) and a single exit point (the last instruction). No branches exist in the middle of a basic block, and no branches target the middle of a basic block. This property makes basic blocks ideal for efficient instrumentation.

Instead of instrumenting every instruction (as in the basic `instr_count` tool), `instr_count_bb`:

1. Identifies all basic blocks in a function
2. Instruments only the first instruction of each basic block
3. Passes the basic block size to the instrumentation function
4. Updates the counter based on the entire block size

This approach drastically reduces the number of instrumentation points and function calls during execution while still providing an accurate instruction count.

## Code Structure

The tool consists of:

- `instr_count.cu` – Host code that:
  - Analyzes kernel code to identify basic blocks
  - Instruments the first instruction of each basic block
  - Tracks predicated instructions separately (optional)
  - Reports results after kernel execution
  
- `inject_funcs.cu` – Device code that provides:
  - `count_instrs` function to increment the counter by the basic block size
  - `count_pred_off` function to track predicated-off instructions

## How It Works: Host Side (instr_count.cu)

The host-side implementation differs from `instr_count` in how it analyzes and instruments the code:

### 1. Basic Block Analysis

```cpp
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // ... setup similar to instr_count ...
    
    /* Get the static control flow graph of instruction */
    const CFG_t &cfg = nvbit_get_CFG(ctx, f);
    if (cfg.is_degenerate) {
        printf("Warning: Function %s is degenerated, we can't compute basic blocks statically",
               nvbit_get_func_name(ctx, f));
    }
    
    /* Iterate on basic blocks and inject at the first instruction */
    for (auto &bb : cfg.bbs) {
        Instr *i = bb->instrs[0];
        /* inject device function */
        nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
        /* add size of basic block in number of instruction */
        nvbit_add_call_arg_const_val32(i, bb->instrs.size());
        /* add count warp level option */
        nvbit_add_call_arg_const_val32(i, count_warp_level);
        /* add pointer to counter location */
        nvbit_add_call_arg_const_val64(i, (uint64_t)&counter);
    }
    
    /* Handle predicated instructions if required */
    if (exclude_pred_off) {
        /* iterate on instructions */
        for (auto i : nvbit_get_instrs(ctx, f)) {
            /* inject only if instruction has predicate */
            if (i->hasPred()) {
                /* inject function */
                nvbit_insert_call(i, "count_pred_off", IPOINT_BEFORE);
                /* add guard predicate as argument */
                nvbit_add_call_arg_guard_pred_val(i);
                /* add count warp level option */
                nvbit_add_call_arg_const_val32(i, count_warp_level);
                /* add pointer to counter predicate off location */
                nvbit_add_call_arg_const_val64(i, (uint64_t)&counter_pred_off);
            }
        }
    }
}
```

Key differences from `instr_count`:

1. We get the static Control Flow Graph (CFG) using `nvbit_get_CFG`
2. We iterate over basic blocks (`cfg.bbs`) instead of individual instructions
3. We only instrument the first instruction of each basic block
4. We pass the entire basic block size as an argument
5. If predicate tracking is enabled, we instrument those separately

### 2. Result Reporting

```cpp
/* After kernel completion */
uint64_t kernel_instrs = counter - counter_pred_off;
tot_app_instrs += kernel_instrs;

printf("kernel %d - %s - #thread-blocks %d, kernel instructions %ld, total instructions %ld\n",
       kernel_id++, nvbit_get_func_name(ctx, func, mangled), num_ctas,
       kernel_instrs, tot_app_instrs);
```

When reporting results, we subtract the predicated-off instruction count from the total if that tracking is enabled.

## How It Works: Device Side (inject_funcs.cu)

The device code provides two functions:

### 1. Basic Block Counter

```cpp
extern "C" __device__ __noinline__ void count_instrs(int num_instrs,
                                                    int count_warp_level,
                                                    uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);
    
    /* each thread will get a lane id */
    const int laneid = get_laneid();
    
    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;
    
    /* count all the active thread */
    const int num_threads = __popc(active_mask);
    
    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            atomicAdd((unsigned long long*)pcounter, 1 * num_instrs);
        } else {
            atomicAdd((unsigned long long*)pcounter, num_threads * num_instrs);
        }
    }
}
```

The key difference from `instr_count` is the `num_instrs` parameter, which contains the number of instructions in the basic block. We multiply this by either 1 (warp-level counting) or by the number of active threads (thread-level counting).

### 2. Predicate Tracking

```cpp
extern "C" __device__ __noinline__ void count_pred_off(int predicate,
                                                      int count_warp_level,
                                                      uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);
    
    /* get predicate mask */
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    
    /* get mask of threads that have their predicate off */
    const int mask_off = active_mask ^ predicate_mask;
    
    /* count the number of threads that have their predicate off */
    const int num_threads_off = __popc(mask_off);
    
    /* only the first active thread updates the counter */
    if (get_laneid() == __ffs(active_mask) - 1) {
        if (count_warp_level) {
            if (predicate_mask == 0) {
                atomicAdd((unsigned long long*)pcounter, 1);
            }
        } else {
            atomicAdd((unsigned long long*)pcounter, num_threads_off);
        }
    }
}
```

This function tracks instructions that would be skipped due to predicates being false. It uses a bitwise XOR between the active mask and predicate mask to identify threads where the instruction would be skipped.

## Building the Tool

The build process follows the same pattern as other NVBit tools:

1. Compile the host code:
   ```
   $(NVCC) -dc -c -std=c++11 $(INCLUDES) ... instr_count.cu -o instr_count.o
   ```

2. Compile the device functions with the necessary flags:
   ```
   $(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions ... inject_funcs.cu -o inject_funcs.o
   ```

3. Link into a shared library:
   ```
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o instr_count_bb.so
   ```

## Running the Tool

Launch your CUDA application with the tool preloaded:

```bash
LD_PRELOAD=./tools/instr_count_bb/instr_count_bb.so ./your_cuda_application
```

### Environment Variables

The tool supports the same environment variables as `instr_count`:

- `START_GRID_NUM`/`END_GRID_NUM`: Kernel launch range to instrument
- `COUNT_WARP_LEVEL`: Count at warp or thread level (default: 1)
- `EXCLUDE_PRED_OFF`: Track predicated-off instructions (default: 0)
- `ACTIVE_FROM_START`: Instrument from start or wait for profiler commands (default: 1)
- `MANGLED_NAMES`: Print mangled kernel names (default: 1)
- `TOOL_VERBOSE`: Enable verbose output (default: 0)

## Sample Output

The output format is the same as `instr_count`:

```
------------- NVBit (NVidia Binary Instrumentation Tool) Loaded --------------
[Environment variables and settings shown here]
----------------------------------------------------------------------------------------------------
kernel 0 - vecAdd(double*, double*, double*, int) - #thread-blocks 98, kernel instructions 50077, total instructions 50077
Final sum = 100000.000000; sum/n = 1.000000 (should be ~1)
```

## Performance Benefits

The key advantage of basic block instrumentation is performance. For a kernel with many instructions but few basic blocks, the difference can be substantial:

| Kernel                | Instructions | Basic Blocks | Instrumentation Calls (instr_count) | Instrumentation Calls (instr_count_bb) | Speedup |
|-----------------------|--------------|--------------|-------------------------------------|----------------------------------------|---------|
| Small vector addition | 10,000       | 20           | 10,000                              | 20                                     | ~500x   |
| Complex ML kernel     | 1,000,000    | 500          | 1,000,000                           | 500                                    | ~2000x  |

The basic block approach makes dynamic instrumentation practical for production-sized applications and complex kernels.

## How CFG Analysis Works

NVBit provides a static Control Flow Graph (CFG) analysis through `nvbit_get_CFG()`. This analyzes the instruction sequence and branch targets to partition code into basic blocks:

1. Each function entry point starts a new basic block
2. Each branch instruction ends a basic block
3. Each branch target starts a new basic block
4. Instructions between these boundaries form a linear sequence in a single basic block

The CFG represents control flow as a directed graph where:
- Nodes are basic blocks
- Edges represent possible flow paths

A "degenerate" CFG (indicated by `cfg.is_degenerate`) means that the static analysis couldn't properly identify all basic blocks, usually due to complex control flow patterns or indirect branches.

## Extending the Tool

The basic block approach can be extended for other analyses:

1. **Path profiling**: Count how many times each control flow path is taken
2. **Hot spot identification**: Identify the most frequently executed basic blocks
3. **Dynamic CFG construction**: Build a runtime CFG based on actual execution
4. **Block-level memory tracking**: Analyze memory access patterns per basic block

## When to Use Basic Block Instrumentation

Basic block instrumentation is ideal for:

1. Large, complex kernels where instrumentation overhead matters
2. Production profiling where minimizing slowdown is important
3. Tools that need to collect aggregate statistics rather than per-instruction data

For analyses that need instruction-specific details (like the opcode histogram), you'll still need to instrument individual instructions.

## Limitations

A few limitations to be aware of:

1. Cannot capture instruction-specific behavior within a basic block
2. The static CFG analysis may miss some control flow paths for complex code
3. Some GPU architectures may have special cases that affect basic block identification

## Next Steps

After understanding basic block instrumentation, consider:

1. Combining this approach with `mem_trace` for efficient memory access analysis
2. Using the CFG information to build visualization tools for kernel control flow
3. Implementing more sophisticated analyses like critical path identification

Basic block instrumentation is a powerful technique that balances detail and performance, making it ideal for many CUDA analysis tools.
