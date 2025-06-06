# NVBit Tutorial: Opcode Histogram

This tutorial explores the `opcode_hist` tool, which builds a histogram of executed instructions categorized by their opcodes. It's a powerful way to understand the instruction mix in your CUDA kernels and identify optimization opportunities.

## Overview

The opcode histogram tool extends the basic concepts from the instruction counting example but adds the ability to categorize instructions by their opcode (operation code). This provides insight into which types of instructions dominate your kernel's execution, helping you focus optimization efforts.

For example, you might discover:
- A high percentage of memory operations, suggesting memory-bound code
- Many type conversions, indicating potential data type mismatches
- Excessive synchronization instructions that could be optimized

## Code Structure

Like other NVBit tools, `opcode_hist` consists of two main components:

- `opcode_hist.cu` – Host code that:
  - Maps instruction opcodes to unique IDs
  - Inserts instrumentation for each instruction
  - Aggregates and prints the histogram after kernel execution
  
- `inject_funcs.cu` – Device code that:
  - Executes on the GPU for each instruction
  - Updates histogram counters in managed memory

## How It Works: Host Side (opcode_hist.cu)

Let's examine the key elements of the host-side implementation:

### 1. Global Variables

```cpp
/* Histogram array updated by GPU threads */
#define MAX_OPCODES (16 * 1024)
__managed__ uint64_t histogram[MAX_OPCODES];

/* Map to translate opcode strings to numeric IDs */
std::map<std::string, int> instr_opcode_to_num_map;
```

The `histogram` array stores the count for each opcode type. It's declared as `__managed__` so it can be updated directly by GPU code. The map translates between opcode strings (like "MOV", "ADD", etc.) and numeric indices in the histogram.

### 2. Instrumentation Logic

```cpp
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // ... similar to instr_count setup ...
    
    /* Iterate on all instructions */
    for (auto i : instrs) {
        /* Check if in target range */
        if (i->getIdx() < instr_begin_interval || i->getIdx() >= instr_end_interval) {
            continue;
        }
        
        /* Get the opcode string and map it to a numeric ID */
        std::string opcode = i->getOpcode();
        if (instr_opcode_to_num_map.find(opcode) == instr_opcode_to_num_map.end()) {
            size_t size = instr_opcode_to_num_map.size();
            instr_opcode_to_num_map[opcode] = size;
        }
        int instr_type = instr_opcode_to_num_map[opcode];
        
        /* Insert call to counting function */
        nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
        
        /* Add arguments */
        nvbit_add_call_arg_guard_pred_val(i);  // predicate value
        nvbit_add_call_arg_const_val32(i, instr_type);  // opcode ID
        nvbit_add_call_arg_const_val32(i, count_warp_level);  // count mode
        nvbit_add_call_arg_const_val64(i, (uint64_t)histogram);  // histogram array
    }
}
```

The key differences from the basic instruction counter:

1. We extract each instruction's opcode string using `i->getOpcode()`
2. We maintain a map from opcode strings to numeric IDs
3. We pass the opcode ID to the device function, allowing it to update the correct histogram slot

### 3. Result Reporting

```cpp
/* After kernel completion */
uint64_t counter = 0;
for (auto a : instr_opcode_to_num_map) {
    if (histogram[a.second] != 0) {
        counter += histogram[a.second];
    }
}
tot_app_instrs += counter;
printf("kernel %d - %s - #thread-blocks %d, kernel instructions %ld, total instructions %ld\n",
       kernel_id++, nvbit_get_func_name(ctx, func), num_ctas, counter, tot_app_instrs);

/* Print non-zero histogram entries */
for (auto a : instr_opcode_to_num_map) {
    if (histogram[a.second] != 0) {
        printf("  %s = %ld\n", a.first.c_str(), histogram[a.second]);
    }
}
```

After kernel execution, we:
1. Calculate the total instruction count by summing histogram entries
2. Print overall statistics similar to the instruction counter
3. Iterate through the opcode map to print each non-zero histogram entry

## How It Works: Device Side (inject_funcs.cu)

The device function is similar to `instr_count` but updates the histogram instead:

```cpp
extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                    int instr_type,
                                                    int count_warp_level,
                                                    uint64_t p_hist) {
    /* Calculate active threads and predicates */
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    
    /* Only the first active thread updates the histogram */
    if (first_laneid == laneid) {
        uint64_t* hist = (uint64_t*)p_hist;
        if (count_warp_level) {
            /* Count once per warp */
            if (num_threads > 0)
                atomicAdd((unsigned long long*)&hist[instr_type], 1);
        } else {
            /* Count once per thread */
            atomicAdd((unsigned long long*)&hist[instr_type], num_threads);
        }
    }
}
```

Key differences from `instr_count`:
1. We take an `instr_type` parameter that specifies which histogram bucket to update
2. We update `hist[instr_type]` instead of a single counter
3. The same warp/thread-level counting options are supported

## Building the Tool

The build process is identical to the instruction counter tool:

1. Compile the host code:
   ```
   $(NVCC) -dc -c -std=c++11 $(INCLUDES) ... opcode_hist.cu -o opcode_hist.o
   ```

2. Compile the device function:
   ```
   $(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions ... inject_funcs.cu -o inject_funcs.o
   ```

3. Link into a shared library:
   ```
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o opcode_hist.so
   ```

## Running the Tool

Preload the shared library with your CUDA application:

```bash
LD_PRELOAD=./tools/opcode_hist/opcode_hist.so ./your_cuda_application
```

### Environment Variables

The tool supports the same environment variables as `instr_count`:
- `INSTR_BEGIN`/`INSTR_END`: Instruction range to instrument
- `KERNEL_BEGIN`/`KERNEL_END`: Kernel launch range to instrument
- `COUNT_WARP_LEVEL`: Count at warp or thread level
- `EXCLUDE_PRED_OFF`: Skip predicated-off instructions
- `TOOL_VERBOSE`: Enable verbose output

## Sample Output

Here's an example of what the output might look like for a vector addition kernel:

```
------------- NVBit (NVidia Binary Instrumentation Tool) Loaded --------------
[Environment variables and settings shown here]
----------------------------------------------------------------------------------------------------
kernel 0 - vecAdd(double*, double*, double*, int) - #thread-blocks 98, kernel instructions 50077, total instructions 50077
  LDG.E = 19600
  SHL = 4900
  IMAD = 9800
  MOV = 4900
  IADD3 = 2450
  STG.E = 9800
  DMUL = 9800
  S2R = 980
  ISETP.GE.AND = 980
  IMAD.MOV = 980
  EXIT = 98
  BRA = 98
```

This output tells us:
- The kernel executed a total of 50,077 instructions
- The most frequent operations were loads (`LDG.E`), stores (`STG.E`), and double-precision multiply (`DMUL`)
- There were relatively few branch instructions (`BRA`)

## Analyzing the Histogram

The histogram provides valuable insights:

1. **Memory Operations**: High counts for `LDG`, `LDS`, `STG`, etc. indicate memory-bound code
2. **Compute Operations**: Many `FMUL`, `FADD`, `IMAD`, etc. suggest compute-bound sections
3. **Control Flow**: `BRA`, `SYNC`, `BAR` instructions indicate branching and synchronization overhead
4. **Data Movement**: `MOV` operations show register-to-register transfers

Comparing the histogram across different implementations of the same algorithm can highlight efficiency differences.

## Extending the Tool

You can extend this tool in several ways:

1. **Categorize by instruction type**: Group similar operations (all loads, all math, etc.)
2. **Track instruction mix over time**: Record histogram at different points in execution
3. **Focus on hotspots**: Instrument only specific functions or code regions
4. **Export data**: Write the histogram to a file for offline analysis

## Performance Considerations

Like all instruction-level instrumentation, this tool adds overhead. For massive kernels, consider:

1. Instrumenting a subset of instructions using `INSTR_BEGIN`/`INSTR_END`
2. Sampling by only enabling instrumentation periodically
3. Using basic block instrumentation concepts (as in `instr_count_bb`) to reduce the number of instrumentation points

## Next Steps

After understanding your kernel's instruction mix with `opcode_hist`, you might want to:

1. Use `mem_trace` to examine memory access patterns for memory-bound kernels
2. Try `mov_replace` to see how to replace specific instructions
3. Create a custom tool that focuses on the specific operations you want to optimize

The opcode histogram is one of the most useful analysis tools for initial CUDA kernel optimization, helping you focus your efforts where they'll have the most impact.
