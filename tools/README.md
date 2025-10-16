# NVBit Tutorial Tools

This directory contains a collection of example instrumentation tools built with NVBit (NVIDIA Binary Instrumentation Tool). Each tool demonstrates different capabilities of NVBit and serves as a starting point for developing your own instrumentation tools.

## Overview of Tools

1. **instr_count** - Basic instruction counting tool
2. **opcode_hist** - Histogram of executed instructions by opcode
3. **record_reg_vals** - Records register values during execution
4. **instr_count_bb** - Instruction counting at basic block level
5. **instr_count_cuda_graph** - Instruction counting for CUDA graph execution
6. **mem_printf2** - Memory access printing tool
7. **mem_trace** - Memory trace collection tool
8. **mov_replace** - MOV instruction replacement example

## Common Structure

Each tool typically consists of:
- **Host-side code** (.cu file with the same name as the tool) - Sets up instrumentation, manages callbacks, and processes results
- **Device-side code** (usually inject_funcs.cu) - Contains functions injected into the GPU code
- **Makefile** - Compiles the tool into a shared library (.so)
- **README.md** - Documentation for the specific tool

## Running the Tools

Or alternatively:

```bash
CUDA_INJECTION64_PATH=/path/to/tool.so ./your_cuda_application
```

Most tools support configuration through environment variables. Check each tool's README for specific options.

## Test Application

The repository includes a simple vector addition test application in `/test-apps/vectoradd/` that can be used to test these tools:

```bash
# Compile the test application
cd ../test-apps/vectoradd
make

# Run with a tool
LD_PRELOAD=../../tools/instr_count/instr_count.so ./vectoradd
```

## Tool-Specific Information

### instr_count

Counts dynamic instructions executed by each kernel. It instruments every instruction and increments a counter for each execution.

### opcode_hist

Builds a histogram of instruction opcodes, showing which types of instructions are most common in your application.

### record_reg_vals

Records register values during kernel execution, allowing analysis of data flow.

### instr_count_bb

Similar to instr_count but instruments at basic block level for better performance.

### instr_count_cuda_graph

Extends instruction counting to handle CUDA graphs (a more advanced CUDA feature).

### mem_printf2

Prints memory access information during execution.

### mem_trace

Captures detailed memory access traces for analysis, using a channel-based approach to send data from GPU to CPU.

### mov_replace

Demonstrates how to replace specific instructions (MOV in this case) with custom implementations.

## Developing Your Own Tools

To create a new tool:
1. Copy an existing tool directory that's closest to your needs
2. Modify the host and device code
3. Update the Makefile if needed
4. Run `make` to build your tool

The key NVBit APIs are documented in `../core/nvbit.h`.

For more detailed information on each tool, refer to their individual README files and source code. 