# NVBit Tutorial: Comprehensive Guide to GPU Binary Instrumentation

This repository provides a comprehensive, blog-style tutorial for learning NVIDIA Binary Instrumentation Tool (NVBit). It offers detailed, step-by-step guidance with in-depth explanations of code to help you understand GPU binary instrumentation concepts and techniques.

NVBit is covered by the same End User License Agreement as that of the
NVIDIA CUDA Toolkit. By using NVBit you agree to End User License Agreement
described in the EULA.txt file.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

## About This Tutorial Repository

This tutorial repository goes beyond basic examples to provide:

- **Detailed Blog-Style Documentation** for each tool with comprehensive code explanations
- **Step-by-Step Implementation Guides** showing how each tool is built
- **Visual Diagrams and Examples** to illustrate key concepts
- **Best Practices and Performance Considerations**
- **Extension Ideas** for developing your own custom tools

The repository contains:

- **Core NVBit Library** (`core/`) - The main NVBit library and header files
- **Example Tools** (`tools/`) - A collection of practical instrumentation tools with detailed explanations
- **Test Applications** (`test-apps/`) - Simple CUDA applications to demonstrate the tools

Each tool in the `tools/` directory includes a comprehensive tutorial README that walks through the code line-by-line, explains the build process, and describes how to interpret the output.

## Learning Path

The tools are organized in order of increasing complexity to provide a structured learning experience:

1. **instr_count**: Learn the basics of CUDA instrumentation by counting instructions
2. **opcode_hist**: Understand instruction mix analysis with opcode histograms
3. **instr_count_bb**: Improve performance with basic block instrumentation
4. **mov_replace**: Modify kernel behavior by replacing instructions
5. **mem_trace**: Track memory access patterns with sophisticated data collection
6. **mem_printf2**: Implement GPU-to-CPU communication for debugging
7. **record_reg_vals**: Analyze register values during execution
8. **instr_count_cuda_graph**: Handle modern CUDA features like CUDA graphs

## Introduction to NVBit

NVBit (NVIDIA Binary Instrumentation Tool) is a research prototype of a dynamic
binary instrumentation library for NVIDIA GPUs.

NVBit provides a set of simple APIs that enable writing a variety of
instrumentation tools. Example of instrumentation tools are: dynamic
instruction counters, instruction tracers, memory reference tracers,
profiling tools, etc.

NVBit allows writing instrumentation tools (which we call **NVBit tools**)
that can inspect and modify the assembly code (SASS) of a GPU application
without requiring recompilation, thus dynamic. NVBit allows instrumentation
tools to inspect the SASS instructions of each function (\_\_global\_\_ or
\_\_device\_\_) as it is loaded for the first time in the GPU. During this
phase is possible to inject one or more instrumentation calls to arbitrary
device functions before (or after) a SASS instruction. It is also possible to
remove SASS instructions, although in this case NVBit does not guarantee that
the application will continue to work correctly.

NVBit tries to be as low overhead as possible, although any injection of
instrumentation function has an associated cost due to saving and restoring
application state before and after jumping to/from the instrumentation
function.

Because NVBit does not require application source code, any pre-compiled GPU
application should work regardless of which compiler (or version) has been
used (i.e. nvcc, pgicc, etc).

## Requirements

* SM compute capability:              >= 3.5 && <= 12.1
* Host CPU:                           x86\_64, aarch64
* OS:                                 Linux
* GCC version :                       >= 8.5.0 for x86\_64; >= 8.5.0 for aarch64
* CUDA version:                       >= 12.0
* CUDA driver version:                <= 575.xx

## Getting Started with NVBit

This repository uses **NVBit v1.7.6** which includes support for newer CUDA versions and SM architectures up to SM_120.

NVBit is provided in this repository with the following structure:
1. A ```core``` folder, which contains the main static library
```libnvbit.a``` and various headers files (among which the ```nvbit.h```
file which contains all the main NVBit APIs declarations).
2. A ```tools``` folder, which contains various source code examples of NVBit
tools with detailed tutorial documentation. After learning from these examples, you can make a copy of one and modify it to create your own tool.
3. A ```test-apps``` folder, which contains a simple application that can be
used to test NVBit tools. It features a simple vector addition program.

## Building the Tools and Test Applications

To compile the NVBit tools:
```bash
cd tools
make
```

To compile the test application:
```bash
cd test-apps
make
```

**Important Notes for CUDA 12.x+**:
- This repository has been updated to use **g++** for final linking instead of nvcc to avoid device linking issues with CUDA 12.8+
- Test applications should use `-cudart shared` flag when compiling with nvcc
- The Makefiles have been configured to handle these requirements automatically

**Note**: When creating your own tool, make sure you link it with g++ and include the necessary CUDA libraries (`-lcuda -lcudart_static -lpthread -ldl`). The provided Makefiles show the correct pattern.

## Using an NVBit Tool

Before running an NVBit tool, make sure ```nvdisasm``` is in your PATH. In
Ubuntu distributions, this is typically done by adding `/usr/local/cuda/bin` or
`/usr/local/cuda-"version"/bin` to the PATH environment variable.

To use an NVBit tool, you can either:

1. LD_PRELOAD the tool before the application command:
```bash
LD_PRELOAD=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
```

2. Or use CUDA_INJECTION64_PATH:
```bash
CUDA_INJECTION64_PATH=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
```

**NOTE**: NVBit uses the same mechanism as nvprof, nsight system, and nsight compute,
thus they cannot be used together.

## Key Concepts Covered in This Tutorial

Throughout the tutorial, you'll learn important concepts in GPU binary instrumentation:

1. **SASS Instruction Analysis** - Understanding GPU assembly instructions
2. **Function Instrumentation** - Adding code to existing GPU functions
3. **Basic Block Analysis** - Working with control flow graphs
4. **Memory Access Tracking** - Capturing and analyzing memory patterns
5. **Efficient Communication** - Moving data between GPU and CPU
6. **Register Manipulation** - Reading and writing GPU registers directly
7. **Instruction Replacement** - Modifying the behavior of GPU code
8. **Performance Optimization** - Minimizing instrumentation overhead

## Creating Your Own Tools

After working through the examples, you'll be ready to create your own custom instrumentation tools. The repository includes templates and guidance for:

1. **Tool Structure** - Understanding the host/device code organization
2. **Build Systems** - Setting up Makefiles for your tools
3. **Common Patterns** - Reusing code for frequently needed functionality
4. **Debugging Techniques** - Troubleshooting instrumentation issues

## Contributing

We welcome contributions to improve the tutorial! If you find issues or have suggestions:

1. Open an issue describing the problem or enhancement
2. Submit a pull request with your proposed changes
3. Follow the coding style of the existing examples

## Further Resources

For more details on the NVBit APIs, see the comments in `core/nvbit.h`.

You may also find these resources helpful:
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA GPU Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

Happy learning!
