# NVBit Tutorial: GPU-to-CPU Printing with mem_printf2

> Github repo: <https://github.com/eunomia-bpf/nvbit-tutorial>

**⚠️ Known Issue:** This tool has device function invocation issues in the current implementation. It's included as an educational example of GPU-to-CPU communication concepts.

**TL;DR:** Demonstrates sending printf-style messages from GPU to CPU. Educational tool showing inter-device communication patterns.

**Status:** Partially functional - channel communication works, but device function calls may fail. See main README FAQ for details.

## Comparison: NVBit Printf vs CUDA Printf

| Feature | NVBit mem_printf2 | CUDA printf |
|---------|-------------------|-------------|
| **Requires source code** | No (binary instrumentation) | Yes |
| **Can add to any binary** | Yes | No |
| **Selective instrumentation** | Yes (by kernel, instruction) | No |
| **Performance overhead** | High (50-200x) | Medium (5-20x) |
| **Setup complexity** | Complex (channel, threads) | Simple (builtin) |
| **Best for** | Learning, binary analysis | Normal development |

**When to use CUDA printf instead:**
```cuda
// If you have source code, just use this:
__global__ void kernel() {
    printf("Thread %d value %f\n", threadIdx.x, value);
}
```

**When NVBit printf is useful:**
- Analyzing third-party binaries without source
- Adding instrumentation to specific instructions
- Learning GPU-to-CPU communication patterns
- Research on binary instrumentation

## Overview

The `mem_printf2` tool provides a simplified printf-like functionality for GPU code. Traditional debugging methods like CPU-side printf or debuggers have limitations when working with GPU code:

1. Standard `printf` doesn't work in GPU code across all CUDA versions
2. CPU debuggers can't easily step through thousands of concurrent GPU threads
3. Adding instrumentation often requires modifying the original source code

This tool addresses these issues by:
1. Intercepting memory operations using NVBit
2. Adding instrumentation to format and send custom messages back to the host
3. Processing these messages on the CPU side without modifying the original application

## Code Structure

The tool consists of two main components:

- `mem_trace.cu` – Host-side code that:
  - Sets up a communication channel between GPU and CPU
  - Instruments memory operations in CUDA kernels
  - Receives and prints messages from GPU code
  
- `inject_funcs.cu` – Device-side code that:
  - Formats a message when a memory instruction executes
  - Sends the message through the channel to the host

## How It Works: Communication Channel

At the core of this tool is the communication channel that allows data to flow from GPU to CPU:

```cpp
/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
ChannelDev* channel_dev;
ChannelHost channel_host;
```

The channel has two components:
- `ChannelDev`: A device-side interface accessible from GPU code
- `ChannelHost`: A host-side interface that receives messages

A dedicated host thread continuously polls this channel for new messages.

## How It Works: Host Side (mem_trace.cu)

Let's examine the key elements of the host-side implementation:

### 1. Context Initialization

```cpp
void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = false;
    
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
3. Start a receiver thread that processes messages

### 2. Instrumentation Logic

```cpp
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // ... standard NVBit setup ...
    
    /* Iterate on instructions */
    for (auto instr : instrs) {
        /* Skip non-memory or constant memory instructions */
        if (instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
            instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
            continue;
        }
        
        /* Map opcode to ID for more compact representation */
        if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
            int opcode_id = opcode_to_id_map.size();
            opcode_to_id_map[instr->getOpcode()] = opcode_id;
            id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
        }
        int opcode_id = opcode_to_id_map[instr->getOpcode()];
        
        /* Find memory operands */
        for (int i = 0; i < instr->getNumOperands(); i++) {
            const InstrType::operand_t* op = instr->getOperand(i);
            
            if (op->type == InstrType::OperandType::MREF) {
                /* Insert call to instrumentation function */
                nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                
                /* Add arguments */
                nvbit_add_call_arg_guard_pred_val(instr);  // predicate
                nvbit_add_call_arg_const_val32(instr, opcode_id);  // opcode ID
                nvbit_add_call_arg_mref_addr64(instr, 0);  // memory address
                nvbit_add_call_arg_launch_val64(instr, 0);  // grid launch ID
                nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);  // channel
            }
        }
    }
}
```

The key aspects of instrumentation:
1. We filter for memory instructions
2. For each memory instruction, we insert a call to `instrument_mem`
3. We pass several arguments:
   - The instruction's predicate
   - The opcode ID
   - The memory address
   - The grid launch ID
   - A pointer to the communication channel

### 3. Receiver Thread

```cpp
void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;
    CTXstate* ctx_state = ctx_state_map[ctx];
    ChannelHost* ch_host = &ctx_state->channel_host;
    
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);
    
    while (!ctx_state->recv_thread_done) {
        /* Receive data from the channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        
        if (num_recv_bytes > 0) {
            /* Simply print the received message */
            printf("PRINTF2: %s\n", recv_buffer);
        }
    }
    
    free(recv_buffer);
    ctx_state->recv_thread_done = false;
    return NULL;
}
```

The receiver thread:
1. Continuously polls the channel for new data
2. When data is received, it simply prints the message
3. Unlike more complex tools, it doesn't need to parse a specific data structure

## How It Works: Device Side (inject_funcs.cu)

The device-side code formats and sends messages:

```cpp
extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                      uint64_t addr,
                                                      uint64_t grid_launch_id,
                                                      uint64_t pchannel_dev) {
    /* If thread is predicated off, return */
    if (!pred) {
        return;
    }
    
    /* Format a simple message */
    char formatted_msg[] = "your formatted message\n";
    
    /* Could be replaced with sprintf-like functionality */
    // formatted_msg = your_sprintf("opcode: %d, addr %x\n", opcode_id, addr);
    
    /* Send the message through the channel */
    ((ChannelDev*)pchannel_dev)->push(formatted_msg, sizeof(formatted_msg));
}
```

Key aspects of the device function:
1. We check if the thread is predicated (should execute)
2. We create a simple message (which could be customized)
3. We push the message to the channel for transmission to the host

## Building the Tool

The build process follows the standard pattern for NVBit tools:

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
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o mem_printf2.so
   ```

## Running the Tool

Launch your CUDA application with the tool preloaded:

```bash
LD_PRELOAD=./tools/mem_printf2/mem_printf2.so ./your_cuda_application
```

## Sample Output

The output is simply the messages pushed from the GPU:

```
PRINTF2: Memory address 0x7f1234567890 accessed by thread (0,0,5)
PRINTF2: Value loaded: 3.14159
PRINTF2: Entering branch at block (2,1,0)
```

## Extending the Tool for Real Debugging

While the example shows a simple fixed message, you can extend it for powerful debugging:

### 1. Custom Message Formatting

Implement a GPU-side formatting function to include:
- Thread and block indices
- Memory addresses and values
- Variable values and state information
- Execution paths and conditional branches

### 2. Selective Instrumentation

Modify the host-side code to only instrument:
- Specific functions or kernels
- Particular regions of interest
- Instructions matching certain patterns

### 3. Conditional Messaging

Add conditions in the device function to only print messages when:
- Specific values are encountered
- Errors or boundary conditions occur
- Execution reaches particular code paths

## Applications

This simple printf-like functionality has numerous applications:

1. **Debugging Complex Kernels**: Print internal state at key points
2. **Algorithm Verification**: Trace execution paths and intermediate results
3. **Performance Analysis**: Track memory access patterns and execution flow
4. **Error Detection**: Identify when and where invalid values appear

## Advantages Over Traditional Methods

Compared to other debugging approaches, this tool offers several advantages:

1. **Non-invasive**: Doesn't require modifying application source code
2. **Selective**: Can be applied to specific instructions
3. **Low overhead**: Minimal impact when not actively printing
4. **Thread-specific**: Can identify which thread generated each message

## Performance Considerations

While useful for debugging, consider these performance implications:

1. **Channel Congestion**: Too many messages can overflow the channel
2. **Execution Overhead**: Each instrumented instruction takes longer
3. **Memory Overhead**: Large messages consume more bandwidth

For optimal use:
- Be selective about which instructions trigger messages
- Keep messages concise
- Consider filtering messages on the device side

## Building Your Own Printf Tool

You can customize this tool for your specific needs:

1. **Custom Data Structures**: Send structured data instead of text
2. **Advanced Filtering**: Only report interesting events
3. **Integration with Analysis**: Process messages for automatic analysis
4. **Pattern Detection**: Look for specific sequences of events

## Next Steps

After mastering basic GPU-to-CPU communication with `mem_printf2`, consider:

1. Implementing a more sophisticated printf with formatting support
2. Creating a conditional debugging system that triggers on specific events
3. Building a hybrid analysis tool that combines printf with other instrumentation
4. Developing a real-time visualization of GPU execution based on the messages

The ability to send messages from GPU to CPU opens up numerous possibilities for understanding and debugging GPU code that would otherwise be difficult to inspect.
