# NVBit Tutorial: MOV Instruction Replacement

> Github repo: <https://github.com/eunomia-bpf/nvbit-tutorial>

**⚠️ WARNING: Research/Educational Tool Only**
This tool has 100-1000x overhead. Use only for:
- Learning instruction replacement concepts
- Research (fault injection, custom operations)
- Small-scale testing

**NOT for production or performance profiling!**

**TL;DR:** Replaces MOV instructions with custom function calls. Demonstrates how to modify GPU code at runtime.

## Overview

The `mov_replace` tool intercepts all MOV (move) instructions in a CUDA kernel and replaces them with calls to a custom device function that performs the equivalent operation. This demonstrates several advanced capabilities of NVBit:

1. Removing original instructions
2. Reading and writing registers directly
3. Handling different operand types (immediate values, registers, uniform registers)
4. Preserving the original semantics of an instruction

While this specific example simply replicates the MOV functionality, the same approach can be extended to:
- Add logging/profiling to specific instructions
- Implement custom versions of operations for debugging
- Simulate architectural features not present in hardware
- Test alternative implementations of operations

## Code Structure

The tool consists of two main components:

- `mov_replace.cu` – Host-side code that:
  - Identifies MOV instructions
  - Analyzes operand types
  - Removes original instructions
  - Inserts calls to the custom replacement function
  
- `inject_funcs.cu` – Device-side code that:
  - Implements the `mov_replace` function
  - Uses register access intrinsics to read/write registers
  - Handles various operand types

## How It Works: Host Side (mov_replace.cu)

Let's examine the key parts of the host-side implementation:

### 1. Instruction Analysis and Replacement

```cpp
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // ... standard NVBit setup ...
    
    /* We iterate on the vector of instruction */
    for (auto instr : instrs) {
        /* Check if in target range */
        if (instr->getIdx() < instr_begin_interval || instr->getIdx() >= instr_end_interval) {
            continue;
        }

        std::string opcode = instr->getOpcode();
        /* match every MOV instruction */
        if (opcode.compare(0, 3, "MOV") == 0) {
            /* assert MOV has really two arguments */
            assert(instr->getNumOperands() == 2);
            const InstrType::operand_t *op0 = instr->getOperand(0);
            assert(op0->type == InstrType::OperandType::REG);
            const InstrType::operand_t *op1 = instr->getOperand(1);

            if(op1->type != InstrType::OperandType::REG &&
               op1->type != InstrType::OperandType::UREG &&
               op1->type != InstrType::OperandType::IMM_UINT64 &&
               op1->type != InstrType::OperandType::CBANK) {
                instr->printDecoded();
                printf("Operand %s not handled\n", InstrType::OperandTypeStr[(int) op1->type]);
                continue;
            }

            if (verbose) {
                instr->printDecoded();
            }

            /* Insert a call to "mov_replace" before the instruction */
            nvbit_insert_call(instr, "mov_replace", IPOINT_BEFORE);

            /* Add predicate as argument to the instrumentation function */
            nvbit_add_call_arg_guard_pred_val(instr);

            /* Add destination register number as argument (first operand
             * must be a register)*/
            nvbit_add_call_arg_const_val32(instr, op0->u.reg.num);

            /* add second operand */
            /* 0: non reg, 1: vector reg, 2: uniform reg */
            int is_op1_reg = 0;
            if (op1->type == InstrType::OperandType::REG) {
                is_op1_reg = 1;
                /* register number as immediate */
                nvbit_add_call_arg_const_val32(instr, op1->u.reg.num);

            } else if (op1->type == InstrType::OperandType::UREG) {
                is_op1_reg = 2;
                /* register number as immediate */
                nvbit_add_call_arg_const_val32(instr, op1->u.reg.num);

            } else if (op1->type == InstrType::OperandType::IMM_UINT64) {
                /* Add immediate value (registers are 32 bits so immediate
                 * is also 32-bit and we can cast to int safely) */
                nvbit_add_call_arg_const_val32(
                    instr, (int)op1->u.imm_uint64.value);

            } else if (op1->type == InstrType::OperandType::CBANK) {
                /* Add value from constant bank (passed as immediate to
                 * the mov_replace function) */
                nvbit_add_call_arg_cbank_val(instr, op1->u.cbank.id,
                                             op1->u.cbank.imm_offset);
            }

            /* Add flag to specify if value or register number */
            nvbit_add_call_arg_const_val32(instr, is_op1_reg);

            /* Remove original instruction */
            nvbit_remove_orig(instr);
        }
    }
}
```

The key steps in this function are:

1. **Identify MOV instructions**: We look for instructions with opcodes starting with "MOV"
2. **Analyze operands**: MOV instructions have two operands:
   - The first (destination) is always a register
   - The second (source) can be a register, uniform register, immediate value, or constant bank
3. **Insert replacement call**: We insert a call to our custom `mov_replace` function
4. **Pass arguments**:
   - The instruction's predicate (to handle conditional execution)
   - The destination register number
   - The source (either a register number or immediate value)
   - A flag indicating the source type
5. **Remove original instruction**: The critical `nvbit_remove_orig(instr)` call removes the original MOV instruction

### 2. CUDA Event Callback

```cpp
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunchKernel || /* other launch events */) {
        /* Get the kernel function */
        CUfunction func = /* extract from params */;

        if (!is_exit) {
            /* Instrument the function before launch */
            instrument_function_if_needed(ctx, func);
            /* Enable instrumented version */
            nvbit_enable_instrumented(ctx, func, true);
        }
    }
}
```

This callback is simpler than in other tools:
1. We intercept kernel launches
2. We instrument the function if needed
3. We enable our instrumented version
4. We don't need to maintain counters or print results

## How It Works: Device Side (inject_funcs.cu)

The replacement device function implements the MOV instruction functionality:

```cpp
extern "C" __device__ __noinline__ void mov_replace(int pred, int reg_dst_num,
                                                   int value_or_reg,
                                                   int is_op1_reg) {
    if (!pred) {
        return;
    }

    if (is_op1_reg) {
        if (is_op1_reg == 1) {
            /* read value of register source */
            int value = nvbit_read_reg(value_or_reg);
            /* write value in register destination */
            nvbit_write_reg(reg_dst_num, value);
        } else if (is_op1_reg == 2) {
            /* read value of uniform register source */
            int value = nvbit_read_ureg(value_or_reg);
            /* write value in register destination */
            nvbit_write_reg(reg_dst_num, value);
        }
    } else {
        /* immediate value, just write it in the register */
        nvbit_write_reg(reg_dst_num, value_or_reg);
    }
}
```

Key aspects of this function:

1. **Predicate handling**: Skip execution if the instruction's predicate is false
2. **Register-to-register move**: If source is a register, read its value with `nvbit_read_reg` and write to destination
3. **Uniform register handling**: If source is a uniform register, use `nvbit_read_ureg` instead
4. **Immediate value handling**: If source is an immediate value, write it directly to the destination register
5. **Register access intrinsics**: The NVBit-provided intrinsics `nvbit_read_reg`, `nvbit_read_ureg`, and `nvbit_write_reg` allow direct manipulation of GPU registers

## Register Access Intrinsics

NVBit provides special intrinsics for direct register access that are essential for instruction replacement:

1. **nvbit_read_reg(reg_num)**: Reads the value of a vector register
2. **nvbit_read_ureg(reg_num)**: Reads the value of a uniform register
3. **nvbit_write_reg(reg_num, value)**: Writes a value to a register

These intrinsics are implemented by the NVBit framework and allow our custom function to behave like a real GPU instruction.

## Building the Tool

The build process follows the same pattern as other NVBit tools:

1. Compile the host code:
   ```
   $(NVCC) -dc -c -std=c++11 $(INCLUDES) ... mov_replace.cu -o mov_replace.o
   ```

2. Compile the device function with special flags:
   ```
   $(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions ... inject_funcs.cu -o inject_funcs.o
   ```

3. Link into a shared library:
   ```
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o mov_replace.so
   ```

## Running the Tool

Launch your CUDA application with the tool preloaded:

```bash
LD_PRELOAD=./tools/mov_replace/mov_replace.so ./your_cuda_application
```

### Environment Variables

The tool supports these environment variables:

- `INSTR_BEGIN`/`INSTR_END`: Instruction range to instrument
- `TOOL_VERBOSE`: Enable verbose output to see which instructions are being replaced

## Verifying the Replacement

Since this tool replaces instructions with functionally equivalent implementations, the application should produce identical results. To verify that the replacement is working:

1. **Run with verbose output**: Set `TOOL_VERBOSE=1` to see which MOV instructions are being replaced
2. **Examine with `nvdisasm`**: Disassemble the original and instrumented versions of the kernel to see the difference
3. **Compare results**: Run the application with and without the tool to ensure output is identical

## Understanding GPU Assembly (SASS)

To work effectively with instruction replacement, it helps to understand NVIDIA's GPU assembly language (SASS). Some common MOV variations include:

- **MOV**: Basic register-to-register move
- **MOV32I**: Move 32-bit immediate to register
- **MOVS**: Move with shift
- **MOVC**: Move with conditional execution

The `mov_replace` tool handles the most common variants, but could be extended to handle more specialized forms.

## Other Replaceable Instructions

The same technique can replace other SASS instructions:

### Easy to Replace (Similar to MOV)
- **ADD/SUB/MUL**: Arithmetic operations
- **AND/OR/XOR**: Bitwise operations
- **SHL/SHR**: Bit shifts
- **SEL**: Conditional select

### Moderate Difficulty
- **FADD/FMUL**: Floating-point math (watch for precision)
- **IMAD**: Integer multiply-add (3 operands)
- **LEA**: Load effective address

### Hard to Replace
- **LDG/STG**: Memory operations (need complex address handling)
- **BRA**: Branches (control flow complications)
- **FFMA**: Fused ops (may need multiple replacements)

### Example: Replacing ADD

```cpp
// In instrument_function_if_needed():
if (opcode.compare(0, 3, "ADD") == 0) {
    // Similar to MOV but with 3 operands
    nvbit_insert_call(instr, "add_replace", IPOINT_BEFORE);
    // Add destination, source1, source2
    nvbit_remove_orig(instr);
}

// In inject_funcs.cu:
__device__ void add_replace(int pred, int dst, int src1, int src2, ...) {
    if (!pred) return;
    int val1 = nvbit_read_reg(src1);
    int val2 = nvbit_read_reg(src2);
    nvbit_write_reg(dst, val1 + val2);
}
```

## Extending the Tool

Practical extensions:

1. **Instruction tracing**: Log when/where instructions execute
2. **Fault injection**: Deliberately corrupt values to test resilience
3. **Value tracking**: Monitor specific register values
4. **Custom operations**: Implement ops not in hardware

## Advanced Applications

Instruction replacement can be used for sophisticated applications:

1. **Fault injection**: Deliberately introduce errors to test resilience
2. **Performance modeling**: Add delays to simulate different hardware
3. **Custom operations**: Implement operations not available in hardware
4. **Security analysis**: Track information flow through registers

## Performance Considerations

Replacing instructions with function calls has significant overhead:

1. **Function call overhead**: Each replaced instruction requires state saving/restoring
2. **Register access cost**: Register reads/writes via intrinsics are slower than native instructions
3. **Execution divergence**: Custom handling of predicates may cause additional warp divergence

This approach is primarily for analysis, debugging, or research rather than production use.

## Understanding NVBit's Capabilities

This tool demonstrates several key NVBit capabilities:

1. **Instruction removal**: `nvbit_remove_orig` allows removing original instructions
2. **Register access**: Special intrinsics enable reading and writing registers
3. **Operand analysis**: NVBit's API provides detailed information about instruction operands
4. **Constant values**: `nvbit_add_call_arg_cbank_val` allows accessing constant memory values

These capabilities make NVBit a powerful platform for GPU code manipulation and analysis.

## Next Steps

After understanding instruction replacement, consider:

1. Using `opcode_hist` to identify other common instructions to replace
2. Combining with `mem_trace` to add custom memory access handling
3. Creating a tool that implements custom versions of compute operations
4. Exploring how instruction replacement affects performance

Instruction replacement provides the deepest level of control over GPU execution in the NVBit framework and opens up possibilities for sophisticated GPU code analysis and transformation.
