# record_reg_vals

This tool captures register values for selected instructions and prints them on the host. It can be used to inspect the behaviour of a kernel without modifying the original code.

## Source Files
- `record_reg_vals.cu` – host side logic that instruments instructions and receives register information.
- `inject_funcs.cu` – device function `record_reg_val` which collects register contents from all lanes and pushes a `reg_info_t` structure.
- `common.h` – defines the structure sent over the channel.

## Instrumentation Flow
1. During instrumentation each chosen instruction is examined. The list of register operands is built and passed as variadic arguments to the injected call.
2. `record_reg_val` uses `__shfl_sync` to gather each lane's register value so the host receives a full warp's worth of data. A flush kernel signals the end of a kernel launch.
3. The host thread prints the values grouped by warp and operand index.

## Building
Run `make` here or use `make -C tools` to build all tools. This produces `record_reg_vals.so`.

## Running
Preload the tool before your application:

```bash
LD_PRELOAD=./tools/record_reg_vals/record_reg_vals.so ./app
```

## Interpreting Results
For every instrumented instruction the host prints a heading indicating the CTA and warp followed by the register values for all 32 threads. Values are shown in the order the operands were specified.
