# record_reg_vals

Records register values for selected instructions.  The values of each
register operand are collected for every thread in a warp and printed on
the host.

## How it works

`record_reg_vals.cu` instruments instructions within the chosen range and
uses variadic arguments to pass register contents to
`record_reg_val` in `inject_funcs.cu`.  The device function gathers the
values from all 32 lanes and writes them to a channel.  A host thread
receives `reg_info_t` structures and prints the values per warp.

## Building

Run `make` here or from the top `tools` directory.

## Running

```bash
LD_PRELOAD=./tools/record_reg_vals/record_reg_vals.so ./app
```

The output shows the register values for each instrumented instruction
and warp.
