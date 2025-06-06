# mov_replace

Illustrates how to modify kernel instructions.  The tool replaces every
`MOV` instruction with a call to a custom device function that performs
the move using NVBit register read/write intrinsics.

## How it works

During instrumentation `mov_replace.cu` searches for instructions whose
opcode begins with `MOV`.  Each matching instruction is removed and a
call to `mov_replace` (from `inject_funcs.cu`) is injected instead.  The
replacement function reads the original source value and writes it to the
destination register.

## Building

Run `make` in this directory or from the main `tools` folder.

## Running

```bash
LD_PRELOAD=./tools/mov_replace/mov_replace.so ./test-apps/vectoradd/vectoradd
```

The program behaves identically but all MOV instructions are executed via
the injected function.
