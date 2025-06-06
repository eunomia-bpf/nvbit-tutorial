# mem_printf2

This example shows how a tool can send custom text from GPU code back to the host using NVBit channels. It is essentially a simplified printf implementation that you can adapt for debugging or tracing.

## Source Files
- `mem_trace.cu` – host side logic which sets up a channel, injects calls and receives messages.
- `inject_funcs.cu` – device function `instrument_mem` that formats a string and pushes it on the channel.

## Instrumentation Flow
1. On initialization a channel is allocated (`ChannelDev`/`ChannelHost`). A host thread waits for messages.
2. Every memory reference instruction is instrumented to call `instrument_mem` before execution. This device function formats a short message (currently a placeholder) and sends it through the channel.
3. The host thread prints each received string so you can see the accesses as they occur.

## Building
Run `make` in this directory to produce `mem_printf2.so`.

## Running
Preload the library with your CUDA program:

```bash
LD_PRELOAD=./tools/mem_printf2/mem_printf2.so ./app
```

## Interpreting Results
Each printed line originates from GPU code. You can modify `instrument_mem` to include any information you need such as opcode and address. This tool is intended as a starting point for custom logging.
