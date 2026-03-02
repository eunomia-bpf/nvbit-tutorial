# NVBit 内部机制深度分析

> 本文档通过源码分析、二进制逆向、strace/ltrace/GDB 运行时追踪，全面剖析 NVBit v1.7.6 的内部工作原理。
> 适合希望理解 NVBit 底层实现并进行扩展开发的读者。

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 注入机制：从加载到拦截](#2-注入机制从加载到拦截)
- [3. CUDA API 拦截原理](#3-cuda-api-拦截原理)
- [4. SASS 反汇编流水线](#4-sass-反汇编流水线)
- [5. 二进制打补丁引擎](#5-二进制打补丁引擎)
- [6. 寄存器保存与恢复](#6-寄存器保存与恢复)
- [7. GPU-CPU 通信机制](#7-gpu-cpu-通信机制)
- [8. ELF/Cubin 处理](#8-elfcubin-处理)
- [9. 硬件抽象层 (HAL)](#9-硬件抽象层-hal)
- [10. libnvbit.a 内部结构](#10-libnvbita-内部结构)
- [11. 工具 .so 的二进制结构](#11-工具-so-的二进制结构)
- [12. 完整运行时序](#12-完整运行时序)
- [13. 关键 API 参考](#13-关键-api-参考)
- [14. 扩展开发指南](#14-扩展开发指南)
- [15. strace 实证分析](#15-strace-实证分析)

---

## 1. 整体架构

NVBit 的架构分为四个层次：

```
┌─────────────────────────────────────────────────────┐
│                  用户工具层 (Tool)                     │
│  instr_count.cu / inject_funcs.cu / Makefile         │
│  实现 nvbit_at_init, nvbit_at_cuda_event 等回调       │
├─────────────────────────────────────────────────────┤
│                NVBit 公开 API 层                      │
│  nvbit.h: nvbit_get_instrs, nvbit_insert_call, ...   │
│  nvbit_reg_rw.h: nvbit_read_reg, nvbit_write_reg     │
│  utils/channel.hpp: ChannelDev, ChannelHost           │
├─────────────────────────────────────────────────────┤
│              NVBit 核心引擎 (libnvbit.a)              │
│  Nvbit 单例类 → 回调分发、模块管理、函数追踪           │
│  Function 类 → 反汇编、代码生成、二进制打补丁          │
│  SassInstr 类 → SASS 指令编解码                       │
│  ELF 处理库 → cubin 解析、符号表、重定位               │
├─────────────────────────────────────────────────────┤
│            硬件抽象层 (HAL)                            │
│  gk11x_hal (Kepler) │ gm10x_hal (Maxwell)            │
│  gv10x_hal (Volta)  │ gv11x_hal / tu10x_hal (Turing) │
│  ga10x_hal (Ampere) │ gh10x_hal (Hopper)             │
│  gb10x_hal (Blackwell SM100) │ gb12x_hal (SM120)     │
└─────────────────────────────────────────────────────┘
```

### 1.1 libnvbit.a 内部模块

静态库 `libnvbit.a` 包含 **24 个目标文件**，按功能可分为：

| 模块 | 目标文件 | 代码大小 | 功能 |
|------|---------|---------|------|
| 核心引擎 | `nvbit_imp.o` | 239KB text + 372KB data | NVBit 主实现 + 预编译 GPU 辅助内核 |
| 代码补丁 | `function.o` | 129KB | SASS 代码生成、trampoline、分支修正 |
| 指令处理 | `instr.o` | 55KB | SASS 指令解析 (SassInstr 类) |
| 公开 API | `nvbit.o` | 25KB | C 风格 API 包装 (委托给 Nvbit:: 类方法) |
| HAL 后端 | 9 个 `*_hal.o` | 332KB | 架构特定的 SASS 指令编码器 |
| ELF 处理 | `Elf.o` + `tools_shared_readelf*.o` | 82KB | GPU ELF 格式解析 (32/64位) |
| 数据结构 | `tools_shared_{hashmap,list,...}.o` | 17KB | 哈希表、链表、红黑树、区间映射 |

其中 `nvbit_imp.o` 是最大的单体，372KB 的 data 段包含为所有支持的 SM 架构预编译的 GPU 辅助内核（用于运行时寄存器读写）。

---

## 2. 注入机制：从加载到拦截

### 2.1 两种注入方式

NVBit 工具以共享库 (`.so`) 形式存在，通过以下方式注入到目标应用：

```bash
# 方式一：LD_PRELOAD（利用动态链接器）
LD_PRELOAD=./tools/instr_count/instr_count.so ./app

# 方式二：CUDA_INJECTION64_PATH（CUDA 原生注入机制，推荐）
CUDA_INJECTION64_PATH=./tools/instr_count/instr_count.so ./app
```

**关键区别**：`LD_PRELOAD` 是 Linux 通用机制，在所有库之前加载 `.so`。而 `CUDA_INJECTION64_PATH` 是 NVIDIA CUDA 驱动提供的官方工具注入接口，NVBit 优先使用。strace 可以观察到 CUDA 驱动在初始化时检查 `/dev/shm/cuda_injection_path_shm`。

### 2.2 加载顺序（LD_DEBUG=libs 实证）

通过 `LD_DEBUG=libs` 跟踪，库的加载顺序为：

```
1. ld-linux-x86-64.so.2    (动态链接器)
2. libc.so.6 等系统库
3. libstdc++.so.6           (C++ 运行时)
4. libcuda.so.1             (NVIDIA 驱动, ~88MB)
5. libcudart.so.12          (CUDA 运行时, ~5MB)
6. instr_count.so           (NVBit 工具, ~2.6MB, 最后加载)
```

这个顺序是有意为之的：NVBit 需要 `libcuda.so.1` 先完成加载，才能在其上注册回调。

### 2.3 初始化链

工具 `.so` 的 `.init_array` 段包含 **10 个构造函数**，按顺序执行：

```
.init_array 执行流程:
  1. libcudart_static 初始化
  2. frame_dummy (GCC 帧初始化)
  3. __sti____cudaRegisterAll()     ← 向 CUDA 运行时注册 fatbin（设备代码）
  4. __sti____cudaRegisterAll()     ← 额外的 CUDA 注册
  5. C++ 全局构造函数               ← 工具自身的全局变量
  6-10. NVBit 内部初始化           ← Nvbit 单例构造、HAL 初始化等
```

之后调用链为：
```
Nvbit::Nvbit()                      → 构造单例
  → Nvbit::init()                   → 初始化回调系统
    → toolsElfLibInitialize()       → ELF 处理库初始化
    → init_hal_gk11x/gm10x/...()   → 初始化所有 HAL 后端
    → cuGetExportTable()            → 注册 CUDA Tools 回调
  → nvbit_at_init()                 → 调用工具的初始化回调
```

### 2.4 核心拦截点：cuGetExportTable

**NVBit 并不是通过 LD_PRELOAD 符号拦截来 hook 单个 CUDA 函数**。它使用 NVIDIA 的内部函数表机制 `cuGetExportTable`，这是 NVIDIA 官方提供给性能分析工具的接口（nvprof、Nsight 使用相同机制，因此不能同时运行）。

通过 `nm` 分析确认：

```
U cuGetExportTable              ← 外部依赖 (来自 libcuda.so.1)
T nvbitToolsCallbackFunc        ← NVBit 定义的核心回调分发器
```

`nvbitToolsCallbackFunc` 的签名为：
```cpp
void nvbitToolsCallbackFunc(void*, CUtools_cb_domain_enum, unsigned int, void const*);
```

NVBit 通过 `cuGetExportTable` 获取 CUDA 驱动内部的回调订阅接口，然后将 `nvbitToolsCallbackFunc` 注册为回调处理器。此后，CUDA 驱动在每次 API 调用的入口和出口都会调用此函数。

---

## 3. CUDA API 拦截原理

### 3.1 回调域

NVBit 订阅了 CUDA Tools API 的多个回调域（CUtools_cb_domain_enum）：

| 回调域 | 事件示例 | 说明 |
|--------|---------|------|
| **Context** | CONTEXT_CREATED, CONTEXT_DESTROY_STARTING | GPU 上下文生命周期 |
| **Module** | MODULE_LOADED, MODULE_UNLOAD_STARTING | GPU 模块（cubin）加载/卸载 |
| **Function** | FUNCTION_LOADING, FUNCTION_LOADED, FUNCTION_PATCHED | 函数加载和补丁 |
| **Launch** | LAUNCH, BEFORE_LAUNCH_PUSHED, AFTER_GRID_LAUNCHED | 内核启动 |
| **Graph** | GRAPH_CREATED, GRAPHEXEC_CREATED, GRAPHNODE_CREATED | CUDA Graph 操作 |
| **Library** | LIBRARY_LOADED | 库加载 |
| **CUDA API** | cuLaunchKernel, cuMemAlloc, cuModuleLoad, ... | 所有 CUDA 驱动 API |

### 3.2 回调分发流程

```
应用调用 cudaLaunchKernel()
  → libcudart.so 转换为 cuLaunchKernel()
    → libcuda.so 内部执行
      → 触发 NVBit 注册的回调
        → nvbitToolsCallbackFunc(ptr, domain, cbid, params)
          → Nvbit::callback()          [内部分发]
            → nvbit_at_cuda_event()    [工具自定义回调]
```

### 3.3 工具可拦截的 CUDA API

NVBit 拦截 **数百个** CUDA 驱动 API 函数，包括所有 `_v2`、`_v3`、`_ptsz` 变体。在 `tools_cuda_api_meta.h` 中，每个 API 都有唯一的 enum ID（cbid），工具通过比较 cbid 来过滤感兴趣的事件：

```cpp
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (cbid == API_CUDA_cuLaunchKernel && !is_exit) {
        // 内核启动前：进行 instrumentation
        instrument_function_if_needed(ctx, func);
    }
    if (cbid == API_CUDA_cuLaunchKernel && is_exit) {
        // 内核启动后：收集结果
        cudaDeviceSynchronize();
        printf("instructions: %ld\n", counter);
    }
}
```

参数 `params` 可以转换为 `generated_cuda_meta.h` 中定义的结构体，获取 API 调用的具体参数。

### 3.4 可选回调（弱符号探测）

NVBit 通过 `dlsym` 运行时探测工具是否定义了可选回调：

```
nvbit_at_ctx_init    → 可选，CUDA 上下文创建时调用
nvbit_tool_init      → 可选，首次内核启动前调用（可安全进行 CUDA 内存分配）
nvbit_at_ctx_term    → 可选，上下文销毁时调用
nvbit_at_load_device_var → 可选，设备变量加载时调用
```

`LD_DEBUG` 输出中的 "undefined symbol" 错误实际上不是错误，只是 NVBit 在探测可选符号。

---

## 4. SASS 反汇编流水线

### 4.1 外部工具依赖

NVBit **没有内置完整的 SASS 反汇编器**。它通过 `system()` 调用外部的 `nvdisasm` 和 `cuobjdump`：

```c
// libnvbit.a 中的关键符号
U system    // 用于调用外部命令
```

### 4.2 反汇编流程（strace 实证）

**阶段一：提取工具设备代码**

NVBit 首次加载时，使用 `cuobjdump` 从工具 `.so` 中提取匹配当前 GPU 架构的 cubin：

```bash
# NVBit 内部执行的命令（strace 捕获）
cp ./tools/instr_count/instr_count.so /tmp/nvbit_tool_tmpdir.XmFNnT
cd /tmp/nvbit_tool_tmpdir.XmFNnT
cuobjdump instr_count.so -arch sm_120 -xelf all > /dev/null
# 产出: instr_count.1.sm_120.cubin
```

**阶段二：内核函数反汇编**

当需要分析某个 GPU 内核时（通常在首次启动该内核时触发）：

```bash
# 1. 将内核的原始二进制代码写入临时文件
mkstemp("/tmp/nvbit_code_XXXXXX")    # 如 /tmp/nvbit_code_WnBVBR

# 2. 创建 SASS 输出文件
mkstemp("/tmp/nvbit_sass_XXXXXX")    # 如 /tmp/nvbit_sass_yVdJsU

# 3. 调用 nvdisasm 进行反汇编
nvdisasm -b SM120 /tmp/nvbit_code_WnBVBR > /tmp/nvbit_sass_yVdJsU

# 4. 读取 SASS 输出，解析为 Instr 对象
# 5. 删除临时文件
unlink("/tmp/nvbit_code_WnBVBR")
unlink("/tmp/nvbit_sass_yVdJsU")
```

**阶段三：SASS 解析**

`SassInstr` 类将 nvdisasm 的文本输出解析为结构化的指令对象：

```
输入 (nvdisasm 文本):
  /*0000*/ LDC R1, c[0x0][0x37c] ;
  /*0010*/ S2R R0, SR_TID.X ;
  /*00d0*/ LDG.E.64 R2, desc[UR4][R2.64] ;

输出 (Instr 对象):
  Instr { idx=0,  offset=0x0,   opcode="LDC",      sass="LDC R1, c[0x0][0x37c]" }
  Instr { idx=1,  offset=0x10,  opcode="S2R",       sass="S2R R0, SR_TID.X" }
  Instr { idx=13, offset=0xd0,  opcode="LDG.E.64",  sass="LDG.E.64 R2, desc[UR4][R2.64]" }
```

每条 SM120 指令为 16 字节（0x10 偏移步长）。

### 4.3 TOOL_VERBOSE=1 输出示例

```
inspecting vecAdd(double*, double*, double*, int) - num instrs 32
Instr 0  @ 0x0   (0)   - LDC R1, c[0x0][0x37c] ;
Instr 1  @ 0x10  (16)  - S2R R0, SR_TID.X ;
Instr 2  @ 0x20  (32)  - S2UR UR4, SR_CTAID.X ;
...
Instr 13 @ 0xd0  (208) - LDG.E.64 R2, desc[UR4][R2.64] ;
Instr 14 @ 0xe0  (224) - LDG.E.64 R4, desc[UR4][R4.64] ;
Instr 18 @ 0x120 (288) - STG.E.64 desc[UR4][R8.64], R6 ;
Instr 19 @ 0x130 (304) - EXIT ;
Instr 20 @ 0x140 (320) - BRA 0x140;    // 无限循环（不可达代码）
Instr 21-31: NOP 填充至 512 字节对齐
```

---

## 5. 二进制打补丁引擎

这是 NVBit 最核心的部分：在 SASS 二进制层面修改 GPU 代码。

### 5.1 补丁流水线

```
Nvbit::module_loaded()                    ← 模块加载，解析 ELF
  → Nvbit::func_loading()                 ← 函数加载
  → Nvbit::disassemble_func()             ← 调用 nvdisasm 反汇编
  → Nvbit::get_instrs()                   ← 返回指令列表给工具
  → [工具调用 nvbit_insert_call()]         ← 记录注入点（此时不修改代码）
  → [工具调用 nvbit_add_call_arg_*()]      ← 记录参数信息
  → nvbit_enable_instrumented()            ← 触发实际代码生成
    → Function::register_assignment_for_parameters()  ← 分配寄存器
    → Function::gen_save_routine()         ← 生成寄存器保存代码
    → Function::gen_call_args()            ← 生成参数传递代码 (35KB)
    → Function::gen_restore_routine()      ← 生成寄存器恢复代码
    → Function::gen_new_code()             ← 生成完整新代码 (29KB，最大函数)
    → Function::compute_new_code_offsets() ← 重算所有分支目标偏移
    → Function::gen_patched_code()         ← 编码为最终二进制
  → Nvbit::patch_func()                   ← 将补丁代码安装到 GPU
  → Nvbit::config_patched_func()          ← 配置补丁函数执行
  → Nvbit::select_instrumented_or_orig()  ← 运行时选择原始/补丁版本
```

### 5.2 Trampoline 机制

对于每条被 instrumented 的指令，NVBit 生成一个 trampoline（跳板）：

```
原始代码:                    打补丁后的代码:
  ...                          ...
  LDG.E.64 R2, [R4]           JMP trampoline_13     ← 跳转到跳板
  ...                          ...

trampoline_13:                 ← 跳板代码（NVBit 生成）
  [保存寄存器到本地内存]         ← gen_save_routine()
  [设置参数: pred, opcode_id, addr, ...]  ← gen_call_args()
  JCAL count_instrs            ← 调用注入的设备函数
  [恢复寄存器]                  ← gen_restore_routine()
  LDG.E.64 R2, [R4]           ← 执行原始指令
  JMP next_instr               ← 跳回原始代码流
```

### 5.3 代码生成原语（HAL 提供）

每个 HAL 后端提供的 SASS 指令生成原语：

| 原语 | 功能 | 用途 |
|------|------|------|
| `gen_stl(RZ, offset, reg)` | STL（存储到本地内存）| 保存寄存器 |
| `gen_ldl(reg, offset)` | LDL（从本地内存加载）| 恢复寄存器 |
| `gen_ldc(reg, bank, offset)` | LDC（从常量内存加载）| 加载栈指针 |
| `gen_mov32i(reg, imm)` | MOV 立即数 | 设置常量参数 |
| `gen_iadd32i(reg, reg, imm)` | IADD 立即数 | 栈帧分配/释放 |
| `gen_jcal(target)` | JCAL（函数调用）| 调用注入函数 |
| `gen_jmp(target)` | JMP（无条件跳转）| trampoline 跳转 |
| `gen_jcal_pred` / `gen_jmp_pred` | 条件调用/跳转 | 谓词控制 |
| `gen_depbar()` | 依赖屏障 | 同步依赖 |

### 5.4 分支目标修正

由于插入了 trampoline 代码，所有指令的偏移量都发生了变化。`Function::compute_new_code_offsets()` 负责重新计算所有分支指令（BRA、BRX、BSSY、CALL、RET、JMP）的目标地址。

### 5.5 SASS 指令编解码器

NVBit 内置了多代 SASS 指令集的编解码器：

| 命名空间 | ISA 版本 | 对应架构 |
|----------|---------|---------|
| `NV::Sass7` | SM 7.x | Volta/Turing (sm_70, sm_72, sm_75) |
| `NV::Sass8` | SM 8.x | Ampere (sm_80, sm_86, sm_87) |
| `NV::Sass9` | SM 8.9 | Ada Lovelace (sm_89) |
| `NV::Sass10` | SM 9.0 | Hopper (sm_90) |
| `NV::Sass12` | SM 10.x/12.x | Blackwell (sm_100, sm_101, sm_120) |

每个命名空间提供：
- `DecodeOpcode(InstructionBits)` — 解码原始指令字为操作码枚举
- `GetField/SetField(bits, start, end)` — 位域提取/插入
- 各指令类的 encode/decode：`BRA`, `BRX`, `CALL_ABS_I`, `CALL_REL_I`, `CALL_REL_R`, `RET`, `JMP_I`, `WARPSYNC_*`, `BMOV_*`, `NANOSLEEP_I`, `LDC`, `LEPC` 等

---

## 6. 寄存器保存与恢复

### 6.1 问题

注入的设备函数（如 `count_instrs`）会使用 GPU 寄存器，可能覆盖应用原本正在使用的寄存器值。NVBit 必须在调用注入函数前保存、之后恢复所有受影响的寄存器。

### 6.2 保存的状态类型

```
SAVE_REGS_SPACE    — 通用寄存器 (R0-R255)
SAVE_UREG_SPACE    — 统一寄存器 (Uniform Registers)
SAVE_PRED_SPACE    — 谓词寄存器 (P0-P7)
SAVE_UPRED_SPACE   — 统一谓词寄存器
SAVE_CC_SPACE      — 条件码寄存器
SAVE_MREF_SPACE    — 内存引用地址
```

此外，还需要保存 CBU（Convergence Barrier Unit）状态，通过 `BMOV_CLEAR_RD` 和 `BMOV_PQUAD_R` 指令实现。

### 6.3 实现方式

保存/恢复使用本地内存（每线程私有的栈空间）：

```sass
# 保存阶段 (gen_save_routine)
LDC R1, c[0x0][stack_offset]     # 加载栈指针
IADD32I R1, R1, -sp_size         # 分配栈帧
STL [R1+0x00], R0                # 保存 R0
STL [R1+0x04], R2                # 保存 R2
...

# 调用注入函数
JCAL count_instrs

# 恢复阶段 (gen_restore_routine)
LDL R0, [R1+0x00]               # 恢复 R0
LDL R2, [R1+0x04]               # 恢复 R2
...
IADD32I R1, R1, sp_size          # 释放栈帧
```

### 6.4 寄存器限制

注入函数的寄存器使用不能超过 24 个（`-maxrregcount=24` 编译选项限制）。这避免了寄存器压力过大导致的 spill 问题。

### 6.5 预编译寄存器读写辅助内核

`nvbit_reg_rw.h` 中定义的寄存器读写函数使用 `TAKE_CODE_SPACE(N)` 宏预分配大量 GPU 指令空间：

```cpp
__device__ __noinline__ int32_t nvbit_read_reg(uint64_t reg_num) {
    TAKE_CODE_SPACE(1024);  // 预留 1024 条指令的空间
    return 0;
}
```

这些预留空间是"代码洞"（code cave），NVBit 在运行时用实际的寄存器访问指令替换这些占位代码。`nvbit_imp.o` 的 372KB data 段就包含了为所有 SM 架构预编译的这些辅助内核。

---

## 7. GPU-CPU 通信机制

NVBit 工具使用两种 GPU-CPU 通信模式：

### 7.1 模式 A：CUDA 托管内存（简单工具）

**使用工具**：`instr_count`, `instr_count_bb`, `instr_count_cuda_graph`, `opcode_hist`

```cpp
__managed__ uint64_t counter = 0;  // GPU/CPU 共享的托管内存

// GPU 端：原子累加
atomicAdd((unsigned long long*)&counter, count);

// CPU 端：同步后直接读取
cudaDeviceSynchronize();
printf("instructions: %ld\n", counter);
```

**优点**：实现简单，适合聚合统计。
**缺点**：只能做简单的原子操作，无法传输复杂数据。

### 7.2 模式 B：Channel 环形缓冲区（复杂工具）

**使用工具**：`mem_trace`, `mem_printf2`, `record_reg_vals`

这是一个**门铃驱动的填充-冲刷缓冲区**（Doorbell-based fill-and-flush buffer），而非经典环形缓冲区。

#### 架构图

```
  GPU 端 (ChannelDev)                        CPU 端 (ChannelHost)
  ┌───────────────────────┐                  ┌──────────────────────┐
  │ buff ──────────────── │ ─── PCIe ───→    │ recv_buffer          │
  │ buff_write_head_ptr   │                  │                      │
  │ buff_write_tail_ptr   │                  │ recv() {             │
  │ buff_end              │                  │   check doorbell     │
  │                       │                  │   cudaMemcpyAsync    │
  │ doorbell* ←──── host-mapped ────────→    │   clear doorbell     │
  │                       │                  │ }                    │
  │ push() {              │                  │                      │
  │   atomicAdd(head)     │                  │ 接收线程 (pthread)    │
  │   memcpy              │                  │   while(!done) {     │
  │   atomicAdd(tail)     │                  │     recv()           │
  │ }                     │                  │     process()        │
  │                       │                  │   }                  │
  │ flush() {             │                  │                      │
  │   __threadfence_sys() │                  │                      │
  │   *doorbell = nbytes  │                  │                      │
  │   spin(*doorbell==0)  │                  │                      │
  │ }                     │                  │                      │
  └───────────────────────┘                  └──────────────────────┘
```

#### push() 协议详解

```
Warp A 调用 push(data, 284 bytes):
  1. curr = atomicAdd(&head, 284)          ← 原子预留槽位
  2. if (curr + 284 > buff_end):           ← 缓冲区满？
       if (curr <= buff_end):              ← 我是第一个检测到溢出的 warp
         spin_wait(tail == curr)           ← 等待所有先前写入完成
         flush()                           ← 冲刷缓冲区
       else:                               ← 后续检测到溢出的 warp
         spin_wait(head <= buff_end)       ← 等待冲刷完成
       retry                               ← 重试
  3. memcpy(curr, data, 284)               ← 拷贝数据
  4. atomicAdd(&tail, 284)                 ← 确认写入完成
```

#### flush() 协议详解

```
flush():
  1. nbytes = tail - buff                  ← 计算已缓冲字节数
  2. __threadfence_system()                ← 确保所有写入对 PCIe 可见
  3. *doorbell = nbytes                    ← 通知 CPU（host-mapped 内存）
  4. spin_wait(*doorbell == 0)             ← 等待 CPU 读取完成
  5. tail = buff                           ← 重置尾指针
  6. __threadfence()                       ← 内存屏障
  7. head = buff                           ← 重置头指针
```

关键：步骤 5 先重置 `tail`，再在屏障后重置 `head`。其他在溢出处自旋的 warp 监视 `head`，这个顺序保证它们在看到 `head` 重置时，`tail` 已经安全。

#### 门铃（Doorbell）

Doorbell 是一个 host-mapped 的整数，通过 `cudaHostAlloc` + `cudaHostAllocMapped` 分配，同时在 CPU 和 GPU 端可见。GPU 写入 doorbell 通知 CPU 有数据可读，CPU 清零 doorbell 通知 GPU 可以继续写入。

#### 内核结束时的冲刷

每个内核执行完成后，工具会启动一个单线程内核来冲刷剩余数据：

```cpp
// 在 nvbit_at_cuda_event (is_exit=1) 中
flush_channel<<<1,1>>>(ctx_state->channel_dev);
cudaDeviceSynchronize();
```

#### 性能特征

- **缓冲区大小**：通常 1MB (`CHANNEL_SIZE = 1 << 20`)
- **阻塞性**：GPU 在 flush 期间自旋等待 CPU，性能受限于 PCIe 传输速度
- **数据包大小**：`mem_access_t` = 284 字节/每次内存访问事件
- **并发安全**：通过 atomicAdd 头指针实现无锁多 warp 并发写入

---

## 8. ELF/Cubin 处理

GPU 的编译产物（cubin）是 ELF 格式。NVBit 包含完整的 ELF 处理库。

### 8.1 核心 ELF 类型

```cpp
NV::Symbolics::Elf<Elf32Types, true>    // 只读 32位 ELF
NV::Symbolics::Elf<Elf64Types, true>    // 只读 64位 ELF
NV::Symbolics::Elf<Elf32Types, false>   // 可变 32位 ELF（用于打补丁）
NV::Symbolics::Elf<Elf64Types, false>   // 可变 64位 ELF
```

### 8.2 关键 ELF 操作

| 函数 | 功能 |
|------|------|
| `toolsElf64GetTextSectionContents()` | 提取内核代码段 |
| `toolsElf64GetCudaSMVersion()` | 获取 cubin 的 SM 版本 |
| `toolsElf64ListKernelNames()` | 列举所有内核函数名 |
| `toolsElf64GetCallgraphFuncs()` | 构建函数调用图 |
| `toolsElf64GetRelatedFuncs()` | 获取内核调用的设备函数 |
| `toolsElf64GetLineInfo()` | 提取源码行号信息 |
| `toolsElf64GetDebugFrame()` | 解析 DWARF 调试帧 |
| `toolsElf64GetAtomSysInstrOffsets()` | 定位系统级原子指令 |
| `toolsElf64GetCoopGroupInstrOffsets()` | 定位协作组指令 |
| `toolsElf64GetWarpWideInstrOffsets()` | 定位 warp 级指令 |
| `toolsElf64StripHiddenSymbols()` | 去除隐藏符号 |

### 8.3 处理的 ELF 段

```
.nv.info             — NVIDIA 特定的内核元数据
.nv.constant         — 常量内存段
.text.<function>     — 各内核函数的代码段
.nv_fatbin           — 嵌入的 fat binary 容器
```

### 8.4 缓存机制

NVBit 维护多级缓存来避免重复解析：

- **ELF 模块缓存**：`elfModuleHashMap` 全局哈希表
- **行号缓存**：`insertIntoLineCache()` / `searchLineCache()`
- **内核缓存**：`searchKernelCacheByName()` / `searchKernelCacheByRelocLoc()`
- **重定位缓存**：`createElfReloc()` / `getRelocByOffset()`

---

## 9. 硬件抽象层 (HAL)

### 9.1 HAL 架构

每个 GPU 代际有独立的 HAL 后端，通过函数指针表 `hal_t` 提供统一接口：

```
init_hal_XXX(hal_t* hal) → 填充函数指针表
```

各后端的代码量反映了 SASS ISA 的复杂度演进：

| HAL 后端 | GPU 架构 | SM 版本 | 代码大小 |
|----------|---------|---------|---------|
| `gk11x_hal` | Kepler | sm_35-37 | 10 KB |
| `gm10x_hal` | Maxwell | sm_50-53 | 10 KB |
| `gv10x_hal` | Volta | sm_70 | 35 KB |
| `gv11x_hal` | Volta+ | sm_72-75 | 35 KB |
| `tu10x_hal` | Turing | sm_75 | 41 KB |
| `ga10x_hal` | Ampere | sm_80-86 | 45 KB |
| `gh10x_hal` | Hopper | sm_89-90 | 54 KB |
| `gb10x_hal` | Blackwell | sm_100 | 54 KB |
| `gb12x_hal` | Blackwell+ | sm_120 | 54 KB |

从 Kepler 的 10KB 到 Blackwell 的 54KB，指令编码复杂度增加了 5 倍多。

### 9.2 添加新架构支持

要支持新的 GPU 架构，需要：
1. 在 HAL 层添加新的 `init_hal_XXX()` 实现
2. 在 SASS 编解码器中添加新的 `NV::SassN` 命名空间
3. 将新 HAL 注册到 `Nvbit::init()` 的初始化链中

这是 NVBit 最主要的扩展点之一，但由于 `libnvbit.a` 是闭源的，外部开发者无法直接添加 HAL 后端。

---

## 10. libnvbit.a 内部结构

### 10.1 目标文件列表

```
ar t core/libnvbit.a 输出：

nvbit.o                         ← 公开 API 包装
nvbit_imp.o                     ← 核心引擎（最大，627KB）
function.o                      ← Function 类（代码生成）
instr.o                         ← SassInstr 类（指令解析）
Elf.o                           ← ELF 解析库
gk11x_hal.o ~ gb12x_hal.o      ← 9个 HAL 后端
tools_shared_readelf32.o        ← 32位 ELF 读取
tools_shared_readelf64.o        ← 64位 ELF 读取
tools_shared_readelf_common.o   ← ELF 通用函数
tools_shared_hashmap.o          ← 哈希映射
tools_shared_hashset.o          ← 哈希集合
tools_shared_list.o             ← 链表
tools_shared_rangemap.o         ← 区间映射
tools_shared_rbtr.o             ← 红黑树
```

### 10.2 核心类层次

```
Nvbit (单例)
  ├─ callback()                   → 回调分发
  ├─ module_loaded()              → 模块加载处理
  ├─ func_loading()               → 函数加载处理
  ├─ disassemble_func()           → 反汇编
  ├─ patch_func()                 → 安装补丁
  ├─ config_patched_func()        → 配置补丁函数
  ├─ select_instrumented_or_orig()→ 运行时代码选择
  ├─ build_callgraph_from_elf()   → 构建调用图
  ├─ get_function()               → 获取 Function 对象
  ├─ normalize_cufunc()           → 规范化 CUfunction 句柄
  ├─ create_ctx() / destroy_ctx() → 上下文管理
  └─ init() / term()              → 初始化/终止

Function
  ├─ add_orig_instr()             → 添加原始指令
  ├─ register_assignment_for_parameters() → 寄存器分配
  ├─ gen_save_routine()           → 生成保存代码
  ├─ gen_call_args()              → 生成参数代码
  ├─ gen_restore_routine()        → 生成恢复代码
  ├─ gen_new_code()               → 生成新代码
  ├─ gen_patched_code()           → 编码最终二进制
  ├─ gen_CFG()                    → 生成控制流图
  ├─ compute_new_code_offsets()   → 重算分支偏移
  └─ dump_sass()                  → 输出 SASS

SassInstr
  ├─ decode()                     → 解码 SASS 文本
  ├─ encode()                     → 编码为二进制
  ├─ tokenize()                   → 词法分析
  ├─ to_str()                     → 转为字符串
  └─ get_reg() / gen_reg()        → 寄存器操作
```

### 10.3 关键 IPC 机制

`libnvbit.a` 使用以下 IPC 与 CUDA 驱动通信：

```
动态符号依赖:
  dladdr, dlvsym, dlmopen     — 动态符号解析
  mkstemp                     — 安全创建临时文件
  socket, bind, recvmsg       — 与 CUDA 驱动的 socket 通信
  shmget                      — 共享内存
  mkfifo                      — 命名管道
  syscall                     — 直接系统调用
```

---

## 11. 工具 .so 的二进制结构

### 11.1 段布局

以 `instr_count.so` (2.6MB) 为例：

| 段 | 大小 | 内容 |
|---|------|------|
| `.text` | 1.12 MB | 可执行代码（NVBit 核心 + 工具逻辑）|
| `.rodata` | 115 KB | 只读数据、字符串常量 |
| `.data` | 380 KB | 可写数据（预编译 GPU 内核）|
| `.nv_fatbin` | 80.9 KB | 嵌入的 CUDA fat binary |
| `.eh_frame` | 148 KB | 异常处理帧 |
| `.plt` | 24 KB | PLT（604 个条目）|
| `.init_array` | 80 B | 10 个构造函数指针 |
| `.bss` | 12 KB | 未初始化数据 |
| `.tbss` | 8 KB | 线程本地存储 |

### 11.2 嵌入的 Fat Binary

每个工具 `.so` 是一个"胖二进制"，包含 **17-18 个架构的预编译 cubin**：

```
cuobjdump 输出:
  sm_50, sm_52, sm_53, sm_60, sm_61, sm_62,
  sm_70, sm_72, sm_75,
  sm_80, sm_86, sm_87, sm_89, sm_90,
  sm_100, sm_101, sm_120
  + PTX 后备 (sm_52, sm_120)
```

这确保了工具的设备函数（如 `count_instrs`）在任何支持的 GPU 上都能运行。

### 11.3 CUDA 注册机制

工具 `.so` 通过标准 CUDA 注册 API 注册其设备代码：

```
__cudaRegisterFatBinary        ← 注册 fat binary
__cudaRegisterFunction         ← 注册设备函数
__cudaRegisterVar              ← 注册设备变量
__cudaRegisterManagedVar       ← 注册托管变量
__cudaUnregisterFatBinary      ← 卸载时取消注册
```

---

## 12. 完整运行时序

以 `instr_count` 工具分析 `vectoradd` 为例的完整时序：

```
═══════════════════════════════════════════════════════════
 阶段一：加载和初始化
═══════════════════════════════════════════════════════════

应用启动
  │
  ├─ 动态链接器加载 instr_count.so (LD_PRELOAD)
  │   ├─ 依赖: libcuda.so.1 (88MB), libcudart.so.12, libstdc++.so.6
  │   └─ instr_count.so 最后初始化 (在 libcuda.so.1 之后)
  │
  ├─ .init_array 构造函数执行
  │   ├─ __sti____cudaRegisterAll()  → 注册 fatbin
  │   └─ NVBit 内部初始化
  │
  ├─ Nvbit::init()
  │   ├─ toolsElfLibInitialize()     → ELF 库初始化
  │   ├─ init_hal_gk11x ~ gb12x()   → 初始化所有 9 个 HAL 后端
  │   └─ cuGetExportTable()          → 注册 nvbitToolsCallbackFunc
  │
  └─ nvbit_at_init()                  → 工具打印 banner，读取环境变量

═══════════════════════════════════════════════════════════
 阶段二：CUDA 上下文创建
═══════════════════════════════════════════════════════════

应用调用 cudaMalloc() (首次 CUDA 调用)
  │
  ├─ CUDA 驱动打开 /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm
  ├─ 创建 GPU 上下文
  │   ├─ 分配命令通道 (每个 4KB MMIO 共享映射，约 8 个)
  │   ├─ 预留 ~8.6GB 虚拟地址空间 (PROT_NONE, 不占物理内存)
  │   └─ 创建 3 个额外线程: CUDA 内部线程 + cuda-EvtHandlr
  │
  ├─ NVBit 回调: nvbit_at_ctx_init()
  │
  ├─ cuobjdump 提取工具 cubin (首次)
  │   └─ cp instr_count.so /tmp/nvbit_tool_tmpdir.XXX
  │       cuobjdump -arch sm_120 -xelf all
  │       → /tmp/nvbit_tool_tmpdir.XXX/instr_count.1.sm_120.cubin
  │
  └─ nvbit_tool_init() (如果定义了)

═══════════════════════════════════════════════════════════
 阶段三：内核启动和 Instrumentation
═══════════════════════════════════════════════════════════

应用调用 cudaLaunchKernel(vecAdd, <<<98,1024>>>)
  │
  ├─ CUDA 驱动触发回调 (is_exit=0, cbid=cuLaunchKernel)
  │   └─ nvbit_at_cuda_event(ctx, is_exit=0, ...)
  │
  ├─ instrument_function_if_needed(ctx, func)
  │   │
  │   ├─ nvbit_get_related_functions(ctx, func)
  │   │   → 获取所有相关设备函数
  │   │
  │   ├─ nvbit_get_instrs(ctx, func)
  │   │   └─ Nvbit::disassemble_func()
  │   │       ├─ 写入 /tmp/nvbit_code_XXXXXX   (原始二进制)
  │   │       ├─ 执行: nvdisasm -b SM120 /tmp/nvbit_code_XXXXXX
  │   │       │         > /tmp/nvbit_sass_XXXXXX
  │   │       ├─ 解析 SASS 文本 → 32 个 Instr 对象
  │   │       └─ 清理临时文件
  │   │
  │   └─ 对每条指令:
  │       ├─ nvbit_insert_call(instr, "count_instrs", IPOINT_BEFORE)
  │       ├─ nvbit_add_call_arg_guard_pred_val(instr)
  │       ├─ nvbit_add_call_arg_const_val32(instr, count_warp_level)
  │       └─ nvbit_add_call_arg_const_val64(instr, &counter)
  │
  ├─ nvbit_enable_instrumented(ctx, func, true)
  │   └─ Nvbit::patch_func()
  │       ├─ Function::gen_save_routine()         → 寄存器保存代码
  │       ├─ Function::gen_call_args()            → 参数传递代码
  │       ├─ Function::gen_restore_routine()      → 寄存器恢复代码
  │       ├─ Function::gen_new_code()             → 完整新代码
  │       ├─ Function::compute_new_code_offsets() → 修正分支目标
  │       ├─ Function::gen_patched_code()         → 编码为 SASS 二进制
  │       └─ 上传补丁 cubin 到 GPU
  │
  ├─ Nvbit::config_patched_func()
  │   └─ Nvbit::select_instrumented_or_orig()
  │       → 切换到补丁版本执行
  │
  └─ counter = 0                  → 重置计数器

═══════════════════════════════════════════════════════════
 阶段四：GPU 执行 (补丁后的内核)
═══════════════════════════════════════════════════════════

98 个线程块 × 1024 线程 在 GPU 上执行:

  对于 vecAdd 内核中的每条指令:
    │
    ├─ [NVBit trampoline 开始]
    │   ├─ 保存寄存器到本地内存
    │   ├─ 设置参数 (pred, count_warp_level, &counter)
    │   └─ JCAL count_instrs
    │
    ├─ count_instrs() 执行:
    │   ├─ active_mask = __ballot_sync(__activemask(), 1)
    │   ├─ predicate_mask = __ballot_sync(__activemask(), pred)
    │   ├─ laneid = get_laneid()
    │   ├─ first_lane = __ffs(active_mask) - 1
    │   └─ if (laneid == first_lane):
    │       atomicAdd(&counter, 1)    // warp 级：每 warp 加 1
    │
    ├─ [NVBit trampoline 结束]
    │   └─ 恢复寄存器
    │
    └─ 执行原始指令

═══════════════════════════════════════════════════════════
 阶段五：结果收集
═══════════════════════════════════════════════════════════

CUDA 驱动触发回调 (is_exit=1, cbid=cuLaunchKernel)
  │
  ├─ nvbit_at_cuda_event(ctx, is_exit=1, ...)
  │   ├─ cudaDeviceSynchronize()       → 等待 GPU 完成
  │   ├─ 读取 counter = 62588         → 从托管内存
  │   └─ printf("kernel 0 - _Z6vecAddPdS_S_i - #thread-blocks 98,
  │              kernel instructions 62588, total instructions 62588")
  │
  └─ tot_app_instrs += counter

═══════════════════════════════════════════════════════════
 阶段六：清理和退出
═══════════════════════════════════════════════════════════

应用退出
  ├─ nvbit_at_ctx_term(ctx)       → 上下文清理
  ├─ nvbit_at_term()              → 打印 "Total app instructions: 62588"
  ├─ Nvbit::~Nvbit()              → 析构单例
  └─ toolsElfLibFinalize()        → ELF 库清理
```

---

## 13. 关键 API 参考

### 13.1 工具回调 API（工具实现，NVBit 调用）

| 回调 | 必须? | 调用时机 | 注意事项 |
|------|-------|---------|---------|
| `nvbit_at_init()` | 是 | 工具加载时 | 不要进行 CUDA 内存分配 |
| `nvbit_at_term()` | 是 | 程序退出时 | 最终统计输出 |
| `nvbit_at_cuda_event()` | 是 | 每次 CUDA API 调用 | is_exit=0 入口, is_exit=1 出口 |
| `nvbit_at_ctx_init()` | 否 | CUDA 上下文创建 | **不要** CUDA 内存分配，会死锁 |
| `nvbit_at_ctx_term()` | 否 | 上下文销毁 | 清理 channel 等资源 |
| `nvbit_tool_init()` | 否 | 首次内核启动前 | **安全**进行 CUDA 内存分配 |
| `nvbit_at_graph_node_launch()` | 否 | CUDA Graph 节点启动 | 配合 `nvbit_set_at_launch` |

### 13.2 指令检查 API

```cpp
// 获取函数的所有指令
const std::vector<Instr*>& nvbit_get_instrs(CUcontext ctx, CUfunction func);

// 获取控制流图
const CFG_t& nvbit_get_CFG(CUcontext ctx, CUfunction func);

// 获取相关设备函数
std::vector<CUfunction> nvbit_get_related_functions(CUcontext ctx, CUfunction func);

// 获取函数名（mangled=true 返回修饰名）
const char* nvbit_get_func_name(CUcontext ctx, CUfunction f, bool mangled = false);

// 获取源码行号信息
bool nvbit_get_line_info(CUcontext ctx, CUfunction func, uint32_t offset,
                         char** file_name, char** dir_name, uint32_t* line);

// 获取函数地址、判断是否为内核
uint64_t nvbit_get_func_addr(CUcontext ctx, CUfunction func);
bool nvbit_is_func_kernel(CUcontext ctx, CUfunction func);

// 获取函数配置（网格/块维度、寄存器数等）
void nvbit_get_func_config(CUcontext ctx, CUfunction func, func_config_t *config);

// dump cubin 到文件
bool nvbit_dump_cubin(CUcontext ctx, CUfunction func, const char *filename);

// 获取 SM 计算能力
uint32_t nvbit_get_sm_family(CUcontext ctx);
```

### 13.3 Instr 类方法

```cpp
const char* getSass();               // 完整 SASS 字符串
const char* getOpcode();             // 完整操作码 (如 "LDG.E.64")
const char* getOpcodeShort();        // 短操作码 (如 "LDG")
uint32_t getOffset();                // 字节偏移
uint32_t getIdx();                   // 指令索引
bool isLoad() / isStore();           // 是否为内存操作
int getSize();                       // 访问大小（字节）
InstrType::MemorySpace getMemorySpace(); // 内存空间类型
int getNumOperands();                // 操作数数量
const operand_t* getOperand(int n);  // 获取操作数
bool hasPred();                      // 是否有谓词
int getPredNum();                    // 谓词号
bool isPredNeg();                    // 谓词是否取反
void printDecoded();                 // 打印解码信息
```

### 13.4 注入 API

```cpp
// 在指令前/后插入设备函数调用
void nvbit_insert_call(const Instr* instr, const char* func_name, ipoint_t point);

// 添加参数
void nvbit_add_call_arg_guard_pred_val(const Instr* instr);      // 保护谓词值
void nvbit_add_call_arg_const_val32(const Instr* instr, uint32_t val);  // 32位常量
void nvbit_add_call_arg_const_val64(const Instr* instr, uint64_t val);  // 64位常量
void nvbit_add_call_arg_reg_val(const Instr* instr, int reg_num);       // 寄存器值
void nvbit_add_call_arg_ureg_val(const Instr* instr, int reg_num);      // 统一寄存器值
void nvbit_add_call_arg_mref_addr64(const Instr* instr, int id = 0);    // 内存引用地址
void nvbit_add_call_arg_launch_val64(const Instr* instr, int offset);   // 启动时参数
void nvbit_add_call_arg_cbank_val(const Instr* instr, int bank, int off); // 常量bank值
void nvbit_add_call_arg_pred_val_at(const Instr* instr, int pred_num);  // 特定谓词值

// 删除原始指令（如 mov_replace 工具）
void nvbit_remove_orig(const Instr* instr);

// 运行时控制
void nvbit_enable_instrumented(CUcontext ctx, CUfunction func, bool flag);
void nvbit_set_at_launch(CUcontext ctx, CUfunction func, uint64_t val, ...);

// 线程管理
void nvbit_set_tool_pthread(pthread_t t);    // 注册工具线程（避免触发回调）
void nvbit_unset_tool_pthread(pthread_t t);
```

### 13.5 数据结构

```cpp
// 控制流图
typedef struct { std::vector<Instr*> instrs; } basic_block_t;
typedef struct {
    bool is_degenerate;               // CFG 是否退化（动态跳转）
    std::vector<basic_block_t*> bbs;  // 基本块列表
} CFG_t;

// 函数配置
typedef struct {
    uint32_t blockDimX, blockDimY, blockDimZ;
    uint32_t gridDimX, gridDimY, gridDimZ;
    uint32_t shmem_static_nbytes, shmem_dynamic_nbytes;
    uint32_t num_registers;
} func_config_t;

// 操作数类型
enum class OperandType {
    IMM_UINT64, IMM_DOUBLE, REG, PRED, UREG, UPRED,
    CBANK, MREF, GENERIC, MEM_DESC
};

// 内存空间
enum class MemorySpace {
    NONE, LOCAL, GENERIC, GLOBAL, SHARED, CONSTANT,
    GLOBAL_TO_SHARED, SURFACE, TEXTURE,
    DISTRIBUTED_SHARED, TENSOR_MEM, TENSOR_CORE_MEM
};
```

---

## 14. 扩展开发指南

### 14.1 创建新工具的步骤

```bash
# 1. 复制模板
cp -r tools/instr_count tools/my_tool
cd tools/my_tool

# 2. 重命名文件
mv instr_count.cu my_tool.cu

# 3. 修改 Makefile
# 更改 NVBIT_TOOL 变量为 my_tool.so
# 更改源文件列表
```

### 14.2 工具基础骨架

**my_tool.cu（主机端）**：

```cpp
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
#include "common.h"  // 如需自定义数据结构

// 全局状态
__managed__ uint64_t my_counter = 0;
std::unordered_set<CUfunction> already_instrumented;
pthread_mutex_t mutex;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbose output");
    pthread_mutex_init(&mutex, NULL);
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    auto related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;

        const auto& instrs = nvbit_get_instrs(ctx, f);
        for (auto instr : instrs) {
            // 你的过滤逻辑：选择要 instrument 的指令
            if (!instr->isLoad() && !instr->isStore()) continue;

            nvbit_insert_call(instr, "my_instrument_func", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&my_counter);
            // 添加更多参数...
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            // 获取 CUfunction（根据 cbid 不同用不同方式）
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            instrument_function_if_needed(ctx, p->f);
            nvbit_enable_instrumented(ctx, p->f, true);
            my_counter = 0;
            pthread_mutex_unlock(&mutex);
        } else {
            pthread_mutex_lock(&mutex);
            cudaDeviceSynchronize();
            printf("Result: %ld\n", my_counter);
            pthread_mutex_unlock(&mutex);
        }
    }
}

void nvbit_at_term() {
    printf("Done.\n");
}
```

**inject_funcs.cu（设备端）**：

```cpp
#include <stdint.h>
#include "utils/utils.h"

// 必须声明为 extern "C" __device__ __noinline__
extern "C" __device__ __noinline__ void my_instrument_func(
    int pred,           // 来自 nvbit_add_call_arg_guard_pred_val
    uint64_t pcounter)  // 来自 nvbit_add_call_arg_const_val64
{
    if (!pred) return;  // 谓词为假则跳过

    // Warp 级聚合（减少原子操作冲突）
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    if (laneid == first_laneid) {
        atomicAdd((unsigned long long*)pcounter, 1);
    }
}
```

### 14.3 Makefile 关键编译选项

```makefile
# inject_funcs.cu 必须使用这些特殊标志
$(NVCC) $(INCLUDES) -Xptxas -astoolspatch --keep-device-functions \
        -arch=$(ARCH) -Xcompiler -fPIC -c inject_funcs.cu -o inject_funcs.o

# -Xptxas -astoolspatch   : 启用 AST 工具补丁模式（NVBit 必需）
# --keep-device-functions  : 防止优化器删除未引用的设备函数
# -arch=$(ARCH)            : 为所有目标架构生成代码

# 最终链接必须用 g++，不能用 nvcc
g++ -shared -fPIC $(OBJECTS) -L$(NVBIT_PATH) -lnvbit \
    -L$(CUDA_LIB) -lcuda -lcudart_static -lpthread -ldl -o my_tool.so
```

### 14.4 常见扩展方向

| 方向 | 说明 | 参考工具 |
|------|------|---------|
| 自定义指令计数 | 按条件统计特定类型指令 | `instr_count`, `opcode_hist` |
| 内存访问分析 | 追踪全局/共享内存访问模式 | `mem_trace` |
| 数据流分析 | 记录寄存器值追踪数据流 | `record_reg_vals` |
| 指令替换 | 修改 GPU 代码行为 | `mov_replace` |
| 性能分析 | 基本块级计数降低开销 | `instr_count_bb` |
| CUDA Graph 支持 | 支持图模式的内核分析 | `instr_count_cuda_graph` |
| Channel 通信 | 传输复杂结构化数据 | `mem_trace`, `record_reg_vals` |

### 14.5 扩展限制

由于 `libnvbit.a` 是闭源的，以下扩展**不可能**直接实现：

1. **添加新 GPU 架构的 HAL 后端** — 需要 NVIDIA 更新 libnvbit.a
2. **修改补丁引擎的行为** — 代码生成逻辑在闭源库中
3. **修改 CUDA API 拦截的范围** — 回调订阅在初始化时完成
4. **替换 nvdisasm 依赖** — 反汇编路径硬编码

可以做的扩展：

1. **新工具** — 组合现有 API 实现新的分析工具
2. **自定义设备函数** — inject_funcs.cu 完全可控
3. **自定义通信机制** — 可以不用 Channel，使用自己的方案
4. **后处理** — 收集数据后做任意分析
5. **与其他系统集成** — 通过工具的主机端代码连接外部系统

---

## 15. strace 实证分析

### 15.1 系统调用统计

对 `instr_count` + `vectoradd` 的一次运行：

```
总系统调用数: ~458
子进程数:     11 (shell + which + cuobjdump + nvdisasm)
线程数:       4  (主线程 + 2 CUDA内部 + cuda-EvtHandlr)
ioctl 调用:   427 (GPU 资源管理)
mmap 调用:    100 (内存映射)
openat 调用:  111 (文件操作)
```

### 15.2 关键设备文件

| 设备 | 功能 | 打开次数 |
|------|------|---------|
| `/dev/nvidiactl` | NVIDIA 控制设备 | 多次 (fd 8+) |
| `/dev/nvidia0` | GPU 0 设备 | ~20次 (不同 fd) |
| `/dev/nvidia-uvm` | 统一虚拟内存 | 2次 (fd 9,10) |

### 15.3 内存映射模式

```
阶段一 - 库加载:
  libcuda.so.1:  88MB 映射 (PROT_READ, PROT_READ|PROT_EXEC)
  libcudart.so.12: ~5MB 映射

阶段二 - CUDA 初始化:
  134MB 匿名映射 (PROT_NONE) — CUDA 内部堆
  ~8.6GB 预留 (PROT_NONE @ 0x727428000000) — GPU 地址空间
  ~4.3GB 预留 (PROT_NONE @ 0x200000000) — 设备映射内存

阶段三 - GPU 通道:
  8× 4KB 共享映射 — GPU 命令提交通道 (doorbell 寄存器)

阶段四 - 内核执行:
  2MB MAP_SHARED|MAP_FIXED — GPU MMIO 区域
  2MB 匿名共享映射 — 主机-设备共享缓冲区
```

注意：8.6GB + 4.3GB 的预留是虚拟地址预留（PROT_NONE），不消耗物理内存。

### 15.4 ioctl 命令模式

```
0x46, 0x2a — GPU 资源管理（最频繁）
0x46, 0x2b — GPU 资源分配/绑定
0x46, 0x4e — 内存对象创建（总是跟着 mmap）
0x46, 0xc9 — GPU 初始化/配置
0x46, 0xd6 — 驱动参数查询
UVM ioctls  — 内存注册、映射、页面管理
```

### 15.5 Compute Cache

NVBit/CUDA 使用 `~/.nv/ComputeCache/` 缓存 JIT 编译结果，避免重复编译。

---

## 附录：环境变量速查表

### NVBit 核心环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NVDISASM` | `nvdisasm` | 覆盖 nvdisasm 路径 |
| `NOBANNER` | `0` | 禁止打印 NVBit banner |
| `NO_EAGER_LOAD` | `0` | 关闭急切模块加载 |
| `ACK_CTX_INIT_LIMITATION` | `0` | 抑制 ctx_init 警告 |

### 工具通用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INSTR_BEGIN` | `0` | 开始 instrument 的指令索引 |
| `INSTR_END` | `UINT32_MAX` | 结束 instrument 的指令索引 |
| `START_GRID_NUM` / `KERNEL_BEGIN` | `0` | 开始 instrument 的内核编号 |
| `END_GRID_NUM` / `KERNEL_END` | `UINT32_MAX` | 结束 instrument 的内核编号 |
| `TOOL_VERBOSE` | `0` | 启用详细输出 |
| `COUNT_WARP_LEVEL` | `1` | 1=warp级计数, 0=线程级计数 |
| `EXCLUDE_PRED_OFF` | `0` | 排除谓词关闭的指令 |
| `ACTIVE_FROM_START` | `1` | 1=立即开始, 0=等待 cuProfilerStart |
| `MANGLED_NAMES` | `1` | 1=打印修饰名, 0=反修饰名 |
