/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "op128.h"

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS - 1)

template <typename T, typename RedOp, typename Fan, int Direct, int P2p, bool isNetOffload>
class Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload>
  : public PrimitivesWithoutDirect<Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload>> {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input = 0, Output = 1;
  RedOp redOp;
  const int tid;       // thread index in primitives group
  const int nthreads;  // thread count in primitives group
  const int wid;       // lane index in warp
  const int stepSize;
  const int warp;         // warp index in primitives group
  const int warpInBlock;  // warp index in thread block
  const bool flagThread;
  const int group;
  Fan fan;
  T* userBufs[2];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile struct ncclConnFifo* sendConnFifo = NULL;  // 记录发送了多少数据量等信息的fifo
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache;  // Cache last seen value

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  uint64_t* recvBuff[MaxRecv];
  uint64_t* sendBuff[MaxSend];

  inline __device__ int recvOffset(int i) {
    return (recvStep[i] % NCCL_STEPS) * stepSize;
  }
  inline __device__ int sendOffset(int i) {
    return (sendStep[i] % NCCL_STEPS) * stepSize;
  }
  inline __device__ uint64_t* recvPtr(int i) {
    return recvBuff[i] + recvOffset(i);
  }
  inline __device__ uint64_t* sendPtr(int i) {
    return sendBuff[i] + sendOffset(i);
  }
  inline __device__ uint64_t recvFlag(int i) {
    return recvStep[i] + 1;
  }
  inline __device__ uint64_t sendFlag(int i) {
    return sendStep[i] + 1;
  }

  inline __device__ void barrier() {
    barrier_sync(15 - group, nthreads);
  }

  int abort = 0;

  inline __device__ void waitSend(int nbytes) {
    if (sendConnHeadPtr) {
      // sendConnHeadPtr head 指针, 对端要写过来, sendConnHeadCache 本地缓存
      int spins = 0;
      while (sendConnHead + 1 >
             sendConnHeadCache + NCCL_STEPS) {  // 即将要写入的位置 > 当前对端消费到的位置 + buffer 长度
        sendConnHeadCache = *sendConnHeadPtr;   // 轮询
        if (checkAbort(abort, 1, spins)) break;
      }
      if (sendConnFifo) {
        sendConnFifo[sendStep[wid] % NCCL_STEPS].size = nbytes;  // ring 算法只有 tid == 0 的线程才能到这里~
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) {
#if __CUDA_ARCH__ >= 900
      __threadfence_system();
#else
      __threadfence();
#endif
      *sendConnTailPtr = sendConnTail += 1;
    }
  }

  template <int WordPerThread>
  __device__ __forceinline__ void loadRegsBegin(uint64_t (&regs)[WordPerThread], T const* src, int eltN) {
    // WordPerThread 在调用时没有显式传入，是因为编译器从 uint64_t regs[8] 自动推导出 WordPerThread = 8
    constexpr int EltPer16B = 16 / sizeof(T);  // 16 / 4 = 4
    if (reinterpret_cast<uintptr_t>(src) % 16 == 0) {
/* We are aligned to 16 bytes, so load directly to registers no shmem.
       * Flag threads load half as much data which gets shuffled to the even
       * registers during Finish. The point of splitting into two phases is to
       * defer that shuffle, which incurs a dependency stall, until after other
       * memops are launched by the caller.
       */
      NVCC_PRAGMA_UNROLL_AUTO
      // 如果地址已经对齐到 16 字节，则直接从 global memory 用 128-bit load 到 数组 regs 中
      for (int g = 0; g < WordPerThread / 2; g++) {
        int ix = g * WARP_SIZE - 4 * (g / 2) + wid - (g % 2) * (wid / 8);  // 读第几个 16B 数据块
        if (!flagThread || g % 2 == 0) {
          // 普通线程每次 load 16B数据，flag线程只在 g = 0，2 时 load 16B数据
          // 只要这个 16B chunk 的第一个元素还在有效范围内，就整 16B 读入
          if (ix * EltPer16B < eltN) load128((uint64_t*)(src + ix * EltPer16B), regs[2 * g + 0], regs[2 * g + 1]);
        }
      }
    } else {
      // Not aligned. Stage the smallest 16 byte aligned region subsuming the
      // buffer into shmem.
      int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
      uint64_t* src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));
      uint64_t* shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
      NVCC_PRAGMA_UNROLL_AUTO
      for (int g = 0; g < WordPerThread / 2; g++)
        if ((g * WARP_SIZE + wid) * 16 < misalignment + eltN * sizeof(T))
          load128(src8 + 2 * (g * WARP_SIZE + wid), regs[2 * g + 0], regs[2 * g + 1]);
      NVCC_PRAGMA_UNROLL_AUTO
      for (int g = 0; g < WordPerThread / 2; g++)
        storeShmem128(shm8 + 2 * (g * WARP_SIZE + wid), regs[2 * g + 0], regs[2 * g + 1]);

      __syncwarp();

      // Now load from shmem stage to regs. Preserve the same pre-shuffled layout
      // as the aligned case since Finish() will be applied regardless.
      T* shm = (T*)shm8 + misalignment / sizeof(T);
      NVCC_PRAGMA_UNROLL_AUTO
      for (int g = 0; g < WordPerThread / 2; g++) {
        int ix = g * WARP_SIZE - 4 * (g / 2) + wid - (g % 2) * (wid / 8);
        if (!flagThread || g % 2 == 0) {
          if (ix * EltPer16B < eltN) loadShmemMisaligned128(shm + ix * EltPer16B, regs[2 * g + 0], regs[2 * g + 1]);
        }
      }
    }
  }

  template <int WordPerThread>
  __device__ __forceinline__ void loadRegsFinish(uint64_t (&regs)[WordPerThread]) {
// Move data out of flag registers into the vacant registers.
    NVCC_PRAGMA_UNROLL_AUTO
    for (int g = 1; g < WordPerThread / 2; g += 2) {
      if (flagThread) regs[2 * g] = regs[2 * g - 1];
    }
  }

  template <int WordPerThread>
  __device__ __forceinline__ void storeRegs(T* dst, uint64_t (&regs)[WordPerThread], int eltN) {
    constexpr int EltPer16B = 16 / sizeof(T);
// Reverse Finish() register permuatation.
    NVCC_PRAGMA_UNROLL_AUTO
    for (int g = 1; g < WordPerThread / 2; g += 2) {
      if (flagThread) regs[2 * g - 1] = regs[2 * g];
    }
    // Write to dst if 16-byte aligned, shmem otherwise.
    int misalignment = reinterpret_cast<uintptr_t>(dst) % 16;
    uint64_t* shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
    NVCC_PRAGMA_UNROLL_AUTO
    for (int g = 0; g < WordPerThread / 2; g++) {
      int ix = g * WARP_SIZE - 4 * (g / 2) + wid - (g % 2) * (wid / 8);
      if (!flagThread || g % 2 == 0) {
        if (misalignment == 0 && (ix + 1) * EltPer16B <= eltN)  // 确保当前写入的16B都是 <= eltN 有效的
          store128((uint64_t*)(dst + ix * EltPer16B), regs[2 * g + 0], regs[2 * g + 1]);
        else
          // 不足 16B 的，需要先写入 sharedMemory
          storeShmem128(shm8 + 2 * ix, regs[2 * g + 0], regs[2 * g + 1]);
      }
    }
    __syncwarp();
    // Write rest from shmem to dst. No need to coalesce stores to 16-bytes,
    // the hardware keeps up fine.
    T* shm = (T*)ncclScratchForWarp(warpInBlock);
    int skip = misalignment == 0 ? eltN & -EltPer16B : 0;
    for (int i = skip + wid; i < eltN; i += WARP_SIZE)
      dst[i] = shm[i];  // 尾部不足 16B，先放 shared memory，再标量写 dst[8], dst[9]
  }

#define WARP_MASK 0xffffffff

  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(uint64_t (&v)[ELEMS_PER_THREAD], int ll128Offset, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    uint64_t vr[ELEMS_PER_THREAD];

    __syncwarp();
    /************************ Wait first recv ********************/
    if (RECV) {  // 先收 recv 0 的数据
      uint64_t* ptr = recvPtr(0) + ll128Offset;
      uint64_t flag = recvFlag(0);
      bool needReload;
      int spins = 0;
      do {
        needReload = false;
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          load128(ptr + u * WARP_SIZE, vr[u], vr[u + 1]);
          needReload |= flagThread && (vr[u + 1] != flag);  //(vr[u+1] != flag) 说明数据还没到，则需要继续 reload
        }
        needReload &= (0 == checkAbort(abort, 1, spins));
      } while (__any_sync(WARP_MASK, needReload));

      NVCC_PRAGMA_UNROLL_AUTO
      // 避免之前 load 到的是旧数据，所以等所有的 flag 到了之后，再重新 load 一次数据
      for (int u = 0; u < ELEMS_PER_THREAD; u += 2) load128(ptr + u * WARP_SIZE, vr[u], vr[u + 1]);
    }

    /************* Finish register load **************/
    if (SRC) {
      // By deferring register shuffle here we've overlapped spinning on first
      // peer's data with memory loads of src data.
      loadRegsFinish(v);
      // 结束后 flagThread 变成: regs[0]有效 regs[1] 空 regs[2]有效 regs[3]空
      // regs[4]有效 regs[5]空 regs[6]有效 regs[7]空
      if (SrcBuf == Input) {
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          v[u] = applyPreOp(redOp, v[u]);
          if (!flagThread) v[u + 1] = applyPreOp(redOp, v[u + 1]);
        }
      }
    }

    /************************ Recv rest *********************/
    if (RECV) {
      {  // Consume data from first recv
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          v[u] = SRC ? applyReduce(redOp, vr[u], v[u]) : vr[u];
          v[u + 1] = SRC ? applyReduce(redOp, vr[u + 1], v[u + 1]) : vr[u + 1];
        }
      }

      // 处理剩余的 peer, 也就是树形/多输入场景下 fan.nrecv() > 1 的情况
      // 不需要把所有的 peer 全部提前 load 进来，因为只要 v 保存结果，vr 反复使用，一边load，一遍 reduce
      //  Yes, for some template arguments this code will be unreachable.  That's fine.
      //  coverity[dead_error_line]
      for (int i = 1; i < MaxRecv && i < fan.nrecv(); i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i) + ll128Offset;
        bool needReload;
        int spins = 0;
        do {
          needReload = false;
          NVCC_PRAGMA_UNROLL_AUTO
          for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
            load128(ptr + u * WARP_SIZE, vr[u], vr[u + 1]);
            needReload |= flagThread && (vr[u + 1] != flag);  //(vr[u+1] != flag) 说明数据还没到，则需要继续 reload
          }
          needReload &= (0 == checkAbort(abort, 1, spins));
        } while (__any_sync(WARP_MASK, needReload));

        NVCC_PRAGMA_UNROLL_AUTO
        // 避免之前 load 到的是旧数据，所以等所有的 flag 到了之后，再重新 load 一次数据
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) load128(ptr + u * WARP_SIZE, vr[u], vr[u + 1]);

        NVCC_PRAGMA_UNROLL_AUTO
        // 把本地的数据 v 和 接收到数据 vr 一起做 reduce 操作
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          v[u] = applyReduce(redOp, vr[u], v[u]);
          v[u + 1] = applyReduce(redOp, vr[u + 1], v[u + 1]);
        }
      }
    }
    /********************** End Recv ************************/

    if (postOp) {
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
        v[u] = applyPostOp(redOp, v[u]);
        v[u + 1] = applyPostOp(redOp, v[u + 1]);
      }
    }

    /************************ Send **************************/
    if (SEND) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      for (int i = 1; i < MaxSend && i < fan.nsend(); i++) {
        uint64_t flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i) + ll128Offset;
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
          store128(ptr + u * WARP_SIZE, v[u], flagThread ? flag : v[u + 1]);
        }
      }
      uint64_t flag = sendFlag(0);
      uint64_t* ptr = sendPtr(0) + ll128Offset;
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < ELEMS_PER_THREAD; u += 2) {
        store128(ptr + u * WARP_SIZE, v[u], flagThread ? flag : v[u + 1]);
      }
    }
    /********************** End Send ************************/
  }

  static constexpr int WireWordPerSlice = WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
  // 256，一个 warp 处理的数据切片总字节数，单位是 uint64_t， 8 byte
  static constexpr int DataEltPerSlice =
    (WireWordPerSlice - WireWordPerSlice / NCCL_LL128_LINEELEMS) * (sizeof(uint64_t) / sizeof(T));
  // 256 / 16 = 多少个 128 B，也就是有多少个 8 B flag，括号前半段是数据的长度，单位 8 byte。
  // DataEltPerSlice 最终结果是一个 warp 处理的实际 T 类型数据的元素个数。一个 warp 处理一个slice

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    // chunk 偏移地址
    T const* srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T* dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    // 当前 thread 处理的起始偏移，单位 uint64_t, 当前 warp 的 slice 起点 + 当前 lane 在 slice 内的位置
    int wireOffset = WireWordPerSlice * warp + 2 * wid;
    const int nwarps = nthreads / WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    // divUp(nelem, DataEltPerSlice) 计算 nelem 个 T 类型的数据需要分成多少个 warp 处理
    // slice(warp) 数量 * sliceSize * 8 (实际字节数，包含 flag )
    if (SEND) waitSend(divUp(nelem, DataEltPerSlice) * WireWordPerSlice * sizeof(uint64_t));
    barrier();
    nelem -= DataEltPerSlice * warp;
    srcPtr += DataEltPerSlice * warp;  // 按照 warp 划分不同的数据起始位置
    dstPtr += DataEltPerSlice * warp;
    while (nelem > 0) {  // 如果减完后 <= 0,则此 warp 不参与工作
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];  // 64B 足够
      if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);  // 每个线程加载自己处理的数据, 结束后,
      // flagThread: regs[0]有效 regs[1]有效 regs[2]未定义/无意义 regs[3]未定义/无意义
      // regs[4]有效 regs[5]有效 regs[6]未定义/无意义 regs[7] 未定义/无意义
      // recvReduceSendCopy() 不知道 eltN，它处理的是固定数量的 regs，并按 LL128 line 写通信 buffer。
      // 也就是说尾部无效元素可能被送出去、参与寄存器层面的搬运/规约，但最终 storeRegs() 只把 i < eltN
      // 的元素写回用户输出。
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SrcBuf, DstBuf>(regs, wireOffset, postOp);
      if (DST) storeRegs(dstPtr, regs, eltInSlice);

      // 如果所有 warp 一轮处理不完，放到第二轮处理
      wireOffset += WireWordPerSlice * nwarps;
      srcPtr += DataEltPerSlice * nwarps;
      dstPtr += DataEltPerSlice * nwarps;
      nelem -= DataEltPerSlice * nwarps;
    }

    barrier();
    if (SEND)
      for (int i = 0; i < MaxSend; i++) sendStep[i] += 1;
    if (SEND) postSend();
    if (RECV)
      for (int i = 0; i < MaxRecv; i++) recvStep[i] += 1;
    if (RECV) postRecv();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;  // 只有 i 对应 wid 线程保存 conn 信息，每个 warp 的 lane i 都会保存同一个 conn
    // recvConn 对非最后 warp 基本无用。但 loadRecvConn 仍然要所有线程执行，因为 recvBuff/recvStep 所有线程都要用。
    // wid==i 的 recvConn 赋值只是顺带的、低成本的统一写法。真正更新 head 的只有最后一个 warp 的 lane i。
    // 额外加 tid >= nthreads-WARP_SIZE 不一定更优。它会多一个谓词条件。
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads - WARP_SIZE && wid < fan.nrecv()) {
      // 让最后一个 warp 里的前 fan.nrecv() 个 lane 负责 recv completion 的 head 更新。
      recvConnHeadPtr = recvConn->head;  // 指向连接的 head，接收方消费完数据后会写它，告诉发送方这个 step 可以复用
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;  // send 侧确实需要 前面的 lane i 和最后 warp 的 lane i 都有 sendConn
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < fan.nsend()) {
      // 前面的线程负责 wait-send，它们会读对端 head，判断 send FIFO
      // 是否还有空位，也就是如果对端还没消费，发送方不能覆盖 FIFO slot
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;  // 本地缓存的对端的 head
      sendConnHead = sendConn->step;         // 本地将要发送/占用的 step
      sendConnFifo = sendConn->connFifo;
    }
    if (tid >= nthreads - WARP_SIZE && wid < fan.nsend()) {
      // 最后一个 warp 的部分 lane 负责 post-send，发送完成后写 tail，通知接收方 (proxy) 数据 ready
      if (sendConn->connFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
    }
  }

public:
  __device__ Primitives(const int tid, const int nthreads, int const* recvPeers, int const* sendPeers,
                        void const* inputBuf, void* outputBuf, uint64_t redOpArg, uint8_t group = 0,
                        uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* e = nullptr,
                        bool ipcReg = false, bool netReg = false, int stepSize_ = 0)
    : redOp(redOpArg), tid(tid), nthreads(nthreads), wid(tid % WARP_SIZE), warp(tid / WARP_SIZE),
        /* tid: 当前 primitive group 内的线程编号; wid: 当前线程在 warp 内的 lane id; warp:
      当前 primitive group 内的第几个 warp, 根据 warp 分配 slice*/
      warpInBlock(threadIdx.x / WARP_SIZE),
        // 当前线程在整个 CUDA block 里的 warp id。这个主要用于 scratch/shared memory
      flagThread((tid % 8) == 7),
        // LL128 的 flag lane。每 8 个 lane 组成一个 128B line，第 8 个 lane 负责检查/写 flag
      group(group),
        // 保存当前 NCCL group/subchannel id。后面 barrier 会用
        // LL128 FIFO 每个 step 的大小，单位是 uint64_t
      stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS / sizeof(uint64_t)) {
    auto* channel = &ncclShmem.channel;
    int nrecv = 0, nsend = 0;
    while (nrecv < MaxRecv && recvPeers[nrecv] >= 0) {
      loadRecvConn(&channel->peers[recvPeers[nrecv]]->recv[connIndexRecv], nrecv);
      nrecv++;
    }
    while (nsend < MaxSend && sendPeers[nsend] >= 0) {
      loadSendConn(&channel->peers[sendPeers[nsend]]->send[connIndexSend], nsend);
      nsend++;
    }
    this->fan = Fan(nrecv, nsend);
    // Coverity reports recvConn and sendConn being possibly NULL at this point but that won't actually
    // happen given the two "while" loops just above.
    // coverity[var_deref_model:FALSE]
    loadRecvSync();
    // coverity[var_deref_model:FALSE]
    loadSendSync();
    setDataPtrs(inputBuf, outputBuf);  // 本地数据在哪里
  }

  __device__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads - WARP_SIZE && wid < fan.nrecv()) recvConn->step = recvConnHead;
    if (tid < fan.nsend()) sendConn->step = sendConnHead;
    // Ensure all steps written back
    barrier();
  }

  __device__ void setDataPtrs(void const* inputBuf, void* outputBuf) {
    userBufs[Input] = (T*)inputBuf;
    userBufs[Output] = (T*)outputBuf;
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    userBufs[Input] += delta;
    userBufs[Output] += delta;
  }

  __device__ void send(intptr_t inpIx, int eltN) {
    return GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
    return GenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp = false) {
    return GenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
    return GenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    return GenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    return GenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp = false) {
    return GenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    return GenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
};
