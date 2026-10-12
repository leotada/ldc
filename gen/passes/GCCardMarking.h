//===-- gen/passes/GCCardMarking.h - GC card marking barrier ----*- C++ -*-===//
//
// Post-store barrier for a mostly-concurrent garbage collector (incremental
// update, card marking): after every store of a pointer to memory that is not
// a stack slot or a global, set the byte of the 512 byte card that holds the
// stored-to address. The collector rescans the dirty cards with the program
// stopped, instead of asking the kernel which pages were written.
//
// The collector's memory is one contiguous range, [lo, lo + n * 512). The
// barrier is range checked, so an address outside it (malloc, a pointer with a
// tag in its top byte, a stack slot that escaped) costs a compare and is not
// marked, and nothing needs to be mapped for the rest of the address space:
//
//   if ((addr - lo) >> 9 < n)  base[(addr - lo) >> 9] = 1;
//
// The runtime provides (all zero until the collector starts, which makes the
// barrier a no-op until then):
//   extern(C) __gshared size_t __sd_gc_card_lo;     // start of the range
//   extern(C) __gshared size_t __sd_gc_card_n;      // number of cards in it
//   extern(C) __gshared ubyte* __sd_gc_card_base;   // one byte per card
//   extern(C) void __sd_gc_card_range(void* dst, size_t size);
//
//===----------------------------------------------------------------------===//

#pragma once
#include "gen/llvm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

struct LLVM_LIBRARY_VISIBILITY GCCardMarking {
  bool run(llvm::Module &M);
};

struct LLVM_LIBRARY_VISIBILITY GCCardMarkingPass
    : public llvm::PassInfoMixin<GCCardMarkingPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    return pass.run(M) ? llvm::PreservedAnalyses::none()
                       : llvm::PreservedAnalyses::all();
  }

  static llvm::StringRef name() { return "GCCardMarking"; }

private:
  GCCardMarking pass;
};
