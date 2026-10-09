//===-- gen/passes/GCCardMarking.h - GC card marking barrier ----*- C++ -*-===//
//
// Post-store barrier for a mostly-concurrent garbage collector (incremental
// update, card marking): after every store of a pointer to memory that is not
// a stack slot or a global, set the byte of the 512 byte card that holds the
// stored-to address. The collector rescans the dirty cards with the program
// stopped, instead of asking the kernel which pages were written.
//
// The runtime provides:
//   extern(C) __gshared ubyte* __sd_gc_card_base;   // biased: base + (addr >> 9)
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
