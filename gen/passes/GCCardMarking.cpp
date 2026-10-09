//===-- GCCardMarking.cpp - GC card marking barrier -----------------------===//
//
// See GCCardMarking.h.
//
//===----------------------------------------------------------------------===//

#include "gen/passes/GCCardMarking.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;

namespace {

constexpr unsigned CardShift = 9;     // 512 byte cards
constexpr unsigned SummaryShift = 16; // 64 KiB summary

struct Marker {
  Module &M;
  LLVMContext &C;
  Type *IntPtrTy;
  GlobalVariable *CardBase;
  GlobalVariable *SummaryBase;
  FunctionCallee CardRange;
  InlineAsm *CompilerBarrier;

  explicit Marker(Module &M)
      : M(M), C(M.getContext()),
        IntPtrTy(M.getDataLayout().getIntPtrType(M.getContext())) {
    auto *PtrTy = PointerType::getUnqual(C);
    CardBase = M.getGlobalVariable("__sd_gc_card_base");
    if (!CardBase) {
      CardBase = new GlobalVariable(M, PtrTy, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage, nullptr,
                                    "__sd_gc_card_base");
    }
    SummaryBase = M.getGlobalVariable("__sd_gc_card_summary_base");
    if (!SummaryBase) {
      SummaryBase = new GlobalVariable(M, PtrTy, /*isConstant=*/false,
                                       GlobalValue::ExternalLinkage, nullptr,
                                       "__sd_gc_card_summary_base");
    }
    CardRange = M.getOrInsertFunction(
        "__sd_gc_card_range", Type::getVoidTy(C), PtrTy, IntPtrTy);
    // Keeps the compiler from moving the card store above the pointer store.
    CompilerBarrier = InlineAsm::get(FunctionType::get(Type::getVoidTy(C), false),
                                     "", "~{memory}", /*hasSideEffects=*/true);
  }

  /// Memory that the collector does not need cards for: stack slots (scanned
  /// with the stack) and globals (roots, rescanned with the stacks).
  static bool isUntracked(const Value *Dst) {
    const Value *Obj = getUnderlyingObject(Dst);
    return isa<AllocaInst>(Obj) || isa<GlobalVariable>(Obj) ||
           (isa<Argument>(Obj) && cast<Argument>(Obj)->hasByValAttr());
  }

  static bool storesPointer(const Value *V) {
    Type *T = V->getType();
    return T->isPtrOrPtrVectorTy();
  }

  void markCard(IRBuilder<> &B, Value *Addr) {
    B.CreateCall(CompilerBarrier);
    auto *Int = B.CreatePtrToInt(Addr, IntPtrTy);
    auto *Idx = B.CreateLShr(Int, ConstantInt::get(IntPtrTy, CardShift));
    auto *Base = B.CreateLoad(PointerType::getUnqual(C), CardBase);
    auto *Slot = B.CreateGEP(Type::getInt8Ty(C), Base, Idx);
    B.CreateStore(ConstantInt::get(Type::getInt8Ty(C), 1), Slot);
    markSummary(B, Int);
  }

  /// One byte per 64 KiB, so that finding the dirty cards of a big heap does
  /// not mean reading all of the card table.
  void markSummary(IRBuilder<> &B, Value *AddrInt) {
    auto *SIdx = B.CreateLShr(AddrInt, ConstantInt::get(IntPtrTy, SummaryShift));
    auto *SBase = B.CreateLoad(PointerType::getUnqual(C), SummaryBase);
    auto *SSlot = B.CreateGEP(Type::getInt8Ty(C), SBase, SIdx);
    B.CreateStore(ConstantInt::get(Type::getInt8Ty(C), 1), SSlot);
  }

  void markRange(IRBuilder<> &B, Value *Dst, Value *Size) {
    B.CreateCall(CompilerBarrier);
    if (auto *CI = dyn_cast<ConstantInt>(Size)) {
      uint64_t N = CI->getZExtValue();
      if (N == 0) {
        return;
      }
      // At most two cards for up to 512 bytes: first and last byte.
      if (N <= (1u << CardShift)) {
        markCardNoBarrier(B, Dst);
        if (N > 1) {
          auto *Last = B.CreateGEP(Type::getInt8Ty(C), Dst,
                                   ConstantInt::get(IntPtrTy, N - 1));
          markCardNoBarrier(B, Last);
        }
        return;
      }
    }
    B.CreateCall(CardRange, {Dst, B.CreateZExtOrTrunc(Size, IntPtrTy)});
  }

  void markCardNoBarrier(IRBuilder<> &B, Value *Addr) {
    auto *Int = B.CreatePtrToInt(Addr, IntPtrTy);
    auto *Idx = B.CreateLShr(Int, ConstantInt::get(IntPtrTy, CardShift));
    auto *Base = B.CreateLoad(PointerType::getUnqual(C), CardBase);
    auto *Slot = B.CreateGEP(Type::getInt8Ty(C), Base, Idx);
    B.CreateStore(ConstantInt::get(Type::getInt8Ty(C), 1), Slot);
    markSummary(B, Int);
  }
};

} // namespace

bool GCCardMarking::run(Module &M) {
  Marker K(M);
  bool Changed = false;

  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    // The barrier code itself, and the collector, are never instrumented.
    if (F.getName().starts_with("__sd_gc_")) {
      continue;
    }

    SmallVector<Instruction *, 32> Work;
    for (Instruction &I : instructions(F)) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (Marker::storesPointer(SI->getValueOperand()) &&
            !Marker::isUntracked(SI->getPointerOperand())) {
          Work.push_back(&I);
        }
      } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        if (Marker::storesPointer(RMW->getValOperand()) &&
            !Marker::isUntracked(RMW->getPointerOperand())) {
          Work.push_back(&I);
        }
      } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
        if (Marker::storesPointer(CX->getNewValOperand()) &&
            !Marker::isUntracked(CX->getPointerOperand())) {
          Work.push_back(&I);
        }
      } else if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
        if (!Marker::isUntracked(MT->getRawDest())) {
          Work.push_back(&I);
        }
      }
    }

    if (!Work.empty()) {
      // The barrier reads and writes memory the function did not declare.
      F.setMemoryEffects(MemoryEffects::unknown());
    }

    for (Instruction *I : Work) {
      IRBuilder<> B(I->getNextNode());
      B.SetCurrentDebugLocation(I->getDebugLoc());
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        K.markCard(B, SI->getPointerOperand());
      } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
        K.markCard(B, RMW->getPointerOperand());
      } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
        K.markCard(B, CX->getPointerOperand());
      } else if (auto *MT = dyn_cast<MemTransferInst>(I)) {
        K.markRange(B, MT->getRawDest(), MT->getLength());
      }
      Changed = true;
    }
  }

  return Changed;
}
