//===-- GCCardMarking.cpp - GC card marking barrier -----------------------===//
//
// See GCCardMarking.h.
//
//===----------------------------------------------------------------------===//

#include "gen/passes/GCCardMarking.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool>
    EmitCardSummary("gc-card-summary", cl::ZeroOrMore,
                    cl::init(false),
                    cl::desc("Also emit 64 KiB summary byte marks in GC card barrier (default false)"));

namespace {

constexpr unsigned CardShift = 9;     // 512 byte cards
constexpr unsigned SummaryShift = 16; // 64 KiB summary

struct Cluster {
  Value *Base = nullptr;
  int64_t MinOffset = 0;
  int64_t MaxOffset = 0;
  Value *MinAddr = nullptr;
  Value *MaxAddr = nullptr;
  StoreInst *LastStore = nullptr;
  SmallVector<StoreInst *, 4> Stores;
};

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
        IntPtrTy(M.getDataLayout().getIntPtrType(M.getContext())),
        SummaryBase(nullptr) {
    auto *PtrTy = PointerType::getUnqual(C);
    CardBase = M.getGlobalVariable("__sd_gc_card_base");
    if (!CardBase) {
      CardBase = new GlobalVariable(M, PtrTy, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage, nullptr,
                                    "__sd_gc_card_base");
    }
    if (EmitCardSummary) {
      SummaryBase = M.getGlobalVariable("__sd_gc_card_summary_base");
      if (!SummaryBase) {
        SummaryBase = new GlobalVariable(M, PtrTy, /*isConstant=*/false,
                                         GlobalValue::ExternalLinkage, nullptr,
                                         "__sd_gc_card_summary_base");
      }
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

  static bool typeContainsPointers(Type *T) {
    if (T->isPtrOrPtrVectorTy()) {
      return true;
    }
    if (auto *ST = dyn_cast<StructType>(T)) {
      for (Type *Elem : ST->elements()) {
        if (typeContainsPointers(Elem))
          return true;
      }
      return false;
    }
    if (auto *AT = dyn_cast<ArrayType>(T)) {
      return typeContainsPointers(AT->getElementType());
    }
    if (auto *VT = dyn_cast<VectorType>(T)) {
      return typeContainsPointers(VT->getElementType());
    }
    return false;
  }

  static bool isPointerFreeObject(const Value *V) {
    const Value *Obj = getUnderlyingObject(V);
    if (auto *AI = dyn_cast<AllocaInst>(Obj)) {
      return !typeContainsPointers(AI->getAllocatedType());
    }
    if (auto *GV = dyn_cast<GlobalVariable>(Obj)) {
      return !typeContainsPointers(GV->getValueType());
    }
    return false;
  }

  static bool isPointerFreeCopy(const MemTransferInst *MT) {
    if (MT->hasMetadata("d.pointer_free")) {
      return true;
    }
    if (isPointerFreeObject(MT->getRawSource()) ||
        isPointerFreeObject(MT->getRawDest())) {
      return true;
    }
    return false;
  }

  void markCard(IRBuilder<> &B, Value *Addr) {
    B.CreateCall(CompilerBarrier);
    markCardNoBarrier(B, Addr);
  }

  /// One byte per 64 KiB, so that finding the dirty cards of a big heap does
  /// not mean reading all of the card table.
  void markSummary(IRBuilder<> &B, Value *AddrInt) {
    if (!EmitCardSummary || !SummaryBase)
      return;
    auto *SIdx = B.CreateLShr(AddrInt, ConstantInt::get(IntPtrTy, SummaryShift));
    auto *SBase = B.CreateLoad(PointerType::getUnqual(C), SummaryBase);
    SBase->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(C, {}));
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
    Base->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(C, {}));
    auto *Slot = B.CreateGEP(Type::getInt8Ty(C), Base, Idx);
    B.CreateStore(ConstantInt::get(Type::getInt8Ty(C), 1), Slot);
    markSummary(B, Int);
  }

  void emitCluster(const Cluster &Clust) {
    assert(Clust.LastStore && "Cluster has no stores");
    IRBuilder<> B(Clust.LastStore->getNextNode());
    B.SetCurrentDebugLocation(Clust.LastStore->getDebugLoc());
    B.CreateCall(CompilerBarrier);
    markCardNoBarrier(B, Clust.MinAddr);
    if (Clust.MinOffset != Clust.MaxOffset) {
      markCardNoBarrier(B, Clust.MaxAddr);
    }
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

    bool FnChanged = false;

    for (BasicBlock &BB : F) {
      SmallVector<Instruction *, 64> Insts;
      for (Instruction &I : BB)
        Insts.push_back(&I);

      SmallVector<Cluster, 8> ActiveClusters;

      auto flushClusters = [&]() {
        if (ActiveClusters.empty())
          return;
        llvm::sort(ActiveClusters, [](const Cluster &A, const Cluster &B) {
          return A.LastStore->comesBefore(B.LastStore);
        });
        for (const Cluster &C : ActiveClusters) {
          K.emitCluster(C);
          FnChanged = true;
        }
        ActiveClusters.clear();
      };

      auto addStore = [&](StoreInst *SI) {
        int64_t Offset = 0;
        Value *Base = GetPointerBaseWithConstantOffset(
            SI->getPointerOperand(), Offset, M.getDataLayout());

        for (auto &C : ActiveClusters) {
          if (C.Base == Base) {
            int64_t NewMin = std::min(C.MinOffset, Offset);
            int64_t NewMax = std::max(C.MaxOffset, Offset);
            if (NewMax - NewMin < (1 << CardShift)) {
              if (Offset < C.MinOffset) {
                C.MinOffset = Offset;
                C.MinAddr = SI->getPointerOperand();
              }
              if (Offset > C.MaxOffset) {
                C.MaxOffset = Offset;
                C.MaxAddr = SI->getPointerOperand();
              }
              C.LastStore = SI;
              C.Stores.push_back(SI);
              return;
            }
            K.emitCluster(C);
            FnChanged = true;
            C.MinOffset = Offset;
            C.MaxOffset = Offset;
            C.MinAddr = SI->getPointerOperand();
            C.MaxAddr = SI->getPointerOperand();
            C.LastStore = SI;
            C.Stores.clear();
            C.Stores.push_back(SI);
            return;
          }
        }

        Cluster NewC;
        NewC.Base = Base;
        NewC.MinOffset = Offset;
        NewC.MaxOffset = Offset;
        NewC.MinAddr = SI->getPointerOperand();
        NewC.MaxAddr = SI->getPointerOperand();
        NewC.LastStore = SI;
        NewC.Stores.push_back(SI);
        ActiveClusters.push_back(NewC);
      };

      for (Instruction *I : Insts) {
        if (I->isDebugOrPseudoInst())
          continue;

        if (auto *MT = dyn_cast<MemTransferInst>(I)) {
          flushClusters();
          if (!Marker::isUntracked(MT->getRawDest()) &&
              !Marker::isPointerFreeCopy(MT)) {
            IRBuilder<> B(MT->getNextNode());
            B.SetCurrentDebugLocation(MT->getDebugLoc());
            K.markRange(B, MT->getRawDest(), MT->getLength());
            FnChanged = true;
          }
        } else if (auto *CB = dyn_cast<CallBase>(I)) {
          if (auto *II = dyn_cast<IntrinsicInst>(CB)) {
            if (II->isLifetimeStartOrEnd() ||
                II->getIntrinsicID() == Intrinsic::assume)
              continue;
          }
          flushClusters();
        } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
          flushClusters();
          if (Marker::storesPointer(RMW->getValOperand()) &&
              !Marker::isUntracked(RMW->getPointerOperand())) {
            IRBuilder<> B(RMW->getNextNode());
            B.SetCurrentDebugLocation(RMW->getDebugLoc());
            K.markCard(B, RMW->getPointerOperand());
            FnChanged = true;
          }
        } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
          flushClusters();
          if (Marker::storesPointer(CX->getNewValOperand()) &&
              !Marker::isUntracked(CX->getPointerOperand())) {
            IRBuilder<> B(CX->getNextNode());
            B.SetCurrentDebugLocation(CX->getDebugLoc());
            K.markCard(B, CX->getPointerOperand());
            FnChanged = true;
          }
        } else if (isa<FenceInst>(I)) {
          flushClusters();
        } else if (auto *SI = dyn_cast<StoreInst>(I)) {
          if (SI->isVolatile() || SI->isAtomic()) {
            flushClusters();
            if (Marker::storesPointer(SI->getValueOperand()) &&
                !Marker::isUntracked(SI->getPointerOperand())) {
              IRBuilder<> B(SI->getNextNode());
              B.SetCurrentDebugLocation(SI->getDebugLoc());
              K.markCard(B, SI->getPointerOperand());
              FnChanged = true;
            }
          } else if (Marker::storesPointer(SI->getValueOperand()) &&
                     !Marker::isUntracked(SI->getPointerOperand())) {
            addStore(SI);
          }
        } else if (I->isTerminator()) {
          flushClusters();
        }
      }

      flushClusters();
    }

    if (FnChanged) {
      F.setMemoryEffects(MemoryEffects::unknown());
      Changed = true;
    }
  }

  return Changed;
}
