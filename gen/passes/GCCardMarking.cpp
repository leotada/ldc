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
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

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

/// A barrier to emit. The analysis only records these; the code is generated
/// afterwards, because the range check splits basic blocks.
struct Request {
  Instruction *Before; // the barrier goes right before this instruction
  DebugLoc Loc;
  Value *First;  // the lowest address written
  Value *Second; // a second card of the same object, or null
  Value *Size;   // set for a range: a call to __sd_gc_card_range
};

struct Marker {
  Module &M;
  LLVMContext &C;
  Type *IntPtrTy;
  GlobalVariable *CardBase;
  GlobalVariable *CardLo;
  GlobalVariable *CardCount;
  GlobalVariable *SummaryBase;
  FunctionCallee CardRange;
  InlineAsm *CompilerBarrier;
  MDNode *InRangeWeights;
  SmallVector<Request, 32> Requests;

  explicit Marker(Module &M)
      : M(M), C(M.getContext()),
        IntPtrTy(M.getDataLayout().getIntPtrType(M.getContext())),
        SummaryBase(nullptr) {
    auto *PtrTy = PointerType::getUnqual(C);
    CardBase = global("__sd_gc_card_base", PtrTy);
    // The memory the collector tracks is one contiguous range: [lo, lo + n *
    // 512). An address outside it (malloc, a tagged pointer, a stack slot that
    // escaped) is not tracked, and the barrier skips it. Both are zero until
    // the collector starts, which makes every barrier a no-op until then.
    CardLo = global("__sd_gc_card_lo", IntPtrTy);
    CardCount = global("__sd_gc_card_n", IntPtrTy);
    if (EmitCardSummary)
      SummaryBase = global("__sd_gc_card_summary_base", PtrTy);
    CardRange = M.getOrInsertFunction(
        "__sd_gc_card_range", Type::getVoidTy(C), PtrTy, IntPtrTy);
    // Keeps the compiler from moving the card store above the pointer store.
    CompilerBarrier = InlineAsm::get(FunctionType::get(Type::getVoidTy(C), false),
                                     "", "~{memory}", /*hasSideEffects=*/true);
    // Stores into the collector's memory are the common case.
    InRangeWeights = MDBuilder(C).createLikelyBranchWeights();
  }

  GlobalVariable *global(StringRef Name, Type *Ty) {
    if (auto *G = M.getGlobalVariable(Name))
      return G;
    return new GlobalVariable(M, Ty, /*isConstant=*/false,
                              GlobalValue::ExternalLinkage, nullptr, Name);
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

  //===--------------------------------------------------------------------===//
  // Analysis: record what needs a barrier, and where.
  //===--------------------------------------------------------------------===//

  /// One card, after a store of a pointer that is not a plain store.
  void markCard(Instruction *After, Value *Addr) {
    Requests.push_back({After->getNextNode(), After->getDebugLoc(), Addr,
                        nullptr, nullptr});
  }

  void markRange(MemTransferInst *MT) {
    Value *Dst = MT->getRawDest();
    Value *Size = MT->getLength();
    Instruction *Before = MT->getNextNode();

    if (auto *CI = dyn_cast<ConstantInt>(Size)) {
      uint64_t N = CI->getZExtValue();
      if (N == 0) {
        return;
      }
      // At most two cards for up to 512 bytes: first and last byte.
      if (N <= (1u << CardShift)) {
        Value *Last = nullptr;
        if (N > 1) {
          IRBuilder<> B(Before);
          Last = B.CreateGEP(Type::getInt8Ty(C), Dst,
                             ConstantInt::get(IntPtrTy, N - 1));
        }
        Requests.push_back({Before, MT->getDebugLoc(), Dst, Last, nullptr});
        return;
      }
    }
    Requests.push_back({Before, MT->getDebugLoc(), Dst, nullptr, Size});
  }

  void emitCluster(const Cluster &Clust) {
    assert(Clust.LastStore && "Cluster has no stores");
    Value *Second =
        Clust.MinOffset != Clust.MaxOffset ? Clust.MaxAddr : nullptr;
    Requests.push_back({Clust.LastStore->getNextNode(),
                        Clust.LastStore->getDebugLoc(), Clust.MinAddr, Second,
                        nullptr});
  }

  //===--------------------------------------------------------------------===//
  // Code generation.
  //===--------------------------------------------------------------------===//

  Value *invariantLoad(IRBuilder<> &B, Type *Ty, GlobalVariable *G) {
    auto *L = B.CreateLoad(Ty, G);
    L->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(C, {}));
    return L;
  }

  /// The byte that marks a card (or a summary). The collector must not find it
  /// set before the pointer that was stored is visible, or it could scan the
  /// memory, clear the card, and miss the pointer: on a CPU that reorders
  /// stores (ARM) that takes a release store. On x86 a release store is a plain
  /// `mov`, so there the ordinary store is kept and LLVM stays free to treat it
  /// as one (the compiler barrier above already orders it against the compiler).
  void markByte(IRBuilder<> &B, Value *Ptr) {
    auto *SI = B.CreateStore(ConstantInt::get(Type::getInt8Ty(C), 1), Ptr);
    if (!Triple(M.getTargetTriple()).isX86()) {
      SI->setAlignment(Align(1));
      SI->setAtomic(AtomicOrdering::Release);
    }
  }

  void storeCard(IRBuilder<> &B, Value *Addr, Value *Lo) {
    auto *Rel = B.CreateSub(B.CreatePtrToInt(Addr, IntPtrTy), Lo);
    auto *Idx = B.CreateLShr(Rel, ConstantInt::get(IntPtrTy, CardShift));
    auto *Base = invariantLoad(B, PointerType::getUnqual(C), CardBase);
    markByte(B, B.CreateGEP(Type::getInt8Ty(C), Base, Idx));

    // One byte per 64 KiB, so that finding the dirty cards of a big heap does
    // not mean reading all of the card table.
    if (EmitCardSummary && SummaryBase) {
      auto *SIdx = B.CreateLShr(Rel, ConstantInt::get(IntPtrTy, SummaryShift));
      auto *SBase = invariantLoad(B, PointerType::getUnqual(C), SummaryBase);
      markByte(B, B.CreateGEP(Type::getInt8Ty(C), SBase, SIdx));
    }
  }

  /// if ((First - lo) >> 9 < n) { cards[...] = 1; cards[...] = 1; }
  ///
  /// Only the first address is checked: the second is in the same object, so
  /// it is inside the tracked range whenever the first is.
  void lower(const Request &R) {
    IRBuilder<> B(R.Before);
    B.SetCurrentDebugLocation(R.Loc);
    B.CreateCall(CompilerBarrier);

    auto *Lo = invariantLoad(B, IntPtrTy, CardLo);
    auto *Idx = B.CreateLShr(
        B.CreateSub(B.CreatePtrToInt(R.First, IntPtrTy), Lo),
        ConstantInt::get(IntPtrTy, CardShift));
    auto *N = invariantLoad(B, IntPtrTy, CardCount);
    auto *In = B.CreateICmpULT(Idx, N);

    Instruction *Then = SplitBlockAndInsertIfThen(In, R.Before->getIterator(),
                                                  /*Unreachable=*/false,
                                                  InRangeWeights);
    B.SetInsertPoint(Then);
    B.SetCurrentDebugLocation(R.Loc);

    if (R.Size) {
      B.CreateCall(CardRange, {R.First, B.CreateZExtOrTrunc(R.Size, IntPtrTy)});
      return;
    }

    storeCard(B, R.First, Lo);
    if (R.Second)
      storeCard(B, R.Second, Lo);
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

    K.Requests.clear();

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
            K.markRange(MT);
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
            K.markCard(RMW, RMW->getPointerOperand());
          }
        } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
          flushClusters();
          if (Marker::storesPointer(CX->getNewValOperand()) &&
              !Marker::isUntracked(CX->getPointerOperand())) {
            K.markCard(CX, CX->getPointerOperand());
          }
        } else if (isa<FenceInst>(I)) {
          flushClusters();
        } else if (auto *SI = dyn_cast<StoreInst>(I)) {
          if (SI->isVolatile() || SI->isAtomic()) {
            flushClusters();
            if (Marker::storesPointer(SI->getValueOperand()) &&
                !Marker::isUntracked(SI->getPointerOperand())) {
              K.markCard(SI, SI->getPointerOperand());
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

    if (K.Requests.empty()) {
      continue;
    }

    // The analysis is done: nothing is read from the function's blocks after
    // this point, so splitting them is safe. Each request keeps its own
    // instruction pointer, which stays valid whichever block it ends up in.
    for (const Request &R : K.Requests) {
      K.lower(R);
    }

    F.setMemoryEffects(MemoryEffects::unknown());
    Changed = true;
  }

  return Changed;
}
