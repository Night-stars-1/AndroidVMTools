package com.v7878.vmtools;

import static com.v7878.foreign.MemoryLayout.PathElement.groupElement;
import static com.v7878.llvm.Core.LLVMAddFunction;
import static com.v7878.llvm.Core.LLVMAppendBasicBlock;
import static com.v7878.llvm.Core.LLVMBuildAdd;
import static com.v7878.llvm.Core.LLVMBuildAlloca;
import static com.v7878.llvm.Core.LLVMBuildAnd;
import static com.v7878.llvm.Core.LLVMBuildIntToPtr;
import static com.v7878.llvm.Core.LLVMBuildLoad;
import static com.v7878.llvm.Core.LLVMBuildNot;
import static com.v7878.llvm.Core.LLVMBuildOr;
import static com.v7878.llvm.Core.LLVMBuildRetVoid;
import static com.v7878.llvm.Core.LLVMBuildStore;
import static com.v7878.llvm.Core.LLVMGetParams;
import static com.v7878.llvm.Core.LLVMPositionBuilderAtEnd;
import static com.v7878.unsafe.ArtMethodUtils.ARTMETHOD_LAYOUT;
import static com.v7878.unsafe.ArtMethodUtils.getExecutableFlags;
import static com.v7878.unsafe.ArtModifiers.kAccCompileDontBother;
import static com.v7878.unsafe.ArtModifiers.kAccFastInterpreterToInterpreterInvoke;
import static com.v7878.unsafe.ArtModifiers.kAccIntrinsic;
import static com.v7878.unsafe.ArtModifiers.kAccPreCompiled;
import static com.v7878.unsafe.JNIUtils.getRuntimePtr;
import static com.v7878.unsafe.Reflection.getArtMethod;
import static com.v7878.unsafe.Utils.reachabilityFence;
import static com.v7878.unsafe.foreign.BulkLinker.CallType.NATIVE_STATIC_OMIT_ENV;
import static com.v7878.unsafe.foreign.BulkLinker.MapType.INT;
import static com.v7878.unsafe.foreign.BulkLinker.MapType.LONG_AS_WORD;
import static com.v7878.unsafe.foreign.BulkLinker.MapType.VOID;
import static com.v7878.unsafe.foreign.LibArt.ART;
import static com.v7878.unsafe.llvm.LLVMBuilder.call;
import static com.v7878.unsafe.llvm.LLVMBuilder.const_fnptr;
import static com.v7878.unsafe.llvm.LLVMBuilder.const_int32;
import static com.v7878.unsafe.llvm.LLVMBuilder.const_intptr;
import static com.v7878.unsafe.llvm.LLVMBuilder.no_frame_pointer_elim;
import static com.v7878.unsafe.llvm.LLVMTypes.fn_t;
import static com.v7878.unsafe.llvm.LLVMTypes.int32_t;
import static com.v7878.unsafe.llvm.LLVMTypes.intptr_t;
import static com.v7878.unsafe.llvm.LLVMTypes.ptr_t;
import static com.v7878.unsafe.llvm.LLVMTypes.void_t;

import com.v7878.foreign.Arena;
import com.v7878.foreign.MemorySegment;
import com.v7878.llvm.Types.LLVMTypeRef;
import com.v7878.r8.annotations.DoNotOptimize;
import com.v7878.r8.annotations.DoNotShrink;
import com.v7878.r8.annotations.DoNotShrinkType;
import com.v7878.unsafe.foreign.BulkLinker;
import com.v7878.unsafe.foreign.BulkLinker.CallSignature;
import com.v7878.unsafe.foreign.BulkLinker.LibrarySymbol;
import com.v7878.unsafe.llvm.LLVMUtils;

import java.lang.reflect.Executable;
import java.util.Objects;
import java.util.Optional;

/**
 * Applies raw {@code ArtMethod} changes inside a single ART "suspend all" window.
 *
 * <p>The whole window lives inside one ordinary JNI call. {@code NATIVE_STATIC_OMIT_ENV} produces
 * a plain {@code ACC_NATIVE} method without {@code @CriticalNative} or {@code @FastNative}, so ART
 * moves the calling thread from {@code kRunnable} to {@code kNative} on entry and back to
 * {@code kRunnable} on exit. That state transition is the precondition of
 * {@code ThreadList::SuspendAll}/{@code ResumeAll} and it is what restores the shared ownership of
 * the mutator lock on the way out. Calling ART's suspend-all pair from a {@code kRunnable} thread
 * (as the previous {@code @CriticalNative} based implementation did) leaves the thread
 * {@code kRunnable} while the shared mutator lock is no longer held, which later aborts in an
 * implicit suspend check.
 *
 * <p>Everything between the constructor and the destructor of {@code ScopedSuspendAll} is
 * generated native code: no Java code runs, nothing is allocated and no class is loaded while the
 * exclusive mutator lock is held. All addresses, masks, offsets and code blobs are prepared before
 * the call, and every target is validated (intrinsic rejection) before the suspension starts.
 */
final class ArtTransaction {
    private ArtTransaction() {
    }

    // Mask pairs preserved from the previous in-Java implementation:
    //   Hooks.deoptimize(): ArtMethodUtils.makeExecutableNonCompilable(ex)
    //   Hooks.hook():       ArtMethodUtils.makeExecutableNonCompilable(target) and
    //                       changeExecutableFlags(target, kAccFastInterpreterToInterpreterInvoke, 0)
    // ArtMethodUtils.makeExecutableNonCompilable() rejects intrinsics, removes kAccPreCompiled and
    // adds kAccCompileDontBother.
    private static final int HOOK_REMOVE_FLAGS =
            kAccPreCompiled | kAccFastInterpreterToInterpreterInvoke;
    private static final int HOOK_ADD_FLAGS = kAccCompileDontBother;

    private static final int DEOPTIMIZE_REMOVE_FLAGS = kAccPreCompiled;
    private static final int DEOPTIMIZE_ADD_FLAGS = kAccCompileDontBother;

    // Keys of the generated native code. They are resolved by the lookup below and never reach a
    // real dynamic library, so they must not collide with a libart symbol.
    private static final String PATCH_ONE_SYMBOL = "vmtools_suspend_patch_one";
    private static final String PATCH_TWO_SYMBOL = "vmtools_suspend_patch_two";
    private static final String DEOPTIMIZE_SYMBOL = "vmtools_suspend_deoptimize_boot_image";

    private static final String SUSPEND_ALL_CTOR = "_ZN3art16ScopedSuspendAllC2EPKcb";
    private static final String SUSPEND_ALL_DTOR = "_ZN3art16ScopedSuspendAllD2Ev";
    private static final String DEOPTIMIZE_BOOT_IMAGE = "_ZN3art7Runtime19DeoptimizeBootImageEv";

    static {
        // Generate the native code (LLVM codegen, dex generation and symbol resolution) while this
        // class is first used - never while a suspend-all window is open.
        reachabilityFence(Native.INSTANCE);
    }

    /**
     * Applies one patch: the access flags of {@code target} are updated with the hook masks and its
     * entry point is redirected to {@code entry_point}, all under a single suspension.
     */
    static void hookPatch(Executable target, long entry_point) {
        long target_method = checkedArtMethod(target);
        Native.INSTANCE.patchOne(Native.CAUSE.nativeAddress(), HOOK_REMOVE_FLAGS, HOOK_ADD_FLAGS,
                target_method, entry_point);
        reachabilityFence(target);
    }

    /**
     * Applies two patches atomically, so a method can never be live with its counterpart still
     * unpatched. Both entry points and both trampolines must be prepared by the caller before this
     * call.
     */
    static void hookPatches(Executable first, long first_entry_point,
                            Executable second, long second_entry_point) {
        long first_method = checkedArtMethod(first);
        long second_method = checkedArtMethod(second);
        Native.INSTANCE.patchTwo(Native.CAUSE.nativeAddress(), HOOK_REMOVE_FLAGS, HOOK_ADD_FLAGS,
                first_method, first_entry_point, second_method, second_entry_point);
        reachabilityFence(first);
        reachabilityFence(second);
    }

    /**
     * Applies one patch with the deoptimization masks: the executable becomes non compilable and
     * its entry point is redirected to {@code entry_point}, under a single suspension.
     */
    static void deoptimizePatch(Executable target, long entry_point) {
        long target_method = checkedArtMethod(target);
        Native.INSTANCE.patchOne(Native.CAUSE.nativeAddress(),
                DEOPTIMIZE_REMOVE_FLAGS, DEOPTIMIZE_ADD_FLAGS, target_method, entry_point);
        reachabilityFence(target);
    }

    /**
     * Runs {@code Runtime::DeoptimizeBootImage} inside a suspend-all window. The runtime performs no
     * Java callbacks, so nothing may run between the suspension and the resumption.
     */
    static void deoptimizeBootImage() {
        Native.INSTANCE.suspendDeoptimizeBootImage(
                Native.CAUSE.nativeAddress(), getRuntimePtr());
    }

    private static long checkedArtMethod(Executable ex) {
        Objects.requireNonNull(ex);
        long art_method = getArtMethod(ex);
        // ArtMethodUtils.makeExecutableNonCompilable() rejects intrinsics. The check must happen
        // before the suspension, while Java is still allowed to throw and while nothing has been
        // mutated yet.
        if ((getExecutableFlags(art_method) & kAccIntrinsic) != 0) {
            throw new IllegalArgumentException("Intrinsic executables are not supported: " + ex);
        }
        return art_method;
    }

    private static long flagsOffset() {
        return ARTMETHOD_LAYOUT.byteOffset(groupElement("access_flags_"));
    }

    private static long entryPointOffset() {
        return ARTMETHOD_LAYOUT.byteOffset(groupElement("ptr_sized_fields_"),
                groupElement("entry_point_from_quick_compiled_code_"));
    }

    private static Optional<MemorySegment> lookup(String name) {
        return switch (name) {
            case PATCH_ONE_SYMBOL -> Optional.of(generatePatches(Native.SCOPE, 1));
            case PATCH_TWO_SYMBOL -> Optional.of(generatePatches(Native.SCOPE, 2));
            case DEOPTIMIZE_SYMBOL -> Optional.of(generateDeoptimizeBootImage(Native.SCOPE));
            default -> Optional.empty();
        };
    }

    /**
     * {@code void suspend_patch_n(void* cause, int32_t remove_flags, int32_t add_flags,
     *                             void* target0, void* entry0[, void* target1, void* entry1])}
     *
     * <p>The generated code calls the {@code ScopedSuspendAll} constructor, rewrites the access
     * flags and the entry point of every slot exactly once and then calls the destructor. The slot
     * loop runs while the module is built, so the emitted code is straight line.
     */
    private static MemorySegment generatePatches(Arena arena, int slots) {
        long ctor = ART.findOrThrow(SUSPEND_ALL_CTOR).nativeAddress();
        long dtor = ART.findOrThrow(SUSPEND_ALL_DTOR).nativeAddress();
        long flags_offset = flagsOffset();
        long entry_offset = entryPointOffset();
        return LLVMUtils.generateFunctionCodeSegment((context, module, builder) -> {
            var intptr = intptr_t(context);
            var int32 = int32_t(context);
            var void_type = void_t(context);
            // "this" of the C++ constructor and destructor is a pointer, not an integer.
            var scope_type = ptr_t(intptr);

            var arg_types = new LLVMTypeRef[3 + slots * 2];
            arg_types[0] = intptr;  // cause
            arg_types[1] = int32;   // remove flags
            arg_types[2] = int32;   // add flags
            for (int slot = 0; slot < slots; slot++) {
                arg_types[3 + slot * 2] = intptr;  // ArtMethod
                arg_types[4 + slot * 2] = intptr;  // entry point
            }

            var function = LLVMAddFunction(module,
                    slots == 1 ? "suspend_patch_one" : "suspend_patch_two",
                    fn_t(void_type, arg_types));
            no_frame_pointer_elim(function);
            LLVMPositionBuilderAtEnd(builder, LLVMAppendBasicBlock(function, ""));

            var args = LLVMGetParams(function);

            // Storage for the ScopedSuspendAll object. Modern ART keeps no state in it, the word is
            // only there so that the constructor and the destructor never receive a null "this".
            var scope = LLVMBuildAlloca(builder, intptr, "");
            LLVMBuildStore(builder, const_intptr(context, 0), scope);

            call(builder, const_fnptr(builder, fn_t(void_type, scope_type, intptr, int32), ctor),
                    scope, args[0], const_int32(context, 0));

            for (int slot = 0; slot < slots; slot++) {
                var target = args[3 + slot * 2];
                var entry_point = args[4 + slot * 2];

                var flags_address = LLVMBuildAdd(builder, target,
                        const_intptr(context, flags_offset), "");
                var flags_pointer = LLVMBuildIntToPtr(builder, flags_address, ptr_t(int32), "");
                var old_flags = LLVMBuildLoad(builder, flags_pointer, "");
                var new_flags = LLVMBuildOr(builder, LLVMBuildAnd(builder, old_flags,
                        LLVMBuildNot(builder, args[1], ""), ""), args[2], "");
                LLVMBuildStore(builder, new_flags, flags_pointer);

                var entry_address = LLVMBuildAdd(builder, target,
                        const_intptr(context, entry_offset), "");
                LLVMBuildStore(builder, entry_point,
                        LLVMBuildIntToPtr(builder, entry_address, ptr_t(intptr), ""));
            }

            call(builder, const_fnptr(builder, fn_t(void_type, scope_type), dtor), scope);
            LLVMBuildRetVoid(builder);
            return function;
        }, arena);
    }

    /**
     * {@code void suspend_deoptimize_boot_image(void* cause, void* runtime)}
     *
     * <p>The generated code calls the {@code ScopedSuspendAll} constructor,
     * {@code Runtime::DeoptimizeBootImage} and then the destructor.
     */
    private static MemorySegment generateDeoptimizeBootImage(Arena arena) {
        long ctor = ART.findOrThrow(SUSPEND_ALL_CTOR).nativeAddress();
        long dtor = ART.findOrThrow(SUSPEND_ALL_DTOR).nativeAddress();
        long deoptimize = ART.findOrThrow(DEOPTIMIZE_BOOT_IMAGE).nativeAddress();
        return LLVMUtils.generateFunctionCodeSegment((context, module, builder) -> {
            var intptr = intptr_t(context);
            var int32 = int32_t(context);
            var void_type = void_t(context);
            // "this" of the C++ constructor, destructor and Runtime::DeoptimizeBootImage is a
            // pointer, not an integer.
            var scope_type = ptr_t(intptr);

            var function = LLVMAddFunction(module, "suspend_deoptimize_boot_image",
                    fn_t(void_type, intptr, intptr));
            no_frame_pointer_elim(function);
            LLVMPositionBuilderAtEnd(builder, LLVMAppendBasicBlock(function, ""));

            var args = LLVMGetParams(function);

            var scope = LLVMBuildAlloca(builder, intptr, "");
            LLVMBuildStore(builder, const_intptr(context, 0), scope);

            call(builder, const_fnptr(builder, fn_t(void_type, scope_type, intptr, int32), ctor),
                    scope, args[0], const_int32(context, 0));
            call(builder, const_fnptr(builder, fn_t(void_type, scope_type), deoptimize),
                    LLVMBuildIntToPtr(builder, args[1], scope_type, ""));
            call(builder, const_fnptr(builder, fn_t(void_type, scope_type), dtor), scope);

            LLVMBuildRetVoid(builder);
            return function;
        }, arena);
    }

    @DoNotShrinkType
    @DoNotOptimize
    private abstract static class Native {
        @DoNotShrink
        private static final Arena SCOPE = Arena.ofAuto();

        @DoNotShrink
        static final MemorySegment CAUSE = SCOPE.allocateFrom("Hook");

        /**
         * Ordinary JNI entry point of the single patch transaction. The generated native stub drops
         * the {@code JNIEnv*} and the class, so the generated code receives only the declared
         * arguments.
         */
        @DoNotShrink
        @LibrarySymbol(name = PATCH_ONE_SYMBOL)
        @CallSignature(type = NATIVE_STATIC_OMIT_ENV, ret = VOID,
                args = {LONG_AS_WORD, INT, INT, LONG_AS_WORD, LONG_AS_WORD})
        abstract void patchOne(long cause, int remove_flags, int add_flags,
                               long target, long entry_point);

        /**
         * Ordinary JNI entry point of the two patch transaction; both patches are applied inside one
         * suspend-all window.
         */
        @DoNotShrink
        @LibrarySymbol(name = PATCH_TWO_SYMBOL)
        @CallSignature(type = NATIVE_STATIC_OMIT_ENV, ret = VOID,
                args = {LONG_AS_WORD, INT, INT, LONG_AS_WORD, LONG_AS_WORD,
                        LONG_AS_WORD, LONG_AS_WORD})
        abstract void patchTwo(long cause, int remove_flags, int add_flags,
                               long target0, long entry0, long target1, long entry1);

        /**
         * Ordinary JNI entry point of the {@code DeoptimizeBootImage} transaction.
         */
        @DoNotShrink
        @LibrarySymbol(name = DEOPTIMIZE_SYMBOL)
        @CallSignature(type = NATIVE_STATIC_OMIT_ENV, ret = VOID,
                args = {LONG_AS_WORD, LONG_AS_WORD})
        abstract void suspendDeoptimizeBootImage(long cause, long runtime);

        static final Native INSTANCE = BulkLinker.generateImpl(SCOPE, Native.class,
                ArtTransaction::lookup);
    }
}
