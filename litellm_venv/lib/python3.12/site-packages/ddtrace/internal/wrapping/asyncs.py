import bytecode as bc

from ddtrace.internal.assembly import Assembly
from ddtrace.internal.compat import PYTHON_VERSION_INFO as PY


# -----------------------------------------------------------------------------
# Coroutine and Async Generator Wrapping
# -----------------------------------------------------------------------------
# DEV: The wrapping of async generators is roughly equivalent to
#
# __ddgen = wrapper(wrapped, args, kwargs)
# __ddgensend = __ddgen.asend
# try:
#     value = await __ddgen.__anext__()
#     while True:
#         try:
#             tosend = yield value
#         except GeneratorExit:
#             await __ddgen.aclose()
#         except Exception:
#             value = await __ddgen.athrow(*sys.exc_info())
#         else:
#             value = await __ddgensend(tosend)
# except StopAsyncIteration:
#     return
# -----------------------------------------------------------------------------

COROUTINE_ASSEMBLY = Assembly()
ASYNC_GEN_ASSEMBLY = Assembly()
ASYNC_HEAD_ASSEMBLY = None

if PY >= (3, 12):
    ASYNC_HEAD_ASSEMBLY = Assembly()
    ASYNC_HEAD_ASSEMBLY.parse(
        r"""
            return_generator
            pop_top
        """
    )

    COROUTINE_ASSEMBLY.parse(
        r"""
            get_awaitable                   0
            load_const                      None

        presend:
            send                            @send
            yield_value                     2
            resume                          3
            jump_backward_no_interrupt      @presend
        send:
            end_send
        """
    )

    ASYNC_GEN_ASSEMBLY.parse(
        r"""
        try                                 @stopiter
            copy                            1
            store_fast                      $__ddgen
            load_attr                       (False, 'asend')
            store_fast                      $__ddgensend
            load_fast                       $__ddgen
            load_attr                       (True, '__anext__')
            call                            0

        loop:
            get_awaitable                   0
            load_const                      None
        presend0:
            send                            @send0
        tried

        try                                 @genexit lasti
            yield_value                     3
            resume                          3
            jump_backward_no_interrupt      @loop
        send0:
            end_send

        yield:
            call_intrinsic_1                asm.Intrinsic1Op.INTRINSIC_ASYNC_GEN_WRAP
            yield_value                     3
            resume                          1
            push_null
            swap                            2
            load_fast                       $__ddgensend
            swap                            2
            call                            1
            jump_backward                   @loop
        tried

        genexit:
        try                                 @stopiter
            push_exc_info
            load_const                      GeneratorExit
            check_exc_match
            pop_jump_if_false               @exc
            pop_top
            load_fast                       $__ddgen
            load_attr                       (True, 'aclose')
            call                            0
            get_awaitable                   0
            load_const                      None

        presend1:
            send                            @send1
            yield_value                     4
            resume                          3
            jump_backward_no_interrupt      @presend1
        send1:
            end_send
            pop_top
            pop_except
            return_const                    None

        exc:
            pop_top
            push_null
            load_fast                       $__ddgen
            load_attr                       (False, 'athrow')
            push_null
            load_const                      sys.exc_info
            call                            0
            call_function_ex                0
            get_awaitable                   0
            load_const                      None

        presend2:
            send                            @send2
            yield_value                     4
            resume                          3
            jump_backward_no_interrupt      @presend2
        send2:
            end_send
            swap                            2
            pop_except
            jump_backward                   @yield
        tried

        stopiter:
            push_exc_info
            load_const                      StopAsyncIteration
            check_exc_match
            pop_jump_if_false               @propagate
            pop_top
            pop_except
            return_const                    None

        propagate:
            reraise                         0
        """
    )


elif PY >= (3, 11):
    ASYNC_HEAD_ASSEMBLY = Assembly()
    ASYNC_HEAD_ASSEMBLY.parse(
        r"""
            return_generator
            pop_top
        """
    )

    COROUTINE_ASSEMBLY.parse(
        r"""
            get_awaitable                   0
            load_const                      None

        presend:
            send                            @send
            yield_value
            resume                          3
            jump_backward_no_interrupt      @presend
        send:
        """
    )

    ASYNC_GEN_ASSEMBLY.parse(
        r"""
        try                                 @stopiter
            copy                            1
            store_fast                      $__ddgen
            load_attr                       $asend
            store_fast                      $__ddgensend
            load_fast                       $__ddgen
            load_method                     $__anext__
            precall                         0
            call                            0

        loop:
            get_awaitable                   0
            load_const                      None
        presend0:
            send                            @send0
        tried

        try                                 @genexit lasti
            yield_value
            resume                          3
            jump_backward_no_interrupt      @loop
        send0:

        yield:
            async_gen_wrap
            yield_value
            resume                          1
            push_null
            swap                            2
            load_fast                       $__ddgensend
            swap                            2
            precall                         1
            call                            1
            jump_backward                   @loop
        tried

        genexit:
        try                                 @stopiter
            push_exc_info
            load_const                      GeneratorExit
            check_exc_match
            pop_jump_forward_if_false       @exc
            pop_top
            load_fast                       $__ddgen
            load_method                     $aclose
            precall                         0
            call                            0
            get_awaitable                   0
            load_const                      None

        presend1:
            send                            @send1
            yield_value
            resume                          3
            jump_backward_no_interrupt      @presend1
        send1:
            pop_top
            pop_except
            load_const                      None
            return_value

        exc:
            pop_top
            push_null
            load_fast                       $__ddgen
            load_attr                       $athrow
            push_null
            load_const                      sys.exc_info
            precall                         0
            call                            0
            call_function_ex                0
            get_awaitable                   0
            load_const                      None

        presend2:
            send                            @send2
            yield_value
            resume                          3
            jump_backward_no_interrupt      @presend2
        send2:
            swap                            2
            pop_except
            jump_backward                   @yield
        tried

        stopiter:
            push_exc_info
            load_const                      StopAsyncIteration
            check_exc_match
            pop_jump_forward_if_false       @propagate
            pop_top
            pop_except
            load_const                      None
            return_value

        propagate:
            reraise                         0
        """
    )


elif PY >= (3, 10):
    COROUTINE_ASSEMBLY.parse(
        r"""
            get_awaitable
            load_const                      None
            yield_from
        """
    )

    ASYNC_GEN_ASSEMBLY.parse(
        r"""
        setup_finally                       @stopiter
            dup_top
            store_fast                      $__ddgen
            load_attr                       $asend
            store_fast                      $__ddgensend
            load_fast                       $__ddgen
            load_attr                       $__anext__
            call_function                   0

        loop:
            get_awaitable
            load_const                      None
            yield_from

        yield:
        setup_finally                       @genexit
            yield_value
        pop_block
            load_fast                       $__ddgensend
            rot_two
            call_function                   1
            jump_absolute                   @loop

        genexit:
            dup_top
            load_const                      GeneratorExit
            jump_if_not_exc_match           @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $aclose
            call_function                   0
            get_awaitable
            load_const                      None
            yield_from
        pop_except
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $athrow
            load_const                      sys.exc_info
            call_function                   0
            call_function_ex                0
            get_awaitable
            load_const                      None
            yield_from
            rot_four
        pop_except
            jump_absolute                   @yield

        stopiter:
            dup_top
            load_const                      StopAsyncIteration
            jump_if_not_exc_match           @propagate
            pop_top
            pop_top
            pop_top
            pop_except
            load_const                      None
            return_value

        propagate:
            reraise                         0
        """
    )


elif PY >= (3, 9):
    COROUTINE_ASSEMBLY.parse(
        r"""
            get_awaitable
            load_const                      None
            yield_from
        """
    )

    ASYNC_GEN_ASSEMBLY.parse(
        r"""
        setup_finally                       @stopiter
            dup_top
            store_fast                      $__ddgen
            load_attr                       $asend
            store_fast                      $__ddgensend
            load_fast                       $__ddgen
            load_attr                       $__anext__
            call_function                   0

        loop:
            get_awaitable
            load_const                      None
            yield_from

        yield:
        setup_finally                       @genexit
            yield_value
        pop_block
            load_fast                       $__ddgensend
            rot_two
            call_function                   1
            jump_absolute                   @loop

        genexit:
            dup_top
            load_const                      GeneratorExit
            jump_if_not_exc_match           @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $aclose
            call_function                   0
            get_awaitable
            load_const                      None
            yield_from
        pop_except
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $athrow
            load_const                      sys.exc_info
            call_function                   0
            call_function_ex                0
            get_awaitable
            load_const                      None
            yield_from
            rot_four
        pop_except
            jump_absolute                   @yield

        stopiter:
            dup_top
            load_const                      StopAsyncIteration
            jump_if_not_exc_match           @propagate
            pop_top
            pop_top
            pop_top
            pop_except
            load_const                      None
            return_value

        propagate:
            reraise
        """
    )

elif PY >= (3, 8):
    COROUTINE_ASSEMBLY.parse(
        r"""
            get_awaitable
            load_const                      None
            yield_from
        """
    )

    ASYNC_GEN_ASSEMBLY.parse(
        r"""
        setup_finally                       @stopiter
            dup_top
            store_fast                      $__ddgen
            load_attr                       $asend
            store_fast                      $__ddgensend
            load_fast                       $__ddgen
            load_attr                       $__anext__
            call_function                   0

        loop:
            get_awaitable
            load_const                      None
            yield_from

        yield:
        setup_finally                       @genexit
            yield_value
        pop_block
            load_fast                       $__ddgensend
            rot_two
            call_function                   1
            jump_absolute                   @loop

        genexit:
            dup_top
            load_const                      GeneratorExit
            compare_op                      asm.Compare.EXC_MATCH
            pop_jump_if_false               @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $aclose
            call_function                   0
            get_awaitable
            load_const                      None
            yield_from
        pop_except
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $athrow
            load_const                      sys.exc_info
            call_function                   0
            call_function_ex                0
            get_awaitable
            load_const                      None
            yield_from
            rot_four
        pop_except
            jump_absolute                   @yield

        stopiter:
            dup_top
            load_const                      StopAsyncIteration
            compare_op                      asm.Compare.EXC_MATCH
            pop_jump_if_false               @propagate
            pop_top
            pop_top
            pop_top
            pop_except
            load_const                      None
            return_value

        propagate:
        end_finally
            load_const                      None
            return_value
        """
    )

elif PY >= (3, 7):
    COROUTINE_ASSEMBLY.parse(
        r"""
            get_awaitable
            load_const                      None
            yield_from
        """
    )

    ASYNC_GEN_ASSEMBLY.parse(
        r"""
        setup_except                        @stopiter
            dup_top
            store_fast                      $__ddgen
            load_attr                       $asend
            store_fast                      $__ddgensend
            load_fast                       $__ddgen
            load_attr                       $__anext__
            call_function                   0

        loop:
            get_awaitable
            load_const                      None
            yield_from

        yield:
        setup_except                        @genexit
            yield_value
        pop_block
            load_fast                       $__ddgensend
            rot_two
            call_function                   1
            jump_absolute                   @loop

        genexit:
            dup_top
            load_const                      GeneratorExit
            compare_op                      asm.Compare.EXC_MATCH
            pop_jump_if_false               @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $aclose
            call_function                   0
            get_awaitable
            load_const                      None
            yield_from
        pop_except
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                       $__ddgen
            load_attr                       $athrow
            load_const                      sys.exc_info
            call_function                   0
            call_function_ex                0
            get_awaitable
            load_const                      None
            yield_from
            store_fast                      $__value
        pop_except
            load_fast                       $__value
            jump_absolute                   @yield

        stopiter:
            dup_top
            load_const                      StopAsyncIteration
            compare_op                      asm.Compare.EXC_MATCH
            pop_jump_if_false               @propagate
            pop_top
            pop_top
            pop_top
            pop_except
            load_const                      None
            return_value

        propagate:
        end_finally
            load_const                      None
            return_value
        """
    )


else:
    msg = "No async wrapping support for Python %d.%d" % PY[:2]
    raise RuntimeError(msg)


def wrap_async(instrs, code, lineno):
    if (bc.CompilerFlags.ASYNC_GENERATOR | bc.CompilerFlags.COROUTINE) & code.co_flags:
        if ASYNC_HEAD_ASSEMBLY is not None:
            instrs[0:0] = ASYNC_HEAD_ASSEMBLY.bind()

        if bc.CompilerFlags.COROUTINE & code.co_flags:
            # DEV: This is just
            # >>> return await wrapper(wrapped, args, kwargs)
            instrs[-1:-1] = COROUTINE_ASSEMBLY.bind()

        elif bc.CompilerFlags.ASYNC_GENERATOR & code.co_flags:
            instrs[-1:] = ASYNC_GEN_ASSEMBLY.bind()
