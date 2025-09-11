from ddtrace.internal.assembly import Assembly
from ddtrace.internal.compat import PYTHON_VERSION_INFO as PY


# -----------------------------------------------------------------------------
# Generator Wrapping
# -----------------------------------------------------------------------------
# DEV: This is roughly equivalent to
#
# __ddgen = wrapper(wrapped, args, kwargs)
# __ddgensend = __ddgen.send
# try:
#     value = next(__ddgen)
#     while True:
#         try:
#             tosend = yield value
#         except GeneratorExit:
#             return __ddgen.close()
#         except Exception:
#             value = __ddgen.throw(*sys.exc_info())
#         else:
#             value = __ddgensend(tosend)
# except StopIteration:
#     return
# -----------------------------------------------------------------------------
GENERATOR_ASSEMBLY = Assembly()
GENERATOR_HEAD_ASSEMBLY = None

if PY >= (3, 12):
    GENERATOR_HEAD_ASSEMBLY = Assembly()
    GENERATOR_HEAD_ASSEMBLY.parse(
        r"""
            return_generator
            pop_top
        """
    )

    GENERATOR_ASSEMBLY.parse(
        r"""
        try                             @stopiter
            copy                        1
            store_fast                  $__ddgen
            load_attr                   $send
            store_fast                  $__ddgensend
            push_null
            load_const                  next
            load_fast                   $__ddgen

        loop:
            call                        1
        tried

        yield:
        try                             @genexit lasti
            yield_value                 3
            resume                      1
            push_null
            swap                        2
            load_fast                   $__ddgensend
            swap                        2
            jump_backward               @loop
        tried

        genexit:
        try                             @stopiter
            push_exc_info
            load_const                  GeneratorExit
            check_exc_match
            pop_jump_if_false           @exc
            pop_top
            load_fast                   $__ddgen
            load_method                 $close
            call                        0
            swap                        2
            pop_except
            return_value

        exc:
            pop_top
            push_null
            load_fast                   $__ddgen
            load_attr                   $throw
            push_null
            load_const                  sys.exc_info
            call                        0
            call_function_ex            0
            swap                        2
            pop_except
            jump_backward               @yield
        tried

        stopiter:
            push_exc_info
            load_const                  StopIteration
            check_exc_match
            pop_jump_if_false           @propagate
            pop_top
            pop_except
            return_const                None

        propagate:
            reraise                     0
        """
    )

elif PY >= (3, 11):
    GENERATOR_HEAD_ASSEMBLY = Assembly()
    GENERATOR_HEAD_ASSEMBLY.parse(
        r"""
            return_generator
            pop_top
        """
    )

    GENERATOR_ASSEMBLY.parse(
        r"""
        try                             @stopiter
            copy                        1
            store_fast                  $__ddgen
            load_attr                   $send
            store_fast                  $__ddgensend
            push_null
            load_const                  next
            load_fast                   $__ddgen

        loop:
            precall                     1
            call                        1
        tried

        yield:
        try                             @genexit lasti
            yield_value
            resume                      1
            push_null
            swap                        2
            load_fast                   $__ddgensend
            swap                        2
            jump_backward               @loop
        tried

        genexit:
        try                             @stopiter
            push_exc_info
            load_const                  GeneratorExit
            check_exc_match
            pop_jump_forward_if_false   @exc
            pop_top
            load_fast                   $__ddgen
            load_method                 $close
            precall                     0
            call                        0
            swap                        2
            pop_except
            return_value

        exc:
            pop_top
            push_null
            load_fast                   $__ddgen
            load_attr                   $throw
            push_null
            load_const                  sys.exc_info
            precall                     0
            call                        0
            call_function_ex            0
            swap                        2
            pop_except
            jump_backward               @yield
        tried

        stopiter:
            push_exc_info
            load_const                  StopIteration
            check_exc_match
            pop_jump_forward_if_false   @propagate
            pop_top
            pop_except
            load_const                  None
            return_value

        propagate:
            reraise                     0
        """
    )

elif PY >= (3, 10):
    GENERATOR_ASSEMBLY.parse(
        r"""
        setup_finally                   @stopiter
            dup_top
            store_fast                  $__ddgen
            load_attr                   $send
            store_fast                  $__ddgensend
            load_const                  next
            load_fast                   $__ddgen

        loop:
            call_function               1

        yield:
        setup_finally                   @genexit
            yield_value
        pop_block
            load_fast                   $__ddgensend
            rot_two
            jump_absolute               @loop

        genexit:
            dup_top
            load_const                  GeneratorExit
            jump_if_not_exc_match       @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $close
            call_function               0
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $throw
            load_const                  sys.exc_info
            call_function               0
            call_function_ex            0
            rot_four
        pop_except
            jump_absolute               @yield

        stopiter:
            dup_top
            load_const                  StopIteration
            jump_if_not_exc_match       @propagate
            pop_top
            pop_top
            pop_top
        pop_except
            load_const                  None
            return_value

        propagate:
            reraise                     0
        """
    )

elif PY >= (3, 9):
    GENERATOR_ASSEMBLY.parse(
        r"""
        setup_finally                   @stopiter
            dup_top
            store_fast                  $__ddgen
            load_attr                   $send
            store_fast                  $__ddgensend
            load_const                  next
            load_fast                   $__ddgen

        loop:
            call_function               1

        yield:
        setup_finally                   @genexit
            yield_value
        pop_block
            load_fast                   $__ddgensend
            rot_two
            jump_absolute               @loop

        genexit:
            dup_top
            load_const                  GeneratorExit
            jump_if_not_exc_match       @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $close
            call_function               0
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $throw
            load_const                  sys.exc_info
            call_function               0
            call_function_ex            0
            rot_four
        pop_except
            jump_absolute               @yield

        stopiter:
            dup_top
            load_const                  StopIteration
            jump_if_not_exc_match       @propagate
            pop_top
            pop_top
            pop_top
        pop_except
            load_const                  None
            return_value

        propagate:
            reraise
        """
    )

elif PY >= (3, 8):
    GENERATOR_ASSEMBLY.parse(
        r"""
        setup_finally                   @stopiter
            dup_top
            store_fast                  $__ddgen
            load_attr                   $send
            store_fast                  $__ddgensend
            load_const                  next
            load_fast                   $__ddgen

        loop:
            call_function               1

        yield:
        setup_finally                   @genexit
            yield_value
        pop_block
            load_fast                   $__ddgensend
            rot_two
            jump_absolute               @loop

        genexit:
            dup_top
            load_const                  GeneratorExit
            compare_op                  asm.Compare.EXC_MATCH
            pop_jump_if_false           @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $close
            call_function               0
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $throw
            load_const                  sys.exc_info
            call_function               0
            call_function_ex            0
            rot_four
        pop_except
            jump_absolute               @yield

        stopiter:
            dup_top
            load_const                  StopIteration
            compare_op                  asm.Compare.EXC_MATCH
            pop_jump_if_false           @propagate
            pop_top
            pop_top
            pop_top
        pop_except
            load_const                  None
            return_value

        propagate:
        end_finally
            load_const                  None
            return_value
        """
    )


elif PY >= (3, 7):
    GENERATOR_ASSEMBLY.parse(
        r"""
        setup_except                    @stopiter
            dup_top
            store_fast                  $__ddgen
            load_attr                   $send
            store_fast                  $__ddgensend
            load_const                  next
            load_fast                   $__ddgen

        loop:
            call_function               1

        yield:
        setup_except                    @genexit
            yield_value
        pop_block
            load_fast                   $__ddgensend
            rot_two
            jump_absolute               @loop

        genexit:
            dup_top
            load_const                  GeneratorExit
            compare_op                  asm.Compare.EXC_MATCH
            pop_jump_if_false           @exc
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $close
            call_function               0
            return_value

        exc:
            pop_top
            pop_top
            pop_top
            pop_top
            load_fast                   $__ddgen
            load_attr                   $throw
            load_const                  sys.exc_info
            call_function               0
            call_function_ex            0
            store_fast                  $__value
        pop_except
            load_fast                   $__value
            jump_absolute               @yield

        stopiter:
            dup_top
            load_const                  StopIteration
            compare_op                  asm.Compare.EXC_MATCH
            pop_jump_if_false           @propagate
            pop_top
            pop_top
            pop_top
        pop_except
            load_const                  None
            return_value

        propagate:
        end_finally
            load_const                  None
            return_value
        """
    )

else:
    msg = "No generator wrapping support for Python %d.%d" % PY[:2]
    raise RuntimeError(msg)


def wrap_generator(instrs, code, lineno):
    if GENERATOR_HEAD_ASSEMBLY is not None:
        instrs[0:0] = GENERATOR_HEAD_ASSEMBLY.bind(lineno=lineno)

    instrs[-1:] = GENERATOR_ASSEMBLY.bind(lineno=lineno)
