import signal


def handle_signal(sig, f):
    """
    Returns a signal of type `sig` with function `f`, if there are
    no previously defined signals.

    Else, wraps the given signal with the previously defined one,
    so no signals are overridden.
    """
    old_signal = signal.getsignal(sig)

    def wrap_signals(*args, **kwargs):
        if old_signal is not None:
            old_signal(*args, **kwargs)
        f(*args, **kwargs)

    # Return the incoming signal if any of the following cases happens:
    # - old signal does not exist,
    # - old signal is the same as the incoming, or
    # - old signal is our wrapper.
    # This avoids multiple signal calling and infinite wrapping.
    if not callable(old_signal) or old_signal == f or old_signal == wrap_signals:
        return signal.signal(sig, f)

    return signal.signal(sig, wrap_signals)
