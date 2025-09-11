import math

INF = float("inf")
MINUS_INF = float("-inf")
NaN = float("NaN")


def floatToGoString(d):
    d = float(d)
    if d == INF:
        return "+Inf"
    elif d == MINUS_INF:
        return "-Inf"
    elif math.isnan(d):
        return "NaN"
    else:
        s = repr(d)
        dot = s.find(".")
        # Go switches to exponents sooner than Python.
        # We only need to care about positive values for le/quantile.
        if d > 0 and dot > 6:
            mantissa = f"{s[0]}.{s[1:dot]}{s[dot + 1:]}".rstrip("0.")
            return f"{mantissa}e+0{dot - 1}"
        return s
