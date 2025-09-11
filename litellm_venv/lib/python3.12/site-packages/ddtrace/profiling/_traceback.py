import traceback


def format_exception(e):
    return traceback.format_exception_only(type(e), e)[0].rstrip()
