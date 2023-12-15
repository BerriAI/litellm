def register_sentry(sentry_dsn, **opts):
    """Given a Raven client and an RQ worker, registers exception handlers
    with the worker so exceptions are logged to Sentry.
    """
    import sentry_sdk
    from sentry_sdk.integrations.rq import RqIntegration

    sentry_sdk.init(sentry_dsn, integrations=[RqIntegration()], **opts)
