from typing import Final

from integration.translation.case import TranslationTestCase
from pydantic import JsonValue

"""Provider request and reply shape from https://docs.parallel.ai/api-reference/decisions/make-web-grounded-typed-decisions. Mock reply captured live on 2026-10-31.
"""
_CITATIONS: Final[list[JsonValue]] = [
    {
        "url": "https://access.redhat.com/solutions/7108129",
        "title": 'NFS client hangs and "nfs4_handle_delegation_recall_error" found in dmesg. - Red Hat Customer Portal',
        "excerpts": [
            'NFS client hangs and "nfs4_handle_delegation_recall_error" found in dmesg.\n\nSolution Verified - Updated 2026-06-18T01:07:40+00:00 -\nEnglish\n* No translations currently exist.\n\nIssue\n\n* NFS client hangs while returning delegations.\n* Below delegation recall errors may be reported in dmesg:\nNFS: nfs4_handle_delegation_recall_error: unhandled error -512\nNFS: nfs4_handle_delegation_recall_error: unhandled error -10024\n* We have a hanging process, it\'s using data ontop of a mounted NFS export. The fcntl() system call issued by a user process has not returned, The user process processing does not finish - usually it finishes within 2 minutes.\n\nEnvironment\n\n* Red Hat Enterprise Linux (RHEL) 8, all minor releases\n* kernel:\n* rhel8.4: kernels before 4.18.0-305.194.1.el8_4 affected\n* rhel8.10: kernels before 4.18.0-553.37.1.el8_10 affected\n* NFSv4'
        ],
    },
    {
        "url": "https://github.com/moby/buildkit/issues/2804",
        "title": "GitHub cache exporter (`type=gha`) hangs reletively low number of cache reads/writes, satirating api limits/triggering throttling. · Issue #2804 · moby/buildkit · GitHub",
        "excerpts": [
            'Copy link\nOpen\nOpen\nGitHub cache exporter ( type=gha ) hangs reletively low number of cache reads/writes, satirating api limits/triggering throttling. #2804\nCopy link\n\nDescription\n\n@bendavies\nbendavies\nopened on Apr 13, 2022\nIssue body actions\n\nBehaviour\n\nCache operations using type=gha will start hanging after a relavatively low number of cache reads/writes.\nWe are a small team (5 devs) and are hitting these limits constantly on our main repository. Not exactly high throughput.\n\n...\n\nCache reads and writes go towards saturating the limit, and both are affected by this throttling.\n@dhadka mentions there is a limit of about "150 cache requests in 5 minutes", and seems about right: #2276 (comment)\n\nExpected behaviour\n\nI don\'t know, but the current behaviour of hanging is not great. there is no feedback as to what is hanging resulting in the user just being confused.'
        ],
    },
    {
        "url": "https://github.com/huginn/huginn/issues/2963",
        "title": "WebsiteAgent hangs on a specific url, never finishes job or times out · Issue #2963 · huginn/huginn · GitHub",
        "excerpts": [
            'WebsiteAgent hangs on a specific url, never finishes job or times out\n\n* Page: GitHub issue\n* URL: https://github.com/huginn/huginn/issues/2963\n* State: open\n* Author: BoKKeR\n* Created: 2021-03-14T16:42:01Z\n* Updated: 2021-03-16T19:09:24Z\n* Repository: huginn/huginn\n* Number: #2963\n* Comments: 1\nI know it might be controversial but I am trying to get a new GPU for a while. Wanted to setup a monitor for when the stock on amd.com changes. But the site just won\'t load. The WebsiteAgent gets stuck in\nFetching https://www.amd.com/de/direct-buy/se\nThe only way to stop it is by stopping the container. As I am running the service in a container I cant find any of the debug log locations. Can anyone else try if they are encountering the same error?\n{\n  "debug": "true",\n  "expected_update_period_in_days": "2",\n  "url": "https://www.amd.com/de/direct-buy/se",\n  "type": "html",'
        ],
    },
    {
        "url": "https://github.com/mozilla-firefox/firefox-crb/blob/main/docs/bug-mgmt/guides/severity.md",
        "title": "firefox-crb/docs/bug-mgmt/guides/severity.md at main · mozilla-firefox/firefox-crb · GitHub",
        "excerpts": [
            "We use the severity field in Bugzilla to indicate the scope of a bug's effect on Firefox.\nThe field is display alongside the bug's priority.\n\nValues\n\nSeverity levels and their definitions are enumerated at https://wiki.mozilla.org/BMO/UserGuide/BugFields#bug_severity.\nBy default, new bugs have a severity of --.\n\nExamples of S1 bugs\n\n* WebExtensions disabled for all users\n* Web search not working from URL bar\n* Crashes with data loss\n\nExamples of S2 bugs\n\nBugs that could reasonably be expected to cause a Firefox user to switch browsers, either because the severity is bad enough, or the frequency of occurrence is high enough.\n* Bug 1640441 - Slack hangs indefinitely in a onResize loop"
        ],
    },
    {
        "url": "https://github.com/n0-computer/iroh/issues/4604",
        "title": "Endpoint::close hangs forever if the tokio runtime gets dropped while the connection is open · Issue #4604 · n0-computer/iroh · GitHub",
        "excerpts": [
            "@alpergundogdu\nalpergundogdu\nopened on Oct 14, 2026\nIssue body actions\nDescribe the bug\nAfter opening a connection from inside a short-lived tokio runtime, and dropping the runtime while the connection is still open, Endpoint::close() never returns.\nIn my repro it hangs reliably; when I keep the runtime open, it closes instantly.\nLocal repro/test (has the control tests too): https://github.com/decdn/iroh-tests/blob/main/tests/bug_endpoint_close_after_runtime_drop.rs\n\n...\n\nIf spawning on the caller's runtime is intended, a note on Endpoint::connect saying the driver lives there would have saved us a lot of time. (Maybe it's an expectation/documentation issue?)\nIroh\nVersion: iroh 1.3.0, noq 1.3.0\nEndpoint configuration: bound to 127.0.0.1, no relays.\nPlatform(s)\nDesktop:\n\n...\n\nSorry @flub , that was unclear: the endpoint's runtime stays alive, and close() is called from it. Only a separate short-lived runtime that runs connect() gets dropped."
        ],
    },
    {
        "url": "https://bugzilla.mozilla.org/show_bug.cgi?id=1640441",
        "title": "1640441 - Slack hangs indefinitely in a onResize loop",
        "excerpts": [
            "Closed Bug 1640441 Opened 6 years ago Closed 6 years ago\n\nSlack hangs indefinitely in a on Resize loop\n\nCategories\n\n(Web Compatibility :: Site Reports, defect, P1)\n\nProduct:\nWeb Compatibility ▾\nComponent:\nSite Reports ▾\nType:\ndefect\nPriority:\nP1 Severity:\nS2\n\nTracking\n\n(Not tracked)\n\nStatus:\nRESOLVED FIXED\n\nPeople\n\n(Reporter: Harald, Assigned: emilio)\n\nReferences\n\n(Depends on 1 open bug)\n\nDetails\n\n(Keywords: enterprise, perf, webcompat:site-wait)\n\nAttachments\n\n(1 file, 1 obsolete file)\n\n|relevant unminified code.\n6 years ago Emilio Cobos Álvarez [:emilio]\n* 13.50 KB,\n\n...\n\n* Hover over the link\n* ... tab immediately becomes unresponsive ...\nThis may well be a bug in slack, but IMO it's still on us to try and fix what's going on. When it happens, it affects not only Slack but my entire Firefox session. This is a serious usability issue -- I'm now at the point of restarting Firefox several times a day because of this bug, if"
        ],
    },
    {
        "url": "https://www.ibm.com/support/pages/cognos-business-intelligence-1022-fix-lists",
        "title": "Cognos Business Intelligence 10.2.2 Fix Lists",
        "excerpts": [
            'IBM Support \n\nCognos Business Intelligence 10.2.2 Fix Lists\n\nRelease Notes\n\nAbstract\n\nA comprehensive list of defect corrections for major releases, refresh packs and fix packs of Cognos Business Intelligence 10.2.2.\nDetails of the APARs listed below can be accessed by clicking the link for the APAR number. If you have questions about a particular defect, please contact Customer Support.\n\n...\n\n* PI05143 | MULTIPLE VALUES IN CONTEXT FILTERS CAUSE A CROSSTAB REPORT TO SHOW DASH "-" INSTEAD OF CORRECT VALUES\n* PI05166 | QE-DEF-0359 The query contains a reference to at least one object that does not exist\n* PI05233 | Export deployments from 8.4.1 with export folder or package tasks in Content Administration contain unknown content steps\n\n...\n\n* PI08523 | Progressive Chart - Incorrect Colours when "Reverse Catagory Order" = True\n* PI08585 | Cognos hangs when starting against a Content Store from previous version (upgrade) and using SSL gateway UR'
        ],
    },
    {
        "url": "https://josm.eu/ticket/8175",
        "title": "\n      #8175 ([PATCH] Cadastre grab query sometimes hangs completely)\n          – JOSM\n\n    ",
        "excerpts": [
            'Opened 14 years ago\nClosed 14 years ago\nLast modified 14 years ago\n\n#8175 closed defect ( fixed )\n\n[PATCH] Cadastre grab query sometimes hangs completely\n\n* Reported by: | Don-vip | Owned by: | pieren\n* Priority: | normal | Milestone: | \n* Component: | Plugin cadastre-fr | Version: | \n* Keywords: |  | Cc: | \n\nDescription\n\nDuring cadastre grab, the HTTP connection with the WMS server sometimes completely hangs (the progress dialog is frozen and cancel button does not work).\nThe only solution is to restart JOSM to have a clean state and grab the cadastre again. We had a similar problem with history request ( #7640 ) when dealing with the API and solved it in [r5324](https://josm.eu/changeset/5324/josm "fix #7640 - History query will not cancel or timeout\n\n...\n\nby Don-vip , 14 years ago\n\n* Attachment: | 8175.patch \u200b added\n\ncomment:1 by Don-vip , 14 years ago'
        ],
    },
    {
        "url": "https://bugzilla.mozilla.org/show_bug.cgi?id=1749908",
        "title": "1749908 - Infinite loop in HTTP3 hangs socket thread - Bugzilla",
        "excerpts": [
            "Closed Bug 1749908 Opened 4 years ago Closed 4 years ago\n\nInfinite loop in HTTP3 hangs socket thread\n\nCategories\n\n(Core :: Networking, defect)\n\nProduct:\nCore ▾\nComponent:\nNetworking ▾\nPlatform:\nDesktop\nAll\nType:\ndefect\nPriority:\nNot set Severity:\n--\n\nTracking\n\n( )\n\nStatus:\nVERIFIED DUPLICATE of bug 1749910\n\nPeople\n\n(Reporter: heftig, Unassigned)\n\nDetails\n\nBottom ↓\nTags ▾\nTimeline ▾\n| |Jan Alexander Steffens [:heftig]\n* Reporter | \n* ### Description\n•\n* 4 years ago\n* \nWhile browsing, pages suddenly stopped loading.\n\n...\n\nIt's not related to a specific version, we're getting reports ESR is even affected. Suspicion is around some long existing HTTP3 bug that's being triggered by an external service updating.\n\n...\n\nI agree it is not the root cause but it certainly caused a massive problem for many users of Firefox, the majority of whom are opted-in by default. If they had not been, they would not have been affected."
        ],
    },
    {
        "url": "https://bugzilla.mozilla.org/show_bug.cgi?id=1748414",
        "title": "1748414 - Firefox 94 sometimes hangs on TLSv1.3 requests",
        "excerpts": [
            "Closed Bug 1748414 Opened 4 years ago Closed 4 years ago\n\nFirefox 94 sometimes hangs on TLSv1 .3 requests\n\nCategories\n\n(Core :: Networking: HTTP, defect)\n\nProduct:\nCore ▾\nComponent:\nNetworking: HTTP ▾\nVersion:\nFirefox 94\nType:\ndefect\nPriority:\nNot set Severity:\n--\n\nTracking\n\n( )\n\nStatus:\nRESOLVED DUPLICATE of bug 1740856\n\nPeople\n\n(Reporter: code, Unassigned)\n\nDetails\n\nBottom ↓\n\n...\n\nSteps to reproduce:\nIn the beginning of November, colleagues started to experience hangs of 1 to 1.5 minutes when working with our own web app. We use latest official releases for Windows, Ubuntu and FreeBSD. The problems exist for all plattforms and very likely started with Firefox 94.\nActual results:\nSometimes while interacting normally with our web app, a request hangs for a long time (up to 1.5 minutes; but usually succeeds). This typically happens on POST requests."
        ],
    },
]
DECISION_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "parallel_ai/decision",
        "input": "Ticket (billing): The export job hangs at 99% and never finishes",
        "questions": [
            {"type": "predicate", "name": "defect", "instructions": "Is this a defect?"},
            {
                "type": "choice",
                "name": "severity",
                "instructions": "How severe is it?",
                "choices": [
                    {"value": "low", "description": "cosmetic"},
                    {"value": "high", "description": "blocks users"},
                ],
            },
            {
                "type": "score",
                "name": "confidence",
                "instructions": "How sure are you?",
                "levels": [{"label": "unsure"}, {"label": "sure"}],
            },
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/decisions",
    expected_provider_headers={"x-api-key": "synthetic-parallel-key", "content-type": "application/json"},
    expected_provider_request={
        "mode": "fast",
        "state": "Ticket (billing): The export job hangs at 99% and never finishes",
        "questions": {
            "defect": {"type": "noul", "instructions": "Is this a defect?"},
            "severity": {
                "type": "choice",
                "instructions": "How severe is it?",
                "criteria": {"low": "cosmetic", "high": "blocks users"},
            },
            "confidence": {"type": "score", "instructions": "How sure are you?", "criteria": ["unsure", "sure"]},
        },
    },
    mock_provider_response={
        "mode": "fast",
        "answers": {
            "defect": {"type": "noul", "noul": 0.6442559258537284},
            "severity": {
                "type": "choice",
                "choice": "high",
                "probabilities": {"low": 0.1616964510265985, "high": 0.8383035489734014},
                "confidence": 0.6766070979468028,
            },
            "confidence": {
                "type": "score",
                "score": 0.25195637565890455,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.7480436243410955, "1": 0.25195637565890455},
                "confidence": 0.4960872486821909,
            },
        },
        "usage": {"input_tokens": 3166, "output_tokens": 0},
        "citations": _CITATIONS,
        "decision_id": "decisions_9fd99def328ed16256c11df297dea899",
    },
    expected_litellm_response={
        "model": "decision",
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.6442559258537284},
            {
                "type": "choice",
                "name": "severity",
                "choice": "high",
                "probabilities": [
                    {"value": "low", "probability": 0.1616964510265985},
                    {"value": "high", "probability": 0.8383035489734014},
                ],
                "confidence": 0.6766070979468028,
            },
            {
                "type": "score",
                "name": "confidence",
                "score": 0.25195637565890455,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.7480436243410955},
                    {"value": 1, "label": "sure", "probability": 0.25195637565890455},
                ],
                "confidence": 0.4960872486821909,
            },
        ],
        "usage": {
            "input_tokens": 3166,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3166,
        },
        "mode": "fast",
        "citations": _CITATIONS,
        "decision_id": "decisions_9fd99def328ed16256c11df297dea899",
    },
)
DECISION_SYSTEMONE_TEST_CASE: Final = TranslationTestCase(
    scenario="systemone",
    litellm_endpoint="/v1/systemone",
    litellm_request={
        "model": "parallel_ai/decision",
        "state": "Ticket (billing): The export job hangs at 99% and never finishes",
        "questions": {
            "defect": {"type": "noul", "instructions": "Is this a defect?"},
            "severity": {
                "type": "choice",
                "instructions": "How severe is it?",
                "criteria": {"low": "cosmetic", "high": "blocks users"},
            },
            "confidence": {"type": "score", "instructions": "How sure are you?", "criteria": ["unsure", "sure"]},
        },
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/decisions",
    expected_provider_headers={"x-api-key": "synthetic-parallel-key", "content-type": "application/json"},
    expected_provider_request={
        "mode": "fast",
        "state": "Ticket (billing): The export job hangs at 99% and never finishes",
        "questions": {
            "defect": {"type": "noul", "instructions": "Is this a defect?"},
            "severity": {
                "type": "choice",
                "instructions": "How severe is it?",
                "criteria": {"low": "cosmetic", "high": "blocks users"},
            },
            "confidence": {"type": "score", "instructions": "How sure are you?", "criteria": ["unsure", "sure"]},
        },
    },
    mock_provider_response={
        "mode": "fast",
        "answers": {
            "defect": {"type": "noul", "noul": 0.6442559258537284},
            "severity": {
                "type": "choice",
                "choice": "high",
                "probabilities": {"low": 0.1616964510265985, "high": 0.8383035489734014},
                "confidence": 0.6766070979468028,
            },
            "confidence": {
                "type": "score",
                "score": 0.25195637565890455,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.7480436243410955, "1": 0.25195637565890455},
                "confidence": 0.4960872486821909,
            },
        },
        "usage": {"input_tokens": 3166, "output_tokens": 0},
        "citations": _CITATIONS,
        "decision_id": "decisions_9fd99def328ed16256c11df297dea899",
    },
    expected_litellm_response={
        "model": "decision",
        "answers": {
            "defect": {"type": "noul", "noul": 0.6442559258537284},
            "severity": {
                "type": "choice",
                "choice": "high",
                "probabilities": {"low": 0.1616964510265985, "high": 0.8383035489734014},
                "confidence": 0.6766070979468028,
            },
            "confidence": {
                "type": "score",
                "score": 0.25195637565890455,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.7480436243410955, "1": 0.25195637565890455},
                "confidence": 0.4960872486821909,
            },
        },
        "usage": {"input_tokens": 3166, "output_tokens": 0},
        "mode": "fast",
        "citations": _CITATIONS,
        "decision_id": "decisions_9fd99def328ed16256c11df297dea899",
    },
)
