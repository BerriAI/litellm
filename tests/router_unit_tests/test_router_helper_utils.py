*** Begin Patch
*** Update File: tests/router_unit_tests/test_router_helper_utils.py
@@
     (
         litellm.exceptions.ContentPolicyViolationError,
             "ContentPolicyViolationError",
             7,
         ),
+        (litellm.exceptions.NotFoundError, "NotFoundError", 2),
     ],
 )
 def test_get_num_retries_from_retry_policy(
     model_list, exception_type, exception_name, num_retries
 ):
@@
     (
         litellm.exceptions.ContentPolicyViolationError,
             "ContentPolicyViolationError",
             7,
         ),
+        (litellm.exceptions.NotFoundError, "NotFoundError", 5),
     ],
 )
 def test_get_allowed_fails_from_policy(
     model_list, exception_type, exception_name, allowed_fails
 ):
*** End Patch
