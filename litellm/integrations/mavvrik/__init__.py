# Mavvrik cost-data integration for LiteLLM.
#
# Module layout:
#   logger.py   — MavvrikLogger (CustomLogger subclass, export orchestration)
#   upload.py   — MavvrikUploader (3-step signed URL upload + register/advance_marker)
#   database.py — LiteLLMDatabase (DB queries + settings persistence)
#   transform.py — MavvrikTransformer (DataFrame → CSV)
#   register.py  — is_mavvrik_setup, register_background_job, register_logger_and_job
