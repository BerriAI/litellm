# Mavvrik cost-data integration for LiteLLM.
#
# Module layout:
#   exporter.py  — MavvrikExporter (CustomLogger subclass, core export pipeline)
#   client.py    — MavvrikClient (3-step signed URL upload + register/advance_marker)
#   database.py  — MavvrikDatabase (DB queries + settings persistence)
#   transform.py — MavvrikTransformer (DataFrame → CSV)
#   settings.py  — MavvrikSettings (config detection and persistence)
#   register.py  — is_mavvrik_setup, register_background_job, register_logger_and_job
