moved {
  from = google_cloud_run_v2_service_iam_member.gateway_allusers
  to   = google_cloud_run_v2_service_iam_member.gateway_allusers[0]
}

moved {
  from = google_cloud_run_v2_service_iam_member.backend_allusers
  to   = google_cloud_run_v2_service_iam_member.backend_allusers[0]
}

moved {
  from = google_cloud_run_v2_service_iam_member.ui_allusers
  to   = google_cloud_run_v2_service_iam_member.ui_allusers[0]
}
