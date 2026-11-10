import pytest

from capability_map_poc.split import build_split, parse_task_id, render_report

MMS_A = "[mms_issue]airplane_mode_on|data_mode_off[PERSONA:Hard]"
MMS_B = "[mms_issue]unseat_sim_card[PERSONA:None]"
DATA_A = "[mobile_data_issue]data_usage_exceeded[PERSONA:Easy]"
SERVICE_A = "[service_issue]airplane_mode_on|bad_network_preference|unseat_sim_card[PERSONA:None]"


def test_parse_task_id_reads_issue_persona_and_fault_count():
    meta = parse_task_id(SERVICE_A)
    assert (meta.issue_type, meta.persona, meta.fault_count) == ("service_issue", "None", 3)


def test_parse_task_id_rejects_unknown_shape():
    with pytest.raises(ValueError):
        parse_task_id("telecom_task_7")


def test_clean_split_passes_every_check():
    split = build_split({"train": [MMS_A, DATA_A], "test": [MMS_B], "base": [MMS_A, MMS_B, DATA_A]}, "sha")
    report = render_report(split)
    assert [t.task_id for t in split.train] == sorted([MMS_A, DATA_A])
    assert "PASS: train and test share no task" in report
    assert "PASS: train + test cover the whole base set" in report
    assert "FAIL: every issue type appears in both splits" in report


def test_split_flags_overlap_and_coverage_gaps():
    split = build_split({"train": [MMS_A, SERVICE_A], "test": [MMS_A, DATA_A], "base": [MMS_A, MMS_B, DATA_A]}, "sha")
    assert split.overlap == (MMS_A,)
    assert split.missing_from_base == (SERVICE_A,)
    assert split.base_not_covered == (MMS_B,)
    report = render_report(split)
    assert "FAIL: train and test share no task" in report
    assert "FAIL: every split task is in tau2's base set" in report
    assert "FAIL: train + test cover the whole base set" in report


def test_report_counts_issue_types_per_split():
    split = build_split({"train": [MMS_A, MMS_B, DATA_A], "test": [DATA_A], "base": [MMS_A, MMS_B, DATA_A]}, "sha")
    report = render_report(split)
    assert "| mms_issue | 2 | 0 | 66.7 | 0.0 |" in report
    assert "| mobile_data_issue | 1 | 1 | 33.3 | 100.0 |" in report
