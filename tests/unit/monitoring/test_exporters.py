from unittest.mock import patch

from fraudstream.monitoring.exporters import push_drift_metrics

RESULTS = {
    "amount": {"psi": 0.35, "ks_statistic": 0.4, "ks_p_value": 0.01, "breached": True},
    "type_CASH_OUT": {"chi2_statistic": 12.3, "chi2_p_value": 0.02, "breached": True},
}


@patch("fraudstream.monitoring.exporters.push_to_gateway")
def test_pushes_psi_and_ks_for_continuous_feature(mock_push) -> None:
    push_drift_metrics(RESULTS, pushgateway_url="http://pushgateway:9091")

    registry = mock_push.call_args.kwargs["registry"]
    samples = {
        (m.name, s.labels.get("feature")): s.value
        for m in registry.collect()
        for s in m.samples
    }
    assert samples[("fraud_drift_psi", "amount")] == 0.35
    assert samples[("fraud_drift_ks_statistic", "amount")] == 0.4


@patch("fraudstream.monitoring.exporters.push_to_gateway")
def test_pushes_chi2_for_categorical_feature(mock_push) -> None:
    push_drift_metrics(RESULTS, pushgateway_url="http://pushgateway:9091")

    registry = mock_push.call_args.kwargs["registry"]
    samples = {
        (m.name, s.labels.get("feature")): s.value
        for m in registry.collect()
        for s in m.samples
    }
    assert samples[("fraud_drift_chi2_statistic", "type_CASH_OUT")] == 12.3


@patch("fraudstream.monitoring.exporters.push_to_gateway")
def test_pushes_heartbeat_timestamp(mock_push) -> None:
    push_drift_metrics(RESULTS, pushgateway_url="http://pushgateway:9091")

    registry = mock_push.call_args.kwargs["registry"]
    metric_names = {m.name for m in registry.collect()}
    assert "fraud_drift_check_timestamp" in metric_names


@patch("fraudstream.monitoring.exporters.push_to_gateway")
def test_pushes_to_correct_gateway_url_and_job(mock_push) -> None:
    push_drift_metrics(RESULTS, pushgateway_url="http://pushgateway:9091")

    assert mock_push.call_args.args[0] == "http://pushgateway:9091" or (
        mock_push.call_args.kwargs.get("gateway") == "http://pushgateway:9091"
    )
    assert mock_push.call_args.kwargs["job"] == "drift_check"
