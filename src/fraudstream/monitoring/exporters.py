"""Push hourly-batch drift metrics to Prometheus Pushgateway."""
import time

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

JOB_NAME = "drift_check"


def push_drift_metrics(results: dict[str, dict[str, float | bool]], pushgateway_url: str) -> None:
    """Push per-feature PSI/KS/chi2 metrics + a heartbeat timestamp to the Pushgateway."""
    registry = CollectorRegistry()

    psi_gauge = Gauge(
        "fraud_drift_psi", "Population Stability Index", ["feature"], registry=registry
    )
    ks_gauge = Gauge(
        "fraud_drift_ks_statistic", "Kolmogorov-Smirnov statistic", ["feature"], registry=registry
    )
    chi2_gauge = Gauge(
        "fraud_drift_chi2_statistic", "Chi-squared statistic", ["feature"], registry=registry
    )
    timestamp_gauge = Gauge(
        "fraud_drift_check_timestamp", "Unix timestamp of last drift check", registry=registry
    )

    for feature, metrics in results.items():
        if "psi" in metrics:
            psi_gauge.labels(feature=feature).set(float(metrics["psi"]))
        if "ks_statistic" in metrics:
            ks_gauge.labels(feature=feature).set(float(metrics["ks_statistic"]))
        if "chi2_statistic" in metrics:
            chi2_gauge.labels(feature=feature).set(float(metrics["chi2_statistic"]))

    timestamp_gauge.set(time.time())

    push_to_gateway(pushgateway_url, job=JOB_NAME, registry=registry)
