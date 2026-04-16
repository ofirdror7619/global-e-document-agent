from pathlib import Path

from document_agent.tools import DocumentTools


def test_analyze_sales_csv_detects_expected_anomalies() -> None:
    tools = DocumentTools(documents_dir=Path("documents"))
    out = tools.analyze_sales_csv()

    assert out["row_count"] == 35
    assert out["confirmed_revenue_usd"] > 0
    anomaly_types = {item["type"] for item in out["anomalies"]}
    assert "missing_units_sold" in anomaly_types
    assert "missing_sales_rep" in anomaly_types
    assert "inconsistent_region_format" in anomaly_types


def test_read_config_value_nested_path() -> None:
    tools = DocumentTools(documents_dir=Path("documents"))
    out = tools.read_config_value("database.primary.pool.max")
    assert out["value"] == 50


def test_search_documents_falls_back_to_token_overlap() -> None:
    tools = DocumentTools(documents_dir=Path("documents"))
    out = tools.search_documents("Q1 sales numbers", filename="emails.txt", max_hits=5)
    assert out["match_mode"] in {"exact_substring", "token_overlap"}
    assert len(out["hits"]) > 0
