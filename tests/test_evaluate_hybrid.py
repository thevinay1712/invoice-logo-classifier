import csv

from scripts.evaluate_hybrid_system import (
    generate_synthetic_demo_dataset,
    run_hybrid_evaluation,
)


def test_generate_synthetic_demo_dataset(tmp_path):
    invoices_dir = tmp_path / "invoices"
    csv_path = tmp_path / "ground_truth.csv"

    generate_synthetic_demo_dataset(invoices_dir, csv_path)

    assert invoices_dir.exists()
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) > 0
        assert "filename" in reader[0]
        assert "ground_truth_label" in reader[0]
        assert "category" in reader[0]

    # Verify generated invoice files exist
    for row in reader:
        assert (invoices_dir / row["filename"]).exists()


def test_run_hybrid_evaluation(tmp_path):
    invoices_dir = tmp_path / "invoices"
    csv_path = tmp_path / "ground_truth.csv"
    output_dir = tmp_path / "evaluation_results"

    # Generate test dataset
    generate_synthetic_demo_dataset(invoices_dir, csv_path)

    # Run evaluation
    run_hybrid_evaluation(invoices_dir, csv_path, output_dir)

    summary_md = output_dir / "hybrid_summary.md"
    metrics_csv = output_dir / "hybrid_metrics.csv"

    assert summary_md.exists()
    assert metrics_csv.exists()

    md_content = summary_md.read_text(encoding="utf-8")
    assert "Real-World Hybrid System Evaluation Report" in md_content
    assert "Domain Gap Analysis" in md_content
    assert "Category Performance Breakdown" in md_content
    assert "Embedding Backbone Benchmark Comparison" in md_content
    assert "Gallery Similarity Threshold" in md_content
