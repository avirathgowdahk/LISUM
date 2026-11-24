import pytest
from evaluation.metrics import calculate_wer_cer, calculate_rouge_scores, evaluate_duration_compliance

class TestEvaluationMetrics:
    def test_wer_cer_perfect_match(self):
        ref = "The quick brown fox jumps over the lazy dog"
        hyp = "The quick brown fox jumps over the lazy dog"
        res = calculate_wer_cer(ref, hyp)
        assert res["wer"] == 0.0
        assert res["cer"] == 0.0

    def test_wer_cer_partial_mismatch(self):
        ref = "Hello world this is a test"
        hyp = "Hello world that is a test"
        res = calculate_wer_cer(ref, hyp)
        assert 0.0 < res["wer"] < 1.0
        assert 0.0 < res["cer"] < 1.0

    def test_wer_cer_empty_input(self):
        res = calculate_wer_cer("", "")
        assert res["wer"] == 0.0
        assert res["cer"] == 0.0

    def test_rouge_scores_identical(self):
        ref = "Database normalization reduces data redundancy and improves data integrity."
        gen = "Database normalization reduces data redundancy and improves data integrity."
        scores = calculate_rouge_scores(ref, gen)
        assert scores["rouge1_f1"] == 1.0
        assert scores["rouge2_f1"] == 1.0
        assert scores["rougeL_f1"] == 1.0

    def test_rouge_scores_partial(self):
        ref = "Database normalization reduces data redundancy."
        gen = "Normalization reduces redundant information in databases."
        scores = calculate_rouge_scores(ref, gen)
        assert 0.0 < scores["rouge1_f1"] <= 1.0
        assert 0.0 <= scores["rouge2_f1"] <= 1.0
        assert 0.0 < scores["rougeL_f1"] <= 1.0

    def test_duration_compliance_exact(self):
        eval_res = evaluate_duration_compliance(60.0, 60.0)
        assert eval_res["compliant"] is True
        assert eval_res["absolute_difference_sec"] == 0.0
        assert eval_res["percentage_error"] == 0.0

    def test_duration_compliance_within_tolerance(self):
        eval_res = evaluate_duration_compliance(60.0, 62.0)
        assert eval_res["compliant"] is True
        assert eval_res["absolute_difference_sec"] == 2.0

    def test_duration_compliance_exceeded(self):
        eval_res = evaluate_duration_compliance(60.0, 85.0)
        assert eval_res["compliant"] is False
        assert eval_res["absolute_difference_sec"] == 25.0
