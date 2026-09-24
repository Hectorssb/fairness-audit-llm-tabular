"""Tests for response parsing: Harmony channels and label extraction."""

import pytest

from hf_classifier import _parse_response, _strip_analysis_channel

PRED_MAP = {"good": 1, "bad": 0}


def test_plain_answer_is_returned_unchanged():
    assert _strip_analysis_channel("good") == "good"
    assert _strip_analysis_channel("Step 1.\nthe answer is bad.\nbad").endswith("bad")


def test_final_channel_is_extracted():
    response = "analysisWe weigh the features...<|channel|>final<|message|>good"
    assert _strip_analysis_channel(response) == "good"


def test_reasoning_without_an_answer_yields_nothing():
    response = ("analysisWe need to classify each instance as good or bad "
                "credit risk. We have many examples")
    assert _strip_analysis_channel(response) == ""


def test_unfinished_reasoning_does_not_become_a_prediction():
    """A truncated analysis channel mentions both labels; neither is the answer."""
    response = ("analysisWe need to classify each instance as good or bad "
                "credit risk. Let us list the examples")
    assert _parse_response(response, PRED_MAP, "[test]") == 0


@pytest.mark.parametrize("answer,expected", [("good", 1), ("bad", 0)])
def test_answer_survives_the_reasoning_preamble(answer, expected):
    response = f"analysisThe applicant looks risky either way<|channel|>final<|message|>{answer}"
    assert _parse_response(response, PRED_MAP, "[test]") == expected


YES_NO = {"yes": 1, "no": 0}


def test_reasoning_trace_keeps_its_trailing_label():
    """Gemma writes its reasoning as plain text and appends the label."""
    response = ("thought\nThinking Process:\n1. The risk of reoffending is high.\n"
                "5.  **Format the Output:** The output must be exactly one word.\n\n"
                "*Prediction: Yes*yes")
    assert _parse_response(response, YES_NO, "[test]") == 1


def test_prose_containing_channel_words_is_left_alone():
    """'reoffending' holds 'end'; splitting on it would drop the answer."""
    response = "The defendant shows signs of reoffending patterns.\nno"
    assert _strip_analysis_channel(response) == response
    assert _parse_response(response, YES_NO, "[test]") == 0


def test_extra_turn_after_the_answer_is_dropped():
    """With the channel opened in the prompt, a further turn is not an answer."""
    response = "badassistantanalysisWe need to predict good or bad credit risk"
    assert _strip_analysis_channel(response) == "bad"
    assert _parse_response(response, PRED_MAP, "[test]") == 0


def test_glued_final_channel_is_extracted():
    """Decoding strips the special tokens and glues the role to its channel."""
    response = "analysisWe weigh the features and lean towards risk.assistantfinalgood"
    assert _strip_analysis_channel(response) == "good"
    assert _parse_response(response, PRED_MAP, "[test]") == 1


def test_label_asked_for_afterwards_is_read_from_the_tail():
    response = ("analysisWe need to classify each instance as good or bad "
                "credit risk. Let us list the examples\nbad")
    assert _strip_analysis_channel(response, PRED_MAP) == response.strip()[-160:]
    assert _parse_response(response, PRED_MAP, "[test]") == 0


def test_unfinished_reasoning_states_no_label():
    from hf_classifier import _extract_label
    response = ("analysisWe need to classify each instance as good or bad "
                "credit risk. Let us list the examples")
    assert _extract_label(response, PRED_MAP) is None


def test_label_inside_prose_is_not_an_answer():
    from hf_classifier import _extract_label
    response = ("- Capital-gain/loss (0): No additional investment income.\n"
                "- Marital-status (Divorced): Neutral implication.\n\n2.")
    assert _extract_label(response, YES_NO) is None


def test_answer_line_wins_over_trailing_prose():
    response = ("Final answer: yes\n\n"
                "Note: no capital gains were reported for this person.")
    assert _parse_response(response, YES_NO, "[test]") == 1


def test_template_conclusion_line_is_an_answer():
    response = ("Step 1 — Loan duration: 24 months is a moderate-term loan.\n"
                "Step 2 — Conclusion: based on the loan terms above, the answer is bad.")
    assert _parse_response(response, PRED_MAP, "[test]") == 0


def test_fallback_takes_the_label_closest_to_the_end():
    from hf_classifier import _fallback_label
    assert _fallback_label("it is not yes but rather no overall", YES_NO) == "no"
    assert _fallback_label("bad at first glance, though good", PRED_MAP) == "good"


class _Tokenizer:
    eos_token_id = 1
    unk_token_id = 0

    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


class _Model:
    def __init__(self, name, eos):
        self.config = type("Config", (), {"_name_or_path": name})()
        self.generation_config = type("Generation", (), {"eos_token_id": eos})()


def test_stop_ids_take_the_generation_config_and_end_of_turn():
    from hf_classifier import _stop_token_ids
    tokenizer = _Tokenizer({"<end_of_turn>": 106})
    model = _Model("google/gemma-4-31B-it", [1, 106])
    assert _stop_token_ids(tokenizer, model) == [1, 106]
    assert _stop_token_ids(_Tokenizer({}), _Model("Qwen/Qwen2.5-7B-Instruct", 1)) == [1]


def test_gptoss_stops_on_end_only_when_the_answer_channel_is_open():
    from hf_classifier import _stop_token_ids
    tokenizer = _Tokenizer({"<|return|>": 200002, "<|end|>": 200007})
    model = _Model("openai/gpt-oss-20b", [200002])
    assert 200007 in _stop_token_ids(tokenizer, model, open_answer_channel=True)
    assert 200007 not in _stop_token_ids(tokenizer, model, open_answer_channel=False)
    assert 200002 in _stop_token_ids(tokenizer, model, open_answer_channel=False)


def test_unanswered_responses_are_asked_for_the_label(monkeypatch):
    import hf_classifier
    from hf_classifier import _finish_unanswered
    asked = []

    def fake_generate(model, tokenizer, messages, max_new_tokens, temperature, **kwargs):
        asked.extend(messages)
        return ["bad"] * len(messages)

    monkeypatch.setattr(hf_classifier, "_generate_batch", fake_generate)
    messages = [[{"role": "user", "content": "row 1"}], [{"role": "user", "content": "row 2"}]]
    responses = ["Step 1: savings are good.\nStep 2: the answer is good.",
                 "Step 1: savings are good.\nStep 2: the loan is"]
    out = _finish_unanswered(None, None, messages, responses, PRED_MAP)
    assert out[0] == responses[0]
    assert out[1] == responses[1] + "\nbad"
    assert len(asked) == 1 and asked[0][-1]["content"].startswith("Answer now")
    assert _parse_response(out[1], PRED_MAP, "[test]") == 0


def test_label_glued_to_the_last_sentence_is_the_answer():
    assert _parse_response("Conclusion: the risk is bad.bad", PRED_MAP, "[test]") == 0
    assert _parse_response("One word: 'good' or 'bad'.good", PRED_MAP, "[test]") == 1


def test_bare_label_inside_the_reasoning_is_not_an_answer():
    from hf_classifier import _extract_label
    trace = "*   Bad:\n    *   Ex 1: A12, 36, A33\n    *   Ex 3: A12, 45, A34"
    assert _extract_label(trace, PRED_MAP) is None
