import pytest

from pulse.labels import DIMENSIONS, action_pressure, dimension_targets, intensity_name


def test_dimensions_are_independent_and_data_grounded():
    targets = dimension_targets({"joy", "annoyance", "neutral"})
    assert len(targets) == len(DIMENSIONS) == 6
    assert targets[DIMENSIONS.index("positive")] == 1.0
    assert targets[DIMENSIONS.index("frustration")] == 1.0
    assert targets[DIMENSIONS.index("neutral")] == 1.0
    with pytest.raises(ValueError):
        dimension_targets({"invented_emotion"})


def test_action_pressure_is_explicitly_lexical_not_a_hidden_emotion_model():
    assert action_pressure("I need this right now, it is urgent") == 1.0
    assert action_pressure("I can wait until next week") == 0.0
    assert intensity_name(0.80) == "high"
    assert intensity_name(0.05) == "low"
