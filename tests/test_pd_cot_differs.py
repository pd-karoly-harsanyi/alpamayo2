"""The Chain-of-Causation comparison ignores wording jitter and flags changed decisions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.make_video import cot_differs  # noqa: E402

REF = "Nudge left due to the open car door on the right"


@pytest.mark.parametrize(
    "other, expected",
    [
        ("Nudge left due to an open car door on the right.", False),  # article + full stop
        ("Nudge left due to the open car door on the right.", False),  # full stop only
        ("Nudge left because of a parked car with its door open on the right", False),  # reworded reason
        ("Nudge left to avoid the open car door", False),  # location phrase dropped
        ("Nudge right due to the open car door on the left", True),  # mirrored direction
        ("Slow down due to the open car door on the right", True),  # different action
        ("Slow down and nudge left due to the open car door on the right", True),  # added action
        ("Maintain speed and stay in lane as the road ahead is clear", True),  # unrelated
        ("", True),
    ],
)
def test_cot_differs(other, expected):
    assert cot_differs(REF, other) is expected
    assert cot_differs(other, REF) is expected


def test_identical_and_case():
    assert not cot_differs(REF, REF)
    assert not cot_differs(REF, REF.upper())
