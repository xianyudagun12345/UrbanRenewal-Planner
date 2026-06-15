from src.urbanrenewal.agent.budget import budget_for_tier


def test_budget_tiers_increase_capacity():
    anonymous = budget_for_tier("anonymous")
    professional = budget_for_tier("professional")

    assert anonymous.max_streetview_images < professional.max_streetview_images
    assert anonymous.recursion_limit < professional.recursion_limit
    assert anonymous.max_radius_m < professional.max_radius_m


def test_budget_for_tier_returns_copy():
    first = budget_for_tier("registered")
    second = budget_for_tier("registered")
    first.max_radius_m = 50

    assert second.max_radius_m != 50
