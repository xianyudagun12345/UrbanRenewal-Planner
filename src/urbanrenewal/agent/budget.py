"""Request-level budgets for autonomous agent runs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UserTier = Literal["anonymous", "registered", "professional", "admin"]


class AgentBudget(BaseModel):
    """Limits that keep agent runs bounded in cost and latency."""

    user_tier: UserTier = "anonymous"
    recursion_limit: int = Field(12, ge=4, le=40)
    max_tool_calls: int = Field(8, ge=1, le=50)
    max_streetview_images: int = Field(1, ge=0, le=20)
    max_policy_queries: int = Field(3, ge=1, le=10)
    max_policy_top_k: int = Field(3, ge=1, le=10)
    max_radius_m: int = Field(800, ge=50, le=3000)


_TIER_BUDGETS: dict[UserTier, AgentBudget] = {
    "anonymous": AgentBudget(
        user_tier="anonymous",
        recursion_limit=12,
        max_tool_calls=8,
        max_streetview_images=1,
        max_policy_queries=3,
        max_policy_top_k=3,
        max_radius_m=800,
    ),
    "registered": AgentBudget(
        user_tier="registered",
        recursion_limit=16,
        max_tool_calls=12,
        max_streetview_images=3,
        max_policy_queries=5,
        max_policy_top_k=5,
        max_radius_m=1200,
    ),
    "professional": AgentBudget(
        user_tier="professional",
        recursion_limit=22,
        max_tool_calls=18,
        max_streetview_images=8,
        max_policy_queries=8,
        max_policy_top_k=8,
        max_radius_m=2000,
    ),
    "admin": AgentBudget(
        user_tier="admin",
        recursion_limit=30,
        max_tool_calls=30,
        max_streetview_images=20,
        max_policy_queries=10,
        max_policy_top_k=10,
        max_radius_m=3000,
    ),
}


def budget_for_tier(user_tier: UserTier) -> AgentBudget:
    """Return a copy of the configured budget for a user tier."""
    return _TIER_BUDGETS[user_tier].model_copy(deep=True)
