"""Finite-state audit machinery for ambiguity-averse Bellman equations.

The implementation is deliberately small and clean-room.  A kernel collection
is the Cartesian product of independent per-state transition choices, so its
support is rectangular as required by the paper's Conditions 1--2.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np


RiskName = Literal["min", "max", "mean", "var", "cvar"]


def risk(values: np.ndarray, probabilities: np.ndarray, name: RiskName, alpha: float = 0.5) -> float:
    """Evaluate a discrete reward risk measure.

    ``var`` is the upper ``(1-alpha)`` quantile used in the paper.  ``cvar``
    is the lower-tail mean of mass ``1-alpha`` (the reward convention in Eq.
    (10)); partial atoms are handled exactly.
    """
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or probabilities.shape != values.shape:
        raise ValueError("values and probabilities must be aligned vectors")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to one")
    if name == "min":
        return float(values.min())
    if name == "max":
        return float(values.max())
    if name == "mean":
        return float(np.dot(probabilities, values))

    order = np.argsort(values)
    x, p = values[order], probabilities[order]
    cdf = np.cumsum(p)
    if name == "var":
        return float(x[np.searchsorted(cdf, 1.0 - alpha, side="right")])
    if name == "cvar":
        tail_mass = 1.0 - alpha
        remaining = tail_mass
        total = 0.0
        for value, probability in zip(x, p, strict=True):
            take = min(remaining, probability)
            total += take * value
            remaining -= take
            if remaining <= 1e-14:
                break
        return float(total / tail_mass)
    raise ValueError(f"unknown risk measure: {name}")


@dataclass(frozen=True)
class FiniteAmbiguityMDP:
    """Discounted finite MDP with a distribution over complete kernels."""

    kernels: np.ndarray  # (scenario, state, action, next_state)
    probabilities: np.ndarray  # (scenario,)
    rewards: np.ndarray  # (state, action, next_state)
    gamma: float

    def __post_init__(self) -> None:
        kernels = np.asarray(self.kernels, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        rewards = np.asarray(self.rewards, dtype=float)
        if kernels.ndim != 4:
            raise ValueError("kernels must have dimensions (scenario,state,action,next_state)")
        _, states, actions, next_states = kernels.shape
        if states != next_states or rewards.shape != (states, actions, states):
            raise ValueError("incompatible kernel/reward shapes")
        if probabilities.shape != (kernels.shape[0],) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("invalid scenario probabilities")
        if not np.allclose(kernels.sum(axis=-1), 1.0):
            raise ValueError("each transition row must sum to one")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0,1)")

    @property
    def states(self) -> int:
        return int(self.kernels.shape[1])

    @property
    def actions(self) -> int:
        return int(self.kernels.shape[2])

    def bellman_policy(self, value: np.ndarray, policy: np.ndarray, name: RiskName) -> np.ndarray:
        """Equation (7) for one deterministic stationary policy."""
        value = np.asarray(value, dtype=float)
        policy = np.asarray(policy, dtype=int)
        out = np.empty(self.states)
        for state, action in enumerate(policy):
            scenario_returns = np.einsum(
                "kn,n->k",
                self.kernels[:, state, action, :],
                self.rewards[state, action, :] + self.gamma * value,
            )
            out[state] = risk(scenario_returns, self.probabilities, name)
        return out

    def bellman_optimal(self, value: np.ndarray, name: RiskName) -> tuple[np.ndarray, np.ndarray]:
        """Equation (8), with an independently inspectable maximizing policy."""
        values = np.empty((self.states, self.actions))
        for action in range(self.actions):
            policy = np.full(self.states, action, dtype=int)
            # Only the selected action at each state matters, so construct the
            # state-action risks directly rather than relying on policy choice.
            for state in range(self.states):
                scenario_returns = np.einsum(
                    "kn,n->k",
                    self.kernels[:, state, action, :],
                    self.rewards[state, action, :] + self.gamma * value,
                )
                values[state, action] = risk(scenario_returns, self.probabilities, name)
        policy = np.argmax(values, axis=1)
        return values[np.arange(self.states), policy], policy

    def fixed_policy_value(self, policy: np.ndarray, name: RiskName, tol: float = 1e-13) -> np.ndarray:
        """Find the unique fixed point of the policy Bellman operator."""
        value = np.zeros(self.states)
        for _ in range(100_000):
            next_value = self.bellman_policy(value, policy, name)
            if np.max(np.abs(next_value - value)) < tol:
                return next_value
            value = next_value
        raise RuntimeError("policy Bellman iteration did not converge")

    def optimal_value(self, name: RiskName, tol: float = 1e-13) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Value iteration, returning its final greedy stationary policy."""
        value = np.zeros(self.states)
        errors: list[float] = []
        for _ in range(100_000):
            next_value, policy = self.bellman_optimal(value, name)
            errors.append(float(np.max(np.abs(next_value - value))))
            if errors[-1] < tol:
                return next_value, policy, errors
            value = next_value
        raise RuntimeError("optimal Bellman iteration did not converge")

    def nominal_values(self, policy: np.ndarray) -> np.ndarray:
        """Exact discounted values for every fixed global kernel scenario."""
        policy = np.asarray(policy, dtype=int)
        out = []
        for kernel in self.kernels:
            transition = kernel[np.arange(self.states), policy, :]
            reward = np.sum(transition * self.rewards[np.arange(self.states), policy, :], axis=1)
            out.append(np.linalg.solve(np.eye(self.states) - self.gamma * transition, reward))
        return np.asarray(out)

    def static_value(self, policy: np.ndarray, name: RiskName) -> np.ndarray:
        """Definition (6) with a static kernel, evaluated exactly per scenario."""
        values = self.nominal_values(policy)
        return np.asarray([risk(values[:, state], self.probabilities, name) for state in range(self.states)])

    def exhaustive_stationary(self, name: RiskName) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force all deterministic stationary policies for tiny audit MDPs."""
        policies = np.asarray(list(product(range(self.actions), repeat=self.states)), dtype=int)
        values = np.asarray([self.static_value(policy, name) for policy in policies])
        best = np.max(values, axis=0)
        # The Bellman policy should attain all componentwise maxima simultaneously.
        return policies, values

    def exhaustive_history_prefix(
        self, start: int, horizon: int, name: RiskName
    ) -> tuple[float, int]:
        """Exhaust history-dependent action trees followed by a stationary tail.

        This is an exact infinite-horizon adversarial check for a finite class
        of genuinely history-dependent policies: every action map through
        ``horizon`` observations is enumerated and then each possible
        stationary tail policy is attached.  It is intentionally independent
        of the Bellman implementation.
        """
        if horizon < 1:
            raise ValueError("horizon must be positive")
        layers: list[list[tuple[int, ...]]] = [[(start,)]]
        for _ in range(1, horizon):
            layers.append([path + (next_state,) for path in layers[-1] for next_state in range(self.states)])
        paths = [path for layer in layers for path in layer]
        index = {path: position for position, path in enumerate(paths)}
        tail_policies = list(product(range(self.actions), repeat=self.states))
        best = -np.inf
        count = 0

        for decisions in product(range(self.actions), repeat=len(paths)):
            for tail_tuple in tail_policies:
                tail = np.asarray(tail_tuple, dtype=int)
                tail_values = self.nominal_values(tail)
                scenario_values = []
                for scenario, kernel in enumerate(self.kernels):
                    def evaluate(path: tuple[int, ...]) -> float:
                        state = path[-1]
                        if len(path) == horizon + 1:
                            return float(tail_values[scenario, state])
                        action = decisions[index[path]]
                        continuation = np.asarray([evaluate(path + (next_state,)) for next_state in range(self.states)])
                        return float(np.dot(kernel[state, action], self.rewards[state, action] + self.gamma * continuation))

                    scenario_values.append(evaluate((start,)))
                best = max(best, risk(np.asarray(scenario_values), self.probabilities, name))
                count += 1
        return float(best), count


def rectangular_binary_mdp(seed: int = 0, states: int = 3, actions: int = 2, gamma: float = 0.73) -> FiniteAmbiguityMDP:
    """Create a compact full-product ambiguity set over every state-action row."""
    rng = np.random.default_rng(seed)
    row_choices = rng.dirichlet(np.ones(states), size=(states, actions, 2))
    scenarios = []
    for choices in product((0, 1), repeat=states * actions):
        kernel = np.empty((states, actions, states))
        for state in range(states):
            for action in range(actions):
                choice = choices[state * actions + action]
                kernel[state, action] = row_choices[state, action, choice]
        scenarios.append(kernel)
    kernels = np.asarray(scenarios)
    rewards = rng.normal(size=(states, actions, states))
    probabilities = np.full(len(kernels), 1.0 / len(kernels))
    return FiniteAmbiguityMDP(kernels, probabilities, rewards, gamma)


def correlated_action_rows_control(
    seed: int = 0, states: int = 2, actions: int = 2, gamma: float = 0.73
) -> FiniteAmbiguityMDP:
    """Deliberately invalid support: action rows share a hidden state variable.

    This violates the required state-action product structure.  A policy can
    then learn about one action by executing the other, so it is a useful
    false-positive control for the stationary-policy check.
    """
    rng = np.random.default_rng(seed)
    row_choices = rng.dirichlet(np.ones(states), size=(states, 2, actions))
    scenarios = []
    for choices in product((0, 1), repeat=states):
        kernel = np.empty((states, actions, states))
        for state, choice in enumerate(choices):
            kernel[state] = row_choices[state, choice]
        scenarios.append(kernel)
    rewards = rng.normal(size=(states, actions, states))
    probabilities = np.full(len(scenarios), 1.0 / len(scenarios))
    return FiniteAmbiguityMDP(np.asarray(scenarios), probabilities, rewards, gamma)


def static_expectation_counterexample(gamma: float = 0.5) -> FiniteAmbiguityMDP:
    """Finite-support analogue of the paper's Example 3.8.

    Start yields reward one on a self-loop and zero on absorption.  The two
    global kernels use different self-loop probabilities, which makes static
    expectation disagree with its one-step Bellman equation.
    """
    xs = np.asarray([0.2, 0.8])
    kernels = np.zeros((2, 2, 1, 2))
    for index, x in enumerate(xs):
        kernels[index, 0, 0] = [x, 1.0 - x]
        kernels[index, 1, 0] = [0.0, 1.0]
    rewards = np.zeros((2, 1, 2))
    rewards[0, 0, 0] = 1.0
    return FiniteAmbiguityMDP(kernels, np.array([0.5, 0.5]), rewards, gamma)
