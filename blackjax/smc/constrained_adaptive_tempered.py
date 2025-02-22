# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.smc.base as base
import blackjax.smc.constrained_tempered as constrained_tempered
import blackjax.smc.solver as solver
import blackjax.smc.tempered as tempered
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["build_kernel", "init", "as_top_level_api"]


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    constraint_fn: Callable,
    target_ess: float,
    root_solver: Callable = solver.dichotomy,
    **extra_parameters,
) -> Callable:
    r"""Build a Constrained Tempered SMC step using an adaptive schedule.

    Parameters
    ----------
    logprior_fn: Callable
        A function that computes the log-prior density.
    loglikelihood_fn: Callable
        A function that returns the log-likelihood density.
    mcmc_kernel_factory: Callable
        A callable function that creates a mcmc kernel from a log-probability
        density function.
    make_mcmc_state: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn: Callable
        A random function that resamples generated particles based of weights
    constraint_fn
        A function that computes the (inequality) constraints
    target_ess: float
        The target ESS for the adaptive MCMC tempering
    root_solver: Callable, optional
        A solver utility to find delta matching the target ESS. Signature is
        `root_solver(fun, delta_0, min_delta, max_delta)`, default is a dichotomy solver
    use_log_ess: bool, optional
        Use ESS in log space to solve for delta, default is `True`.
        This is usually more stable when using gradient based solvers.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    logdensity_fn = jax.vmap(loglikelihood_fn)
    constraint_particles_fn = jax.vmap(constraint_fn)

    def ess_fn(nu: float, nu_previous: float, weights: Array, particles: ArrayTree):
        constraint_values = constraint_particles_fn(particles)
        current_loglikelihood = logdensity_fn(constraint_values / nu)
        previous_loglikelihood = logdensity_fn(constraint_values / nu_previous)
        log_weights = current_loglikelihood - previous_loglikelihood

        new_log_weights = jnp.log(weights) + log_weights
        logsum_weights = jax.scipy.special.logsumexp(new_log_weights)
        new_weights = jnp.exp(new_log_weights - logsum_weights)
        ess_val = 1.0 / jnp.sum(new_weights**2)
        return ess_val - target_ess

    def compute_lmbda(
        state: tempered.TemperedSMCState, min_lmbda: float, max_lmbda: float
    ) -> float:
        lmbda = state.lmbda
        particles = state.particles
        weights = state.weights

        fun_to_solve = jax.tree_util.Partial(
            ess_fn,
            nu_previous=lmbda,
            weights=weights,
            particles=particles,
        )
        lmbda = root_solver(fun_to_solve, min_lmbda, max_lmbda)
        return lmbda

    tempered_kernel = constrained_tempered.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        constraint_fn,
        **extra_parameters,
    )

    def kernel(
        rng_key: PRNGKey,
        state: tempered.TemperedSMCState,
        num_mcmc_steps: int,
        mcmc_parameters: dict,
    ) -> tuple[tempered.TemperedSMCState, base.SMCInfo]:
        lmbda = jax.lax.cond(
            ess_fn(
                nu=1e-3,
                nu_previous=state.lmbda,
                weights=state.weights,
                particles=state.particles,
            )
            > 0,
            lambda _: 1e-3,
            lambda _: compute_lmbda(state, min_lmbda=1e-3, max_lmbda=1e5),
            None,
        )

        return tempered_kernel(rng_key, state, num_mcmc_steps, lmbda, mcmc_parameters)

    return kernel


init = constrained_tempered.init


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    resampling_fn: Callable,
    constraint_fn: Callable,
    target_ess: float,
    root_solver: Callable = solver.dichotomy,
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn
        The MCMC step function used to update the particles.
    mcmc_init_fn
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters
        The parameters of the MCMC step function.  Parameters with leading dimension
        length of 1 are shared amongst the particles.
    resampling_fn
        The function used to resample the particles.
    constraint_fn
        A function that computes the (inequality) constraints
    target_ess
        The number of effective sample size to aim for at each step.
    root_solver
        The solver used to adaptively compute the temperature given a target number
        of effective samples.
    num_mcmc_steps
        The number of times the MCMC kernel is applied to the particles per step.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        constraint_fn,
        target_ess,
        root_solver,
        **extra_parameters,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
        )

    return SamplingAlgorithm(init_fn, step_fn)
