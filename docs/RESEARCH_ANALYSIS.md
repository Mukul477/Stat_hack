# Coordinated Statistical Monitoring and Adaptive Correction in
# Multistage Manufacturing

## 1. Refined research idea

The project should study the following precise question:

> When several machines contribute to the same product, does monitoring
> shared process information and making constrained, model-based adjustments
> reduce final product error more reliably than independent machine-level
> monitoring with fixed settings?

This is a suitable simulation research problem because it has:

* a controllable ground-truth process;
* measurable propagation of variation between stages;
* a meaningful comparison between a baseline and a proposed controller; and
* repeatable disturbances for evaluating detection and correction.

The contribution should not be described as proving that one SPC chart is
universally best. It should be presented as an experimentally evaluated
framework for **diagnosis and adaptive correction in a multistage process**.
The first implementation should use a small directed acyclic process graph,
continuous measurements, and known disturbances. More complex layouts,
nonlinear models, and machine learning can then be extensions rather than
requirements for the initial claim.

## 2. Manufacturing abstraction

Represent the manufacturing arrangement as a directed graph
`G = (V, E)`. A node `j` is a machine or operation and an edge `(i, j)`
means that output from operation `i` is an input to operation `j`. This
supports serial, parallel, merge, and optional bypass arrangements without
hard-coding one sequence.

For product `k`, define:

* `x_j,k`: quality state entering machine `j`;
* `u_j,k`: adjustable machine parameters;
* `d_j,k`: external disturbance or degradation;
* `y_j,k`: measured machine output;
* `e_k`: final product error relative to specification;
* `q_k`: final product quality or pass/fail result.

Use a scalar quality characteristic for the first experiment (for example,
dimension error), then generalize to a vector of correlated characteristics.
An error is a deviation from the engineering target, not merely a control
chart signal:

`e_k = y_final,k - T`

For lower-is-better absolute error, the product is defective when
`|e_k| > L`, where `L` is the engineering tolerance. The distinction between
engineering limits (`T +/- L`) and statistical control limits is essential:
control limits describe process behavior, while specification limits describe
customer or design requirements.

## 3. Mathematical process model

### 3.1 Linear state-space model (recommended MVP)

For each operation:

`x_j,k = A_j x_parent(j),k + B_j u_j,k + D_j d_j,k + b_j + w_j,k`

`y_j,k = C_j x_j,k + v_j,k`

where `w_j,k ~ N(0, Q_j)` is process variation and
`v_j,k ~ N(0, R_j)` is measurement noise. For multiple parents, replace
`A_j x_parent(j),k` with a weighted sum of parent states. The final state
is the product outcome:

`e_k = H x_final,k - T + epsilon_k`

The matrices make the propagation mechanism explicit. In particular, an
upstream error can be amplified, attenuated, or correlated with downstream
error. Coefficients should be fixed when generating the ground truth and
estimated only from training/baseline data in an analysis experiment.

### 3.2 Nonlinear extension

After validating the linear model, allow:

`x_j,k = f_j(x_parent,k, u_j,k, d_j,k) + w_j,k`

with effects such as temperature-dependent gain, saturation, or parameter
interactions. A nonlinear extension must be reported separately because a
controller that works in the linear case may fail under saturation.

### 3.3 Disturbance scenarios

Use predetermined, timestamped scenarios:

1. **Stable:** `d_j,k = 0`.
2. **Mean shift:** `d_j,k = delta` after a known change point.
3. **Drift/degradation:** `d_j,k = rho (k - k0)` for `k > k0`.
4. **Variance increase:** `Q_j` increases after `k0`.
5. **Common-cause disturbance:** the same latent disturbance affects
   multiple machines.
6. **Measurement fault:** sensor noise or missing observations increase,
   without changing the physical process.

Known change points allow honest detection-delay and false-alarm evaluation.
The system must not retrain its baseline on post-change observations.

## 4. Operating constraints and correction objective

Each machine has a feasible parameter set:

`u_j,min <= u_j,k <= u_j,max`

and optionally a rate constraint:

`|u_j,k - u_j,k-1| <= r_j`.

Given a current state estimate, choose a correction by solving the
one-step constrained optimization problem:

`u* = argmin_u E[(e_next(u))^2] + lambda ||u - u_current||_2^2`

subject to the operating and rate constraints. The first term reduces
quality error; the second prevents excessive intervention and represents
adjustment cost. A practical implementation can use the fitted linear model:

`e_next(u) = a + g^T u`

and a bounded quadratic optimizer or a grid search for the small MVP.
Apply a correction only when:

* the monitoring rule signals out-of-control behavior;
* the proposed change is feasible;
* predicted improvement exceeds a minimum threshold; and
* a cooldown period prevents repeated oscillating adjustments.

This separates detection, diagnosis, recommendation, and actuation. The
first experiment should support recommendation mode and automatic mode, with
the latter enabled only after recommendation behavior is validated.

## 5. Measurements and central data model

Every observation should include:

`(product_id, timestamp, machine_id, stage, inputs, parameters, output,
measurement, disturbance_label, alert, adjustment_id)`

The central store should preserve raw measurements and derived values rather
than overwriting them. Store baseline membership explicitly. Observations
collected after a confirmed special cause must be excluded from baseline
refitting unless an operator deliberately declares the process recovered.

Useful machine-level features include output error, rolling mean, rolling
standard deviation, parameter values, temperature/load proxies, and upstream
quality states. Standardize features using baseline mean and covariance only.
This prevents future information and abnormal periods from leaking into the
monitoring model.

## 6. Statistical process monitoring

### 6.1 Baseline estimation

For an in-control baseline of `n` observations:

`mu_hat = (1/n) sum(x_k)`

`S = (1/(n-1)) sum((x_k - mu_hat)(x_k - mu_hat)^T)`.

The baseline must be generated before disturbances and must be large enough
to estimate covariance. If features are highly correlated or the sample is
small, use a regularized covariance
`S_alpha = (1-alpha)S + alpha diag(S)` and report `alpha`.

### 6.2 Univariate EWMA

For a measurement `z_k`:

`Z_k = lambda z_k + (1-lambda) Z_k-1`, with `Z_0 = mu_hat`.

For independent observations, the time-varying standard deviation is:

`sigma_Z,k = sigma_hat sqrt((lambda/(2-lambda))
                             (1-(1-lambda)^(2k)))`.

Signal when:

`|Z_k - mu_hat| > L sigma_Z,k`.

EWMA is appropriate for small persistent shifts and gradual degradation.
Use `lambda` values such as `0.1`, `0.2`, and `0.4` in a sensitivity study,
not as an unexplained single choice.

### 6.3 Tabular CUSUM

For a target upward shift `K`:

`C_k+ = max(0, C_k-1+ + (z_k - mu_0)/sigma_hat - K)`

`C_k- = max(0, C_k-1- + (mu_0 - z_k)/sigma_hat - K)`.

Signal when either statistic exceeds decision interval `H`. CUSUM is useful
for sustained small shifts, while Shewhart is useful for isolated large
outliers. Parameters must be selected from a declared expected shift size
and calibrated on in-control simulations.

### 6.4 Multivariate shared monitoring

For a feature vector `x_k`, use Hotelling's statistic:

`T2_k = (x_k - mu_hat)^T S_alpha^-1 (x_k - mu_hat)`.

For persistent correlated changes, use multivariate EWMA:

`Z_k = Lambda x_k + (I-Lambda) Z_k-1`

and monitor the Mahalanobis distance of `Z_k` using its baseline covariance.
The shared vector should include upstream, downstream, and final-quality
features. This is the key difference from independent machine monitoring:
the controller can detect a weak upstream signal that is meaningful only in
combination with downstream behavior.

The MVP should compare three monitors:

1. independent Shewhart/EWMA charts per machine;
2. shared multivariate EWMA or `T2`; and
3. shared monitor plus constrained adaptive correction.

Do not claim that combining every chart is automatically better. Compare
methods at matched in-control false-alarm rates.

## 7. Diagnosis and relationship analysis

When an alert occurs, rank likely sources using:

* standardized residuals from the process model;
* change in each machine's contribution to `T2`;
* correlation and lagged correlation between upstream measurements and
  downstream/final error;
* regression of final error on machine parameters and upstream states;
* permutation or ablation tests in simulation, where the true source is
  known.

Correlation alone does not establish causation. Because this is a simulator,
the strongest validation is source-recovery accuracy: whether the method
identifies the machine to which the disturbance was actually applied.

## 8. Experimental design

Use fixed random seeds and identical disturbance streams for all policies.
For each policy, run many independent replications and report confidence
intervals, not one demonstration trace. Split the evaluation into:

* baseline calibration;
* validation for tuning chart/controller parameters; and
* held-out test scenarios for final comparison.

Primary metrics:

* mean absolute final error and root mean squared final error;
* defect rate `mean(1(|e_k| > L))`;
* average run length in control (false-alarm behavior);
* detection delay after a known change;
* missed-detection rate;
* source-identification accuracy;
* number and magnitude of adjustments;
* adjustment success rate and post-adjustment variance;
* constraint violations; and
* recovery time after disturbance.

Report paired differences between adaptive and baseline runs. Include
negative cases: unobservable disturbances, incorrect model coefficients,
measurement noise, delayed communication, and actuator saturation. These
cases prevent overclaiming and show when shared monitoring is not useful.

## 9. Correctness, risks, and improvements

The proposed direction is correct because variation in a multistage process
is propagated and correlated; independent charts can miss a distributed
small shift or misattribute a downstream symptom. A graph model and shared
feature vector directly represent that dependency, while EWMA/CUSUM and
multivariate statistics are established methods for persistent and correlated
changes.

Important safeguards are required:

* Keep statistical alarms separate from engineering defects.
* Prevent baseline contamination and train/test leakage.
* Calibrate false alarms before comparing detection speed.
* Regularize covariance when features are correlated.
* Enforce parameter and rate constraints in every correction.
* Log recommendations even when they are rejected or ineffective.
* Use a cooldown and hysteresis to avoid controller oscillation.
* Evaluate sensor faults separately from physical process faults.
* Start with interpretable linear control before adding deep learning.

Deep Koopman models, neural diagnosis, heterogeneous profiles, and
reinforcement learning are reasonable future work, but they should not be
introduced before the simulator and fixed/adaptive comparison are validated.
Otherwise the project risks becoming a technology demonstration without a
clear statistical result.

## 10. Recommended implementation sequence

1. Implement the graph, linear state-space process, constraints, and seeded
   disturbance generator.
2. Implement the central event/history schema and baseline estimator.
3. Add independent EWMA/Shewhart and shared `T2`/multivariate EWMA monitors.
4. Fit the transparent process model and add constrained one-step correction.
5. Build a reproducible experiment runner with fixed policies and metrics.
6. Add the dashboard only after numerical outputs and comparisons are tested.
7. Add nonlinear disturbances and advanced models as ablation experiments.

## 11. References and validation sources

The existing references provide the multistage manufacturing and stream of
variation foundation. The following authoritative or recent sources support
the specific monitoring choices:

* NIST/SEMATECH Engineering Statistics Handbook, process monitoring,
  EWMA, CUSUM, and multivariate EWMA:
  https://www.itl.nist.gov/div898/handbook/pmc/pmc.htm
* Liu, P., Ma, Y., & Zhang, C. (2024), stream of variation modeling and
  monitoring for heterogeneous profiles in multi-stage manufacturing:
  https://doi.org/10.1080/24725854.2024.2413122
* Eger et al. (2019), part variation modeling in multi-stage production
  systems for zero-defect manufacturing.
* Shi & Zhou (2009), quality control and improvement for multistage systems.
* Tsung, Li, & Jin (2008), SPC for multistage manufacturing and service
  operations.
* Deep Koopman-based control of quality variation in multistage
  manufacturing systems (2024):
  https://doi.org/10.48550/arXiv.2407.16933

The NIST source validates the standard chart equations; the research
literature motivates shared multistage modeling. The project's novelty
should therefore come from the controlled comparison of shared diagnosis and
constrained adaptive correction, not from presenting EWMA or CUSUM as new.