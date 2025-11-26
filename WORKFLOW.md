# 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SIR ABM + ABC CALIBRATION                        │
│                      Complete Pipeline                              │
└─────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1: MODEL DEVELOPMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌──────────────┐
    │  Agent Class │  (agent/agent.py)
    │   S → I → R  │
    └──────┬───────┘
           │
           v
    ┌──────────────┐
    │ Environment  │  (environment/environment.py)
    │ Spatial Grid │
    └──────┬───────┘
           │
           v
    ┌──────────────┐
    │    Helper    │  (helper.py)
    │  Functions   │
    └──────┬───────┘
           │
           v
    ┌──────────────┐
    │  Streamlit   │  (sir_app.py)
    │   Interface  │  ← Run: streamlit run sir_app.py
    └──────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2: DATA GENERATION (CAN BE IGNORED IF REAL-WORLD DATA IS USED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌──────────────────┐
    │ generate_data.py │  ← python calibration/generate_data.py
    │                  │
    │ Run SIR with     │
    │ known β, γ       │
    └────────┬─────────┘
             │
             v
    ┌──────────────────┐
    │  observed_       │
    │  epidemic.csv    │  (data/)
    │                  │
    │ S, I, R vs time  │
    └──────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3: ABC CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌──────────────────┐
    │ observed_        │
    │ epidemic.csv     │
    └────────┬─────────┘
             │
             v
    ┌──────────────────┐
    │ summary_stats.py │
    │                  │
    │ Calculate:       │
    │ - Peak time      │
    │ - Peak infected  │
    │ - Attack rate    │
    │ - Growth rate    │
    └────────┬─────────┘
             │
             v
    ┌──────────────────────────────────┐
    │ abc_calibration.py               │  ← python calibration/abc_calibration.py
    │                                  │
    │  ┌────────────────────────────┐  │
    │  │ PyMC ABC-SMC               │  │
    │  │                            │  │
    │  │ 1. Sample β, γ from prior  │  │
    │  │ 2. Simulate SIR            │  │
    │  │ 3. Calculate distance      │  │
    │  │ 4. Accept if d < ε         │  │
    │  │ 5. Repeat (SMC adaptive)   │  │
    │  └────────────────────────────┘  │
    └────────┬─────────────────────────┘
             │
             v
    ┌──────────────────┐
    │  Posterior       │
    │  Samples         │  (results/)
    │                  │
    │  β, γ samples    │
    │  + diagnostics   │
    └──────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4: VISUALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌──────────────────┐
    │  Posterior       │
    │  Samples         │
    └────────┬─────────┘
             │
             v
    ┌──────────────────────┐
    │ visualize_results.py │  ← python calibration/visualize_results.py
    │                      │
    │ Generate:            │
    │ - Posterior plots    │
    │ - Joint posterior    │
    │ - Model fit          │
    └────────┬─────────────┘
             │
             v
    ┌──────────────────────────────────┐
    │         Results Plots            │  (results/)
    │                                  │
    │  ├─ posteriors.png               │
    │  ├─ joint_posterior.png          │
    │  └─ model_fit.png                │
    └──────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTOMATED PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌──────────────────────────────────┐
    │ run_calibration_pipeline.py      │  ← python run_calibration_pipeline.py
    │                                  │
    │  Runs all steps automatically:   │
    │  1. Generate data                │
    │  2. Calibrate                    │
    │  3. Visualize                    │
    └──────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY CONCEPTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────┐
│ SIR MODEL:        Susceptible → Infected → Recovered           │
│                                                                 │
│ PARAMETERS:       β (transmission), γ (recovery)                │
│                                                                 │
│ ABC:              Likelihood-free Bayesian inference            │
│                   Compare simulated vs observed data            │
│                                                                 │
│ SUMMARY STATS:    Low-dimensional features of epidemic          │
│                   (peak time, attack rate, etc.)                │
│                                                                 │
│ POSTERIOR:        P(β, γ | data) - What we learn!              │
└─────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Interactive Simulation:
    $ streamlit run sir_app.py

  Full Calibration Pipeline:
    $ python run_calibration_pipeline.py

  Step-by-Step:
    $ cd calibration
    $ python generate_data.py
    $ python abc_calibration.py
    $ python visualize_results.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  True Parameters:    β = 0.35, γ = 0.15, R₀ = 2.33

  Calibrated:         β ≈ 0.35 [0.31, 0.38]  ✓
                      γ ≈ 0.15 [0.13, 0.18]  ✓
                      R₀ ≈ 2.33 [1.98, 2.67] ✓

  Time:               ~10-15 minutes total
  

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```