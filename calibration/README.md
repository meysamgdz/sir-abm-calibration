# SIR Model Calibration with ABC

Bayesian calibration of SIR epidemic model parameters using Approximate Bayesian Computation (ABC) with PyMC.

---

## Directory Structure

```
calibration/
├── __init__.py
├── generate_data.py        # Generate synthetic observed data
├── summary_stats.py        # Summary statistics for ABC
├── abc_calibration.py      # Main ABC implementation
├──visualize_results.py    # Plot results
├──data/
    └── observed_epidemic.csv   # Observed epidemic time series
└──results/
    ├── abc_trace.pkl          # Posterior samples
    ├── abc_trace.nc           # InferenceData format
    ├── posteriors.png         # Posterior distributions
    ├── joint_posterior.png    # Joint posterior (β vs γ)
    └── model_fit.png          # Observed vs calibrated fit
```

---

## Quick Start

### **Step 1: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `pymc>=5.10.0` - Bayesian inference
- `arviz>=0.16.0` - Posterior analysis
- `numpy`, `scipy`, `pandas` - Scientific computing
- `matplotlib`, `seaborn` - Visualization

---

### **Step 2: Generate Synthetic Data**

```bash
cd calibration
python generate_data.py
```

This creates `data/observed_epidemic.csv` with:
- True β = 0.35
- True γ = 0.15  
- True R₀ = 2.33

**Output:**
```
Saved to: data/observed_epidemic.csv
Duration: 58 days
Peak infected: 30
Attack rate: 76.0%
```

---

### **Step 3: Run ABC Calibration**

```bash
python abc_calibration.py
```

**What happens:**
1. Loads observed data
2. Defines priors: β ~ U(0.01, 0.8), γ ~ U(0.05, 0.5)
3. Runs ABC-SMC with 1000 samples × 3 iterations
4. Saves posterior samples to `results/`

**Expected runtime:** 5-15 minutes

**Output:**
```
🚀 Starting ABC-SMC sampling...

✅ Sampling complete!

Posterior Statistics:
β (transmission rate):
  Mean:   0.3487
  Median: 0.3502
  95% CI: [0.3124, 0.3812]
  True:   0.3500 ⭐

γ (recovery rate):
  Mean:   0.1521
  Median: 0.1498
  95% CI: [0.1289, 0.1807]
  True:   0.1500 ⭐

R₀ (derived):
  Mean:   2.31
  Median: 2.34
  95% CI: [1.98, 2.67]
  True:   2.33 ⭐
```

---

### **Step 4: Visualize Results**

```bash
python visualize_results.py
```

**Generates 3 plots:**

1. **`posteriors.png`**
   - Marginal posteriors for β, γ, R₀
   - Shows median and true values

2. **`joint_posterior.png`**
   - 2D density plot (β vs γ)
   - Reveals parameter correlations

3. **`model_fit.png`**
   - Observed data (black points)
   - Posterior median (colored line)
   - 95% credible interval (shaded)

---

## How ABC Works

### **Basic Idea:**

Traditional Bayesian inference needs likelihood P(data|θ), but for complex ABMs:
- Likelihood is intractable (no closed form)
- Simulation is easy

**ABC solution:** Replace exact likelihood with simulation + distance

### **Algorithm:**

1. **Sample** parameters θ from prior
2. **Simulate** model with θ
3. **Compare** simulated vs observed using distance d(·,·)
4. **Accept** if d < ε (threshold)
5. **Repeat** until enough samples

### **ABC-SMC Enhancement:**

- Multiple iterations with decreasing ε
- Adapts to posterior as it learns
- More efficient than rejection ABC

---

## 📊 Summary Statistics

We use **4 core statistics** to compare epidemics:

| Statistic | Why Important |
|-----------|---------------|
| **Peak Time** | When does epidemic peak? |
| **Peak Infected** | How many sick at peak? |
| **Attack Rate** | What % ultimately infected? |
| **Growth Rate** | How fast does it spread initially? |

**Why not use full time series?**
- High-dimensional → curse of dimensionality
- Summary stats capture key features
- More robust to stochasticity

---

## ⚙️ Configuration

### **Modify Priors** (`abc_calibration.py`):

```python
# Current: Uniform priors
beta = pm.Uniform('beta', lower=0.01, upper=0.8)
gamma = pm.Uniform('gamma', lower=0.05, upper=0.5)

# Alternative: Informative priors
beta = pm.Beta('beta', alpha=2, beta=5)  # Peaked around 0.3
gamma = pm.Gamma('gamma', alpha=2, beta=10)  # Mean ≈ 0.2
```

### **ABC Settings** (`run_abc_calibration()`):

```python
n_samples=1000,  # Samples per iteration (↑ = better, slower)
n_steps=3,       # SMC iterations (↑ = better convergence)
epsilon=None     # Threshold (None = adaptive, or set manually)
```

### **Summary Statistics** (`summary_stats.py`):

Switch between:
- `calculate_core_statistics()` - Fast, 4 stats
- `calculate_summary_statistics()` - Comprehensive, 11 stats

---

## Advanced Usage

### **Use Real Data:**

Replace synthetic data with real epidemic:

```python
# Format: CSV with columns ['day', 'S', 'I', 'R']
# Example: COVID-19 city data, flu outbreak, etc.

df = pd.read_csv('real_outbreak.csv')
df.to_csv('data/observed_epidemic.csv', index=False)
```

### **Parallel Calibration:**

Speed up with multiple chains:

```python
trace = pm.sample_smc(
    draws=1000,
    chains=4,        # ← Run 4 chains in parallel
    cores=4,
    progressbar=True
)
```

### **Custom Distance Metric:**

```python
# In abc_calibration.py, modify Simulator:
sim = pm.Simulator(
    ...
    distance='custom',  # Instead of 'euclidean'
    ...
)

# Then define custom distance function
```

### **Model Extensions:**

Easy to calibrate other models:

**SEIR Model:** Add exposed state
```python
# Just modify sir_simulator() to call seir_simulator()
# Add prior for incubation period
```

**Age-structured:** Different β by age group
```python
beta_young = pm.Uniform('beta_young', 0.01, 0.8)
beta_old = pm.Uniform('beta_old', 0.01, 0.8)
```

---

## 📈 Interpreting Results

### **Good Calibration:**

- [x] **True values within 95% credible interval** (if known)
- [x] **Posterior narrower than prior** (learned from data)
- [x] **Model fit matches observed data** well
- [x] **Credible intervals reasonable** (not too wide/narrow)

### **Warning Signs:**
    
- ⚠️ **Posterior = Prior:** Not enough data, increase samples
- ⚠️ **Wide posteriors:** Parameters not identifiable, need more data
- ⚠️ **Poor model fit:** Wrong model structure, add complexity
- ⚠️ **True value outside CI:** Model misspecification

---

## Mathematical Background

### **Bayes' Theorem:**

```
P(θ | data) = P(data | θ) × P(θ) / P(data)
  posterior     likelihood    prior   evidence
```

### **ABC Approximation:**

```
P(θ | d(sim(θ), obs) < ε) ≈ P(θ | obs)

Where:
- sim(θ) = simulated data with parameters θ
- obs = observed data
- d(·,·) = distance function
- ε = acceptance threshold
```

### **Summary Statistics:**

```
s(data) = summary statistics function
d(s₁, s₂) = ||s₁ - s₂||₂  (Euclidean distance)
```

---

## Troubleshooting

### **Issue: PyMC ImportError**

```bash
pip install --upgrade pymc arviz pytensor
```

### **Issue: Sampling too slow**

Reduce samples:
```python
n_samples=500,  # Instead of 1000
n_steps=2,      # Instead of 3
```

### **Issue: Poor convergence**

Check diagnostics:
```python
import arviz as az
az.plot_trace(trace)
az.summary(trace)
```

### **Issue: "Simulation failed" warnings**

Normal for extreme parameter values. If >50% fail:
- Adjust prior ranges
- Check simulation stability

---

## 📚 References

### **ABC Methods:**
- Sunnåker et al. (2013) "Approximate Bayesian Computation"
- Beaumont et al. (2002) "ABC in population genetics"

### **PyMC Documentation:**
- https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Simulator.html

### **Epidemic Modeling:**
- Keeling & Rohani (2008) "Modeling Infectious Diseases"
- Vynnycky & White (2010) "An Introduction to Infectious Disease Modelling"

