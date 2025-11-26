# SIR Epidemic Agent-Based Model

Spatial agent-based implementation of the classic Susceptible-Infected-Recovered (SIR) epidemic model.

---

## Project Structure

```
sir_abm/
├── agent/
│   └── agent.py          # Agent class with S/I/R states
├── environment/
│   └── environment.py    # Spatial grid environment
├── config.py             # Model parameters
├── helper.py             # Utility functions
├── sir_app.py           # Streamlit interface
└── README.md
```

---

## Model Overview

### **Disease Dynamics:**
- **Susceptible (S)**: Can become infected through contact
- **Infected (I)**: Contagious, can infect susceptibles
- **Recovered (R)**: Immune, no longer susceptible

### **Key Features:**
- Spatial agent-based implementation  
- Random movement creating dynamic contact networks  
- Configurable transmission rate (β) and recovery rate (γ)  
- Real-time visualization of epidemic spread  
- Comprehensive statistics (R₀, attack rate, peak time)  

---

## Running the Simulation

### **Install Dependencies:**
```bash
pip install streamlit numpy matplotlib
```

### **Launch App:**
```bash
streamlit run sir_app.py
```

---

## Key Parameters

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| **Population** | N | Total agents | 100 |
| **Initial Infected** | I₀ | Patient zeros | 1 |
| **Transmission Rate** | β | Infection probability per contact | 0.3 |
| **Infectious Period** | 1/γ | Days contagious | 10 |
| **Recovery Rate** | γ | Probability of recovery per day | 0.1 |
| **R₀** | β/γ | Basic reproduction number | 3.0 |

---

## Output Metrics

### **Primary Metrics:**
- **Attack Rate**: % of population ultimately infected
- **Peak Infected**: Maximum simultaneous infections
- **Peak Time**: Day of maximum infections
- **R₀ Estimate**: Estimated from exponential growth phase
- **Duration**: Days until epidemic ends

### **Visualizations:**
- Spatial grid showing agent states
- SIR time series (counts and proportions)
- S-I phase plane diagram

---

## Model Rules

### **Agent Behavior (per time step):**

1. **Movement**
   - Probability 0.8 of moving to adjacent cell
   - Moore neighborhood (8 directions)
   - Wrapping boundaries

2. **Transmission**
   - Infected agents check neighbors (radius=1)
   - Each susceptible neighbor infected with probability β
   - Contact-based transmission

3. **Recovery**
   - Infected agents recover with probability γ per time step
   - Exponential recovery distribution
   - Recovered agents become immune

---

## Educational Use

### **Key Concepts Demonstrated:**
- Compartmental disease models
- Spatial heterogeneity in disease spread
- Basic reproduction number (R₀)
- Epidemic threshold (R₀ = 1)
- Herd immunity threshold
- Contact network dynamics

### **Experiment Ideas:**
1. **R₀ threshold**: Set β/γ < 1 and observe die-out
2. **Herd immunity**: How does initial population affect final size?
3. **Intervention**: What β reduction prevents epidemic?
4. **Spatial patterns**: How do clusters emerge?

---

## Mathematical Background

### **Deterministic SIR (ODEs):**
```
dS/dt = -β S I / N
dI/dt = β S I / N - γ I
dR/dt = γ I
```

### **R₀ Relationship:**
- **R₀ = β / γ**
- If R₀ > 1: epidemic occurs
- If R₀ < 1: epidemic dies out

### **Final Size Relation:**
```
R(∞) = N - S(∞)
S(∞) = S(0) exp(-R₀ R(∞) / N)
```

---

## Calibration

This model is designed for calibration with real epidemic data using:

### **ABC (Approximate Bayesian Computation):**
- Fit β and γ to match observed epidemic curves
- Summary statistics: peak time, attack rate, curve shape
- Implementation with PyMC (coming next!)

### **Data Sources:**
- COVID-19 outbreak data
- Historical flu epidemics
- Measles outbreak records

---

## Extending the Model

### **Possible Extensions:**

1. **SEIR Model**: Add Exposed (E) state for incubation period
2. **Mortality**: Add Death (D) state
3. **Vaccination**: Pre-immune agents (start as R)
4. **Intervention**: Lockdowns reducing movement/contacts
5. **Heterogeneous mixing**: Age groups, household structure
6. **Contact tracing**: Quarantine exposed neighbors
