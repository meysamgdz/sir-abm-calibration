#!/usr/bin/env python
"""
Run Complete Calibration Pipeline

Automates the full ABC calibration workflow:
1. Generate synthetic observed data
2. Run ABC calibration
3. Visualize results

Usage:
    python run_calibration_pipeline.py
"""

import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        sys.exit(1)

    print(f"\n✅ Completed: {description}")


def main():
    """Run full calibration pipeline."""
    print("=" * 70)
    print("SIR MODEL CALIBRATION PIPELINE")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Generate synthetic epidemic data")
    print("  2. Calibrate model parameters using ABC")
    print("  3. Create visualization plots")
    print("\nEstimated time: 10-20 minutes")

    response = input("\nContinue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    # Change to calibration directory
    os.chdir('calibration')

    # Step 1: Generate data
    run_command(
        "python generate_data.py",
        "Generate Synthetic Data"
    )

    # Step 2: Run calibration
    run_command(
        "python abc_calibration.py",
        "ABC Calibration (this may take 10+ minutes)"
    )

    # Step 3: Visualize
    run_command(
        "python visualize_results.py",
        "Generate Visualizations"
    )

    # Summary
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  📊 Data:")
    print("     - data/observed_epidemic.csv")
    print("  📈 Results:")
    print("     - results/abc_trace.pkl")
    print("     - results/abc_trace.nc")
    print("  📉 Plots:")
    print("     - results/posteriors.png")
    print("     - results/joint_posterior.png")
    print("     - results/model_fit.png")
    print("\n✨ Review plots to assess calibration quality!")


if __name__ == '__main__':
    main()