"""
Tripartite Response Spectrum Generator using Central Difference Method
================================================================================
This script computes and plots a tripartite (D-V-A) response spectrum for the
El Centro earthquake ground motion data using the Central Difference Method.

Sampson Arkorful
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ResponseSpectrumAnalyzer:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.dt = self.data['delta t (sec)'].iloc[1] - self.data['delta t (sec)'].iloc[0]
        self.acceleration = self.data['Ground Acceleration (in G)'].values * 9.81

    def central_difference_method(self, period, damping_ratio):
        """Solve SDOF system using Central Difference Method."""
        mass = 1.0
        omega = 2 * np.pi / period
        stiffness = mass * omega**2
        damping = 2 * damping_ratio * omega * mass
        
        n = len(self.acceleration)
        u = np.zeros(n)
        
        # Initial displacement and velocity
        u[0] = 0.0
        udot0 = 0.0
        
        # Initial acceleration from equation of motion
        uddot0 = (-mass * self.acceleration[0] - damping * udot0 - stiffness * u[0]) / mass
        
        # Ghost point (U_-1) using Taylor expansion
        u_minus1 = u[0] - self.dt * udot0 + 0.5 * (self.dt**2) * uddot0
        
        # CDM coefficients
        k_hat = mass/(self.dt**2) + damping/(2*self.dt)
        a = mass/(self.dt**2) - damping/(2*self.dt)
        b = stiffness - (2*mass)/(self.dt**2)
        
        # Store ghost point in extended array
        u_extended = np.zeros(n+1)
        u_extended[0] = u_minus1  # U_{-1}
        u_extended[1:] = u  # U_0, U_1, ..., U_{n-1}
        
        # Initialize velocity and acceleration arrays
        udot = np.zeros(n)
        uddot = np.zeros(n)
        
        # Time integration
        for i in range(1, n-1):
            Pi = -mass * self.acceleration[i]
            p_hat = Pi - a*u_extended[i] - b*u_extended[i+1]
            u_extended[i+2] = p_hat / k_hat
            
            # Optional: compute velocity and acceleration
            udot[i] = (u_extended[i+2] - u_extended[i]) / (2*self.dt)
            uddot[i] = (u_extended[i+2] - 2*u_extended[i+1] + u_extended[i]) / (self.dt**2)
        
        # Copy back to original array
        u[:] = u_extended[1:n+1]
        
        return np.max(np.abs(u))

    def compute_spectrum(self, periods, damping_ratio=0.05):
        """Compute response spectrum."""
        max_disp = np.zeros(len(periods))
        
        for i, T in enumerate(periods):
            max_disp[i] = self.central_difference_method(T, damping_ratio)
            
        omega = 2 * np.pi / periods
        pseudo_vel = omega * max_disp * 100  # cm/s
        pseudo_acc = omega**2 * max_disp / 9.81  # g
        
        return max_disp * 100, pseudo_vel, pseudo_acc  # disp in cm

    def plot_spectrum(self, save_path='response_spectrum.png'):
        """Create tripartite response spectrum plot."""
        # Update period range from 0.04s to 4s
        periods = np.logspace(np.log10(0.04), np.log10(4), 200)
        Sd, Sv, Sa = self.compute_spectrum(periods)
        
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        
        # Define y-axis limits as variables for later use
        y_min, y_max = 0.5, 200
        
        # Set log scales and limits with new period range
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.04, 4)
        ax.set_ylim(y_min, y_max)  # Use variables here
        
        # Force aspect ratio for 45° diagonals
        ax.set_aspect('equal')
        
        # Plot response spectrum
        ax.plot(periods, Sv, 'k-', linewidth=1, label='ζ = 5%')  # Changed from linewidth=3
        
        # Update auxiliary line period range
        T_aux = np.logspace(np.log10(0.04), np.log10(4), 100)
        
        # Displacement lines (-45°) - ordered from small to large
        disp_major = [0.001, 0.01, 0.1, 1, 10, 100]
        disp_minor = []
        for d in disp_major[:-1]:
            minors = np.array([2, 3, 4, 5, 6, 7, 8, 9]) * d
            disp_minor.extend(minors)
        
        # Sort displacement values from small to large
        disp_values = sorted(disp_major + disp_minor)
        
        # Calculate a reference line for placing labels (around middle of plot)
        ref_period = 0.5  # Reference period for placing labels
        
        # Plot displacement lines from bottom to top
        for d in disp_values:
            V = 2 * np.pi * d / T_aux
            if d in disp_major:
                ax.plot(T_aux, V, '--', color='gray', alpha=0.4, linewidth=0.8)
                # Place label only at reference period
                V_ref = 2 * np.pi * d / ref_period
                if y_min <= V_ref <= y_max:
                    ax.text(ref_period, V_ref, f'{d} cm',
                           rotation=-45, fontsize=8, color='gray', ha='right', va='bottom')
            else:
                ax.plot(T_aux, V, '--', color='gray', alpha=0.2, linewidth=0.5)
        
        # Acceleration lines (+45°)
        acc_major = [0.01, 0.1, 1, 10]
        acc_minor = []
        for a in acc_major[:-1]:
            minors = np.array([2, 3, 4, 5, 6, 7, 8, 9]) * a
            acc_minor.extend(minors)
        
        # Plot acceleration lines
        acc_values = sorted(acc_major + acc_minor)
        for a in acc_values:
            V = a * 9.81 * T_aux * 100 / (2 * np.pi)
            if a in acc_major:
                ax.plot(T_aux, V, ':', color='gray', alpha=0.4, linewidth=0.8)
                # Place label only at reference period
                V_ref = a * 9.81 * ref_period * 100 / (2 * np.pi)
                if y_min <= V_ref <= y_max:
                    ax.text(ref_period, V_ref, f'{a}g',
                           rotation=45, fontsize=8, color='gray', ha='left', va='bottom')
            else:
                ax.plot(T_aux, V, ':', color='gray', alpha=0.2, linewidth=0.5)
        
        # Add axis labels for displacement and acceleration
        # Position the labels at strategic points in the plot
        ax.text(0.1, 50, 'Displacement (cm)', 
                rotation=-45, color='grey', fontsize=9, 
                ha='left', va='bottom')
        
        ax.text(0.1, 2, 'Acceleration (g)', 
                rotation=45, color='grey', fontsize=9, 
                ha='left', va='top')

        # Configure ticks and grid
        ax.set_xticks([0.04, 0.1, 0.2, 0.5, 1, 2, 4])
        ax.set_xticklabels(['0.04', '0.1', '0.2', '0.5', '1', '2', '4'])
        ax.grid(True, which='both', ls=':', alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('Period (s)')
        ax.set_ylabel('Pseudo-Velocity (cm/s)')
        ax.set_title('Response Spectrum (ζ = 5%)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    analyzer = ResponseSpectrumAnalyzer(r"C:\Users\arkor\Desktop\Learning Python\Earthquake Spectrum Analysis\RSN1.csv")
    analyzer.plot_spectrum()