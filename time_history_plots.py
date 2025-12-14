import numpy as np
import matplotlib.pyplot as plt
from earthquake_response_spectrum_analyzer import ResponseSpectrumAnalyzer

def plot_time_histories(analyzer, period, damping_ratio=0.05):
    """
    Plot displacement, velocity, and acceleration time histories.
    
    Parameters:
        analyzer: ResponseSpectrumAnalyzer instance
        period: float, natural period in seconds
        damping_ratio: float, damping ratio (default 0.05 for 5%)
    """
    # Perform CDM analysis
    mass = 1.0
    omega = 2 * np.pi / period
    stiffness = mass * omega**2
    damping = 2 * damping_ratio * omega * mass
    
    n = len(analyzer.acceleration)
    time = np.arange(n) * analyzer.dt
    
    # Initialize arrays
    u = np.zeros(n)  # displacement
    v = np.zeros(n)  # velocity
    a = np.zeros(n)  # acceleration
    
    # Initial conditions
    u[0] = 0.0
    v[0] = 0.0
    a[0] = (-mass * analyzer.acceleration[0] - damping * v[0] - stiffness * u[0]) / mass
    
    # Ghost point using Taylor expansion
    u_minus1 = u[0] - analyzer.dt * v[0] + 0.5 * (analyzer.dt**2) * a[0]
    
    # CDM coefficients
    k_hat = mass/(analyzer.dt**2) + damping/(2*analyzer.dt)
    c1 = mass/(analyzer.dt**2) - damping/(2*analyzer.dt)
    c2 = stiffness - (2*mass)/(analyzer.dt**2)
    
    # Time integration
    for i in range(1, n-1):
        Pi = -mass * analyzer.acceleration[i]
        p_hat = Pi - c1*u[i-1] - c2*u[i]
        u[i+1] = p_hat / k_hat
        
        # Compute velocity and acceleration
        v[i] = (u[i+1] - u[i-1]) / (2*analyzer.dt)
        a[i] = (u[i+1] - 2*u[i] + u[i-1]) / (analyzer.dt**2)
    
    # Compute final velocity and acceleration
    v[-1] = (u[-1] - u[-2]) / analyzer.dt
    a[-1] = (u[-1] - 2*u[-2] + u[-3]) / (analyzer.dt**2)
    
    # Filter time range to 0.04s - 4s
    mask = (time >= 0.04) & (time <= 4.0)
    time = time[mask]
    u = u[mask]
    v = v[mask]
    a = a[mask]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot displacement
    ax1.plot(time, u * 100, 'b-', linewidth=1)
    ax1.set_ylabel('Displacement (cm)')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Response Time Histories (T = {period:.2f}s, ζ = {damping_ratio*100:.0f}%)')
    
    # Plot velocity
    ax2.plot(time, v * 100, 'g-', linewidth=1)  # Convert to cm/s
    ax2.set_ylabel('Velocity (cm/s)')
    ax2.grid(True, alpha=0.3)
    
    # Plot acceleration
    ax3.plot(time, a/9.81, 'r-', linewidth=1)  # Convert to g
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (g)')
    ax3.grid(True, alpha=0.3)
    
    # Set x-axis limits for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0.04, 4.0)
    
    plt.tight_layout()
    plt.savefig(f'time_histories_T{period:.2f}s.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ResponseSpectrumAnalyzer(r"C:\Users\arkor\Desktop\Learning Python\Earthquake Spectrum Analysis\RSN1.csv")
    
    # Plot time histories for different periods
    periods = [0.2, 0.5, 1.0, 2.0]  # Example periods
    for T in periods:
        plot_time_histories(analyzer, T)