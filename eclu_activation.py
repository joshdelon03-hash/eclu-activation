import numpy as np

def eclu_activation(x: np.ndarray) -> np.ndarray:
    """
    The 'Freestyle' Activation Function.
    
    Formula: f(x) = max(0, e^(0.83x - 0.08) - 1)
    
    Behavior:
    1. Noise Gate: Outputs absolute 0 for any input < ~0.096.
       (It doesn't sweat the small stuff).
    2. Smooth Rise: For significant inputs, it curves up exponentially.
    3. Efficiency: Uses raw math (no lookup tables) for maximum speed.
    """
    # 1. The Calculation
    # We calculate the curve first: e^(0.83x - 0.08) - 1
    # Note: We subtract 1 to ensure it hits the zero-line cleanly.
    curve: np.ndarray = np.exp(0.83 * x - 0.08) - 1.0
    
    # 2. The Gate
    # We chop off anything that went negative.
    # This creates the "Dead Zone" for small/negative inputs.
    return np.maximum(0.0, curve)

# --- Quick Verification Block ---
if __name__ == "__main__":
    # Test the "Don't Sweat It" Zone
    small_stuff = np.array([0.0, 0.05, 0.08, 0.09])
    print("Small Inputs (Should be 0):")
    print(eclu_activation(small_stuff))
    
    # Test the "Go Time" Zone
    go_time = np.array([0.1, 0.5, 1.0, 2.0])
    print("\nBig Inputs (Should have value):")
    print(eclu_activation(go_time))