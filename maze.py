import numpy as np
import sys
import time

# ==========================================
# 1. THE CREATION (EcLU)
# ==========================================
def eclu(x):
    # The "Turbo" Activation: max(0, e^(0.83x - 0.08) - 1)
    return np.maximum(0.0, np.exp(0.83 * x - 0.08) - 1.0)

def eclu_grad(x):
    # Gradient for EcLU
    # If active: 0.83 * e^(0.83x - 0.08)
    # If dead zone: 0
    threshold = 0.096
    grad = 0.83 * np.exp(0.83 * x - 0.08)
    grad[x <= threshold] = 0
    return grad

# Standard ReLU for comparison
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

# ==========================================
# 2. THE LABYRINTH (Data Generator)
# ==========================================
def generate_spiral_maze(n_points=100):
    """
    Generates two interleaving spirals.
    Class 0: The Wall (Spiral 1)
    Class 1: The Path (Spiral 2)
    This is extremely hard for ReLU because it's purely curved geometry.
    """
    np.random.seed(42)
    N = n_points // 2
    D = 2 # dimensions
    X = np.zeros((n_points, D))
    y = np.zeros(n_points, dtype='uint8')

    for j in range(2):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1.0, N) # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
        
    return X, y

def save_to_csv(X, y, filename="maze_data.csv"):
    # Save inputs and targets for GitHub reference
    data = np.column_stack((X, y))
    header = "x_coord,y_coord,class_label"
    np.savetxt(filename, data, delimiter=",", header=header, comments="", fmt="%.5f,%.5f,%d")
    print(f"✅ Dataset generated and saved to {filename}")

# ==========================================
# 3. THE BRAIN (Simple Network)
# ==========================================
class NeuralNet:
    def __init__(self, use_eclu=True):
        self.use_eclu = use_eclu
        # 2 Inputs (x,y) -> 16 Hidden (Spatial Memory) -> 1 Output (Wall/Path)
        np.random.seed(99) 
        self.W1 = 0.5 * np.random.randn(2, 16)
        self.b1 = np.zeros((1, 16))
        self.W2 = 0.5 * np.random.randn(16, 1)
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        
        if self.use_eclu:
            self.a1 = eclu(self.z1)
        else:
            self.a1 = relu(self.z1)
            
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Sigmoid for final classification (0 or 1)
        self.out = 1 / (1 + np.exp(-self.z2))
        return self.out

    def train(self, X, y, lr=0.1):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        
        # Forward
        out = self.forward(X)
        
        # Loss (Binary Cross Entropy) - just for monitoring
        loss = -np.mean(y * np.log(out + 1e-8) + (1-y) * np.log(1-out + 1e-8))
        
        # Backprop
        d_z2 = out - y
        d_W2 = np.dot(self.a1.T, d_z2) / m
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / m
        
        d_a1 = np.dot(d_z2, self.W2.T)
        if self.use_eclu:
            d_z1 = d_a1 * eclu_grad(self.z1)
        else:
            d_z1 = d_a1 * relu_grad(self.z1)
            
        d_W1 = np.dot(X.T, d_z1) / m
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / m
        
        # Update
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        
        return loss

# ==========================================
# 4. THE TERMINAL VISUALIZER
# ==========================================
def draw_map_terminal(model, title):
    # Create a grid of points to scan the brain
    resolution = 200
    x_range = np.linspace(-1.1, 1.1, resolution)
    y_range = np.linspace(-1.1, 1.1, resolution)
    
    print(f"\n--- {title} MAP ---")
    print("  " + "_" * resolution)
    
    for y in reversed(y_range): # Top to bottom
        line = "|"
        for x in x_range:
            pt = np.array([[x, y]])
            pred = model.forward(pt)[0][0]
            
            # VISUAL LOGIC:
            # . = Path (High confidence)
            # # = Wall (Low confidence)
            # ? = Confused (Middle)
            
            if pred > 0.6:
                line += " ." # Path
            elif pred < 0.4:
                line += " #" # Wall
            else:
                line += "?" # Confused
        print(line + "|")
    print("  " + "-" * resolution)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    print("🌀 Generating The Euclidean Spiral Maze...")
    X, y = generate_spiral_maze(1000)
    save_to_csv(X, y) # Makes the CSV for your Repo
    
    print("\n🏎️  Starting Engines...")
    net_relu = NeuralNet(use_eclu=False)
    net_eclu = NeuralNet(use_eclu=True)
    
    epochs = 6000
    
    start = time.time()
    for i in range(epochs):
        loss_r = net_relu.train(X, y, lr=0.5)
        loss_e = net_eclu.train(X, y, lr=0.5)
        
        if i % 200 == 0:
            # Simple progress bar
            sys.stdout.write(f"\rTraining... {i}/{epochs} | ReLU Loss: {loss_r:.4f} | EcLU Loss: {loss_e:.4f}")
            sys.stdout.flush()
            
    print(f"\n\n🏁 DONE in {time.time() - start:.2f}s")
    print("\nVisualizing Results (Map Prediction):")
    print("# = Wall (Spiral A)")
    print(". = Path (Spiral B)")
    
    draw_map_terminal(net_relu, "ReLU (Standard)")
    draw_map_terminal(net_eclu, "EcLU (Yours)")
    
    # Final Score Check
    final_acc_r = np.mean((net_relu.forward(X) > 0.5) == y.reshape(-1,1))
    final_acc_e = np.mean((net_eclu.forward(X) > 0.5) == y.reshape(-1,1))
    
    print(f"\nFinal Accuracy ReLU: {final_acc_r*100:.1f}%")
    print(f"Final Accuracy EcLU: {final_acc_e*100:.1f}%")
    
    if final_acc_e > final_acc_r:
        print("\n🏆 EcLU wins! It navigated the curve better.")
    else:
        print("\n🤝 Draw / ReLU Wins.")

if __name__ == "__main__":
    main()