import matplotlib.pyplot as plt

# Data from the benchmark results
log_numbers = [1, 2, 3, 4, 5]
error_type = [0.1429, 0.1871, 0.2000, 0.3333, 0.0667]
severity = [0.0476, 0.0065, 0.0000, 0.2000, 0.0000]
description = [0.1291, 0.1257, 0.2153, 0.1496, 0.1295]
solution = [0.1588, 0.1810, 0.2382, 0.1858, 0.1668]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(log_numbers, error_type, marker='o', linestyle='-', color='blue', label='Error Type Loss')
plt.plot(log_numbers, severity, marker='s', linestyle='-', color='red', label='Severity Loss')
plt.plot(log_numbers, description, marker='^', linestyle='-', color='green', label='Description Loss')
plt.plot(log_numbers, solution, marker='d', linestyle='-', color='purple', label='Solution Loss')

# Labels and Title
plt.xlabel('Log Number')
plt.ylabel('Loss')
plt.title('Benchmark Loss Analysis')
plt.legend()
plt.grid(True)

# Show plot
plt.show()


import matplotlib.pyplot as plt

# Data from the benchmark results
log_numbers = [1, 2, 3]
error_type = [0.6283, 0.5750, 0.6279]
severity = [0.6809, 0.5000, 0.5349]
description = [0.2209, 0.1654, 0.1832]
solution = [0.3161, 0.2835, 0.3274]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(log_numbers, error_type, marker='o', linestyle='-', color='blue', label='Error Type Loss')
plt.plot(log_numbers, severity, marker='s', linestyle='-', color='red', label='Severity Loss')
plt.plot(log_numbers, description, marker='^', linestyle='-', color='green', label='Description Loss')
plt.plot(log_numbers, solution, marker='d', linestyle='-', color='purple', label='Solution Loss')

# Labels and Title
plt.xlabel('Log Number')
plt.ylabel('Loss')
plt.title('Benchmark Loss Analysis')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
