# SkipReduce Simulation

This repository contains a simplified Python simulation of a Ring AllReduce communication algorithm across multiple GPUs. This specific version focuses exclusively on a static configuration.

## Features

- **Static Execution**: Runs the Ring AllReduce process with fixed skip and shift values, reducing the total number of communication steps based on the configured skip count.
- **Latency Tracking**: Calculates step-by-step latency by evaluating the maximum GPU compute latency and network link latency occurring in parallel during each step.
- **State Monitoring**: Tracks and optionally prints the state of data chunks residing on each GPU throughout the simulation.

## Requirements

- Python 3.x
- Standard Python libraries only. No external dependencies are required.

## Usage

1. Open the Python script and locate the `if __name__ == "__main__":` block at the bottom.
2. Modify the configuration variables to match your desired simulation environment.
3. Run the script from your terminal:


```bash
python script.py
```

## Configuration Parameters

- **num_gpu**: The total number of GPUs in the simulated ring.
- **gpu_latency**: A list containing the computation latency penalty for each respective GPU.
- **link_latency**: A list containing the network transfer latency penalty for each respective GPU sending data to its neighbor.
- **skip**: The number of steps to truncate from the end of the standard Ring AllReduce sequence.
- **shift**: The index shift applied to determine which initial data chunk each GPU sends first.
- **print_steps**: A boolean flag. Set to `True` to output a detailed trace of the simulation, or `False` to only see the final state and total latency.