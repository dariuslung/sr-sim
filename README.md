# Custom SkipReduce Simulation

This repository contains a Python simulation of a Ring AllReduce communication algorithm across multiple GPUs. The simulation explores different strategies for skipping chunk transfers and reductions to evaluate the impact on total latency and accuracy penalties.

## Features

The simulation implements three different operational modes:

- **Static Mode**: Executes the Ring AllReduce with fixed skip and shift values. Skips occur in consistent strides, and the operation completes in a predetermined number of steps.
- **Random Mode**: Each GPU randomly selects a specified number of chunks to skip during the reduction steps. If a chunk is skipped, the GPU remains idle for that computation, bypassing the reduction step.
- **Exhaustive Mode**: Performs an exhaustive search to find the optimal skip configuration. It evaluates all possible skip combinations up to a maximum limit per GPU. The optimal configuration is determined by minimizing a cost function that balances total communication/computation latency against a penalty factor based on the importance weights of the skipped chunks.

## Requirements

- Python 3.x
- Standard library modules only (`random`, `itertools`). No external dependencies are required.

## Usage

1. Open `main.py` and navigate to the `if __name__ == "__main__":` block at the bottom of the file.
2. Configure the simulation parameters such as the number of GPUs (`num_gpu`), latency profiles (`gpu_latency`, `link_latency`), and the desired mode (`mode`).
3. Uncomment the desired mode: `"static"`, `"random"`, or `"exhaustive"`.
4. Run the script from your terminal:


```bash
python main.py
```

### Configuration Parameters

- `num_gpu`: Total number of GPUs in the ring.
- `gpu_latency`: A list representing the computation latency for each GPU.
- `link_latency`: A list representing the network transfer latency from each GPU to its neighbor.
- `skip`: Number of chunks to skip (used in Static and Random modes).
- `importance_weights`: A list defining the penalty weight of each chunk (used in Exhaustive mode).
- `penalty_factor`: A scalar multiplier that converts the importance weight penalty into latency equivalents (used in Exhaustive mode).
- `print_steps`: Set to `True` to output a detailed trace of the simulation step-by-step.