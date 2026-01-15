import random
import itertools

class GPU:
    def __init__(self, rank, num_gpu):
        self.rank = rank
        self.data = [str(rank) + chr(ord('a') + i) for i in range(num_gpu)]
        self.buffer = [None] * num_gpu

# Static Mode
# Fixed skip and shift values, skips in strides and completes in N-1-S steps.
class StaticMode:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, shift, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip
        self.shift = shift
        self.print_steps = print_steps
        self.gpu = [GPU(i, num_gpu) for i in range(num_gpu)]

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip steps: {self.skip}")
        print(f"Shift value: {self.shift}")
        
        # Ring AllReduce logic
        STEPS = self.num_gpu - 1
        total_latency = 0
        for step in range(STEPS - self.skip):
            step_gpu_latencies = []
            step_link_latencies = []
            if self.print_steps:
                print(f"\n--- Step {step + 1} ---")
            for i in range(self.num_gpu):
                # Shift index
                sender = self.gpu[i]
                receiver = self.gpu[(i + 1) % self.num_gpu]
                # Simulate sending data
                chunk_idx = (i + self.shift - step) % self.num_gpu
                data_to_send = sender.data[chunk_idx]
                sender.data[chunk_idx] = None
                receiver.data[chunk_idx] += data_to_send
                # Record latencies
                step_gpu_latencies.append(self.gpu_latency[i])
                step_link_latencies.append(self.link_latency[i])
                if self.print_steps:
                    print(f"GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)}")

            step_gpu_latency = max(step_gpu_latencies)
            step_link_latency = max(step_link_latencies)
            total_latency += step_gpu_latency + step_link_latency
            if self.print_steps:
                for gpu in self.gpu:
                    print(f"GPU {gpu.rank}: {gpu.data}")
                print(f"Step {step + 1} latency: {step_gpu_latency} + {step_link_latency}")

        # Final data state
        if not self.print_steps:
            print("\nFinal data at each GPU:")
            for gpu in self.gpu:
                print(f"GPU {gpu.rank}: {gpu.data}")

        # Total latency
        print(f"\nTotal latency: {total_latency}")

# Random Mode 1
# Each GPU randomly skips 'skip' chunks during the N-1 steps.
# Skipped GPU is idle during that step.
class RandomMode1:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip
        self.print_steps = print_steps
        self.gpu = [GPU(i, num_gpu) for i in range(num_gpu)]

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip chunks per GPU: {self.skip}")
        
        # Ring AllReduce logic
        STEPS = self.num_gpu - 1
        total_latency = 0
        # Randomly select skip indices for each GPU
        skip_indices = [random.sample(range(self.num_gpu), self.skip) for _ in range(self.num_gpu)]
        skip_indices = [[1], [1], [0], [1]]
        print(f"Skip indices per GPU: {skip_indices}")
        for step in range(STEPS):
            step_gpu_latencies = []
            step_link_latencies = []
            if self.print_steps:
                print(f"\n--- Step {step + 1} ---")
            for i in range(self.num_gpu):
                # Shift index
                sender = self.gpu[i]
                receiver = self.gpu[(i + 1) % self.num_gpu]
                # Simulate sending data
                chunk_idx = (i - step) % self.num_gpu
                if chunk_idx in skip_indices[i]:
                    if self.print_steps:
                        print(f"GPU {sender.rank} -> IDLE (Skipping Chunk {chr(ord('a') + chunk_idx)})")
                    continue # GPU idle during this step
                data_to_send = sender.data[chunk_idx]
                receiver.buffer[chunk_idx] = data_to_send
                # Record latencies
                step_gpu_latencies.append(self.gpu_latency[i])
                step_link_latencies.append(self.link_latency[i])
                if self.print_steps:
                    print(f"GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)}")
            
            # After all sends, update receiver data from buffer
            for i in range(self.num_gpu):
                receiver = self.gpu[i]
                for j in range(self.num_gpu):
                    if receiver.buffer[j] is not None:
                        receiver.data[j] += receiver.buffer[j]
                        receiver.buffer[j] = None

            step_gpu_latency = max(step_gpu_latencies)
            step_link_latency = max(step_link_latencies)
            total_latency += step_gpu_latency + step_link_latency
            if self.print_steps:
                for gpu in self.gpu:
                    print(f"GPU {gpu.rank}: {gpu.data}")
                print(f"Step {step + 1} latency: {step_gpu_latency} + {step_link_latency}")

        # Final data state
        if not self.print_steps:
            print("\nFinal data at each GPU:")
            for gpu in self.gpu:
                print(f"GPU {gpu.rank}: {gpu.data}")

        # Total latency
        print(f"\nTotal latency: {total_latency}")

# Random Mode 2
# Each GPU randomly skips 'skip' chunks during the N-1-S steps.
# Skipped GPU still participates by incrementing chunk index.
class RandomMode2:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip
        self.print_steps = print_steps
        self.gpu = [GPU(i, num_gpu) for i in range(num_gpu)]

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip chunks per GPU: {self.skip}")
        
        # Ring AllReduce logic
        STEPS = self.num_gpu - 1
        total_latency = 0
        # Randomly select skip indices for each GPU
        skip_indices = [random.sample(range(self.num_gpu), self.skip) for _ in range(self.num_gpu)]
        skip_indices = [[1], [1], [0], [1]]
        print(f"Skip indices per GPU: {skip_indices}")
        for step in range(STEPS - self.skip):
            step_gpu_latencies = []
            step_link_latencies = []
            if self.print_steps:
                print(f"\n--- Step {step + 1} ---")
            for i in range(self.num_gpu):
                # Shift index
                sender = self.gpu[i]
                receiver = self.gpu[(i + 1) % self.num_gpu]
                # Simulate sending data
                chunk_idx = (i - step) % self.num_gpu
                if chunk_idx in skip_indices[i]:
                    chunk_idx = (chunk_idx + 1) % self.num_gpu  # Skip this index
                data_to_send = sender.data[chunk_idx]
                receiver.buffer[chunk_idx] = data_to_send
                # Record latencies
                step_gpu_latencies.append(self.gpu_latency[i])
                step_link_latencies.append(self.link_latency[i])
                if self.print_steps:
                    print(f"GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)}")
            
            # After all sends, update receiver data from buffer
            for i in range(self.num_gpu):
                receiver = self.gpu[i]
                for j in range(self.num_gpu):
                    if receiver.buffer[j] is not None:
                        receiver.data[j] += receiver.buffer[j]
                        receiver.buffer[j] = None

            step_gpu_latency = max(step_gpu_latencies)
            step_link_latency = max(step_link_latencies)
            total_latency += step_gpu_latency + step_link_latency
            if self.print_steps:
                for gpu in self.gpu:
                    print(f"GPU {gpu.rank}: {gpu.data}")
                print(f"Step {step + 1} latency: {step_gpu_latency} + {step_link_latency}")

        # Final data state
        if not self.print_steps:
            print("\nFinal data at each GPU:")
            for gpu in self.gpu:
                print(f"GPU {gpu.rank}: {gpu.data}")

        # Total latency
        print(f"\nTotal latency: {total_latency}")

# Exhaustive Mode 1
# Each GPU can skip 'skip' chunks during the N-1 steps, skipped GPU is idle during that step.
# Exhaustive search to find the optimal skip configuration minimizing total latency.
class ExhaustiveMode:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip  # Number of chunks to skip per GPU
        self.print_steps = print_steps

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip chunks per GPU: {self.skip}")

        # 1. Generate all valid skip combinations for a SINGLE GPU
        #    Example: If N=4, skip=1 -> [(0), (1), (2), (3)]
        single_gpu_options = list(itertools.combinations(range(self.num_gpu), self.skip))
        
        # 2. Cartesian Product: Assign one option to each GPU
        #    This creates the "Global Skip Configuration"
        #    WARNING: Search space is (N choose K)^N. For N=4, K=1 -> 4^4 = 256 configs.
        all_configurations = itertools.product(single_gpu_options, repeat=self.num_gpu)
        
        best_latency = float('inf')
        best_config = None

        # 3. Iterate through every possible skip configuration
        for config in all_configurations:
            # config is a tuple of tuples: ((skipped_chunks_gpu_a), (skipped_chunks_gpu_b), ...)
            latency = self.evaluate_configuration(config)
            
            if latency < best_latency:
                best_latency = latency
                best_config = config
        
        # 4. Report Results
        print(f"Skip indices per GPU: {best_config}")
        
        # Re-run the best one with printing enabled to show the trace
        if self.print_steps:
            print("\n--- Simulation Trace of Best Solution ---")
            self.evaluate_configuration(best_config, debug=True)

        print(f"\nMinimum Total Latency: {best_latency}")

    def evaluate_configuration(self, skip_config, debug=False):
        # The Fixed Ring AllReduce Schedule (N-1 Steps)
        # Based on Standard Ring: Each step, GPU i sends chunk (i - step)
        
        steps = self.num_gpu - 1
        total_latency = 0
        
        # Helper to check if a specific chunk is skipped by a specific GPU
        # skip_config[gpu_id] contains a tuple of chunk_indices to skip
        def is_skipped(gpu_id, chunk_idx):
            return chunk_idx in skip_config[gpu_id]

        for step in range(steps):
            step_gpu_latencies = []
            step_link_latencies = []
            
            if debug: print(f"\n--- Step {step + 1} ---")

            # Determine activity for all GPUs in this step
            for sender_idx in range(self.num_gpu):
                # Standard Ring Logic: What chunk *should* I send now?
                # Formula: (sender_rank - step) % N
                chunk_idx = (sender_idx - step) % self.num_gpu
                
                # CHECK: Is this chunk in the skip list for this Sender?
                if is_skipped(sender_idx, chunk_idx):
                    # ACTION: Sleep / Send None
                    if debug: print(f"  GPU {sender_idx} -> IDLE (Skipping Chunk {chr(ord('a') + chunk_idx)})")
                    step_gpu_latencies.append(0)   # No compute cost
                    step_link_latencies.append(0)  # No link cost
                else:
                    # ACTION: Send Data
                    receiver_idx = (sender_idx + 1) % self.num_gpu
                    if debug: print(f"  GPU {sender_idx} -> GPU {receiver_idx} : Chunk {chr(ord('a') + chunk_idx)}")
                    step_gpu_latencies.append(self.gpu_latency[sender_idx])
                    step_link_latencies.append(self.link_latency[sender_idx])

            # Calculate Step Latency (Bottleneck Analysis)
            # The step takes as long as the SLOWEST active component
            current_step_latency = 0
            if step_gpu_latencies and step_link_latencies:
                 # Assuming overlap: Max(GPU_time) + Max(Link_Time) 
                 # Or if pipelined: Max(GPU_time + Link_Time). Adjust based on your model.
                 # Using your previous model:
                 current_step_latency = max(step_gpu_latencies) + max(step_link_latencies)
            
            total_latency += current_step_latency
            
        return total_latency

# Exhaustive Mode 2
# Each GPU can skip 'max_skip' chunks during the N-1 steps.
# Each skipped chunk incurs a penalty based on its importance weight.
# Exhaustive search to find the optimal skip configuration minimizing total latency + penalty.
class ExhaustivePenaltyMode:
    def __init__(self, num_gpu, gpu_latency, link_latency, max_skip, importance_weights, penalty_factor, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.max_skip = max_skip        # Max chunks ANY single GPU is allowed to skip
        self.weights = importance_weights # List of [Weight_Chunk0, Weight_Chunk1...]
        self.penalty = penalty_factor   # How much latency is 1 unit of "Importance" worth?
        self.print_steps = print_steps

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Max skip chunks per GPU: {self.max_skip}")
        print(f"Chunk importance weights: {self.weights}")
        print(f"Penalty factor: {self.penalty}")

        # 1. Generate Variable-Length Skip Options per GPU
        #    Example: If max_skip=2, generating options of size 0, 1, and 2.
        single_gpu_options = []
        for r in range(self.max_skip + 1):
            single_gpu_options.extend(list(itertools.combinations(range(self.num_gpu), r)))
        
        # 2. Cartesian Product (Global Configuration)
        #    WARNING: Search space grows extremely fast.
        #    (Sum of combinations)^N
        all_configurations = itertools.product(single_gpu_options, repeat=self.num_gpu)
        
        best_score = float('inf')
        best_result = None # Stores (config, latency, penalty)

        for config in all_configurations:
            latency, accuracy_penalty = self.evaluate_configuration(config)
            
            # THE COST FUNCTION
            total_score = latency + (accuracy_penalty * self.penalty)
            
            if total_score < best_score:
                best_score = total_score
                best_result = (config, latency, accuracy_penalty)

        # 3. Report Results
        best_config, min_latency, min_penalty = best_result
        print(f"Total Score: {best_score} (Latency: {min_latency} + Penalty: {min_penalty * self.penalty})")
        print(f"Skip indices per GPU: {best_config}")
        
        # Re-run the best one with printing enabled to show the trace
        if self.print_steps:
            print("\n--- Simulation Trace of Best Solution ---")
            self.evaluate_configuration(best_config, debug=True)

        print(f"\nMinimum Total Latency: {min_latency}")

    def evaluate_configuration(self, skip_config, debug=False):
        steps = self.num_gpu - 1
        total_latency = 0
        total_penalty = 0

        # Helper to check skips
        def is_skipped(gpu_id, chunk_idx):
            return chunk_idx in skip_config[gpu_id]

        # A. Calculate Latency (Simulation)
        for step in range(steps):
            step_gpu_latencies = []
            step_link_latencies = []

            if debug: print(f"\n--- Step {step + 1} ---")
            
            for sender_idx in range(self.num_gpu):
                chunk_idx = (sender_idx - step) % self.num_gpu
                
                if is_skipped(sender_idx, chunk_idx):
                    if debug: print(f"  GPU {sender_idx} -> IDLE (Skipping Chunk {chr(ord('a') + chunk_idx)})")
                    step_gpu_latencies.append(0)
                    step_link_latencies.append(0)
                    # Accumulate Penalty
                    total_penalty += self.weights[chunk_idx]
                else:
                    receiver_idx = (sender_idx + 1) % self.num_gpu
                    if debug: print(f"  GPU {sender_idx} -> GPU {receiver_idx} : Chunk {chr(ord('a') + chunk_idx)}")
                    step_gpu_latencies.append(self.gpu_latency[sender_idx])
                    step_link_latencies.append(self.link_latency[sender_idx])

            if step_gpu_latencies:
                total_latency += max(step_gpu_latencies) + max(step_link_latencies)

        # Note: Depending on your logic, you might only want to count the penalty 
        # ONCE per chunk globally, or ONCE per skip action. 
        # Currently, this counts it every time a skip action happens (per step).
        
        return total_latency, total_penalty
    

if __name__ == "__main__":
    # Simulation configuration
    num_gpu = 4
    gpu_latency = [1, 2, 3, 4]
    link_latency = [5, 6, 7, 8]
    skip = 1
    shift = 0
    print_steps = True
    
    # Validate inputs
    if len(gpu_latency) != num_gpu or len(link_latency) != num_gpu:
        raise ValueError("Length of gpu_latency and link_latency must match num_gpu")

    # Run simulation
    # mode = "static"
    # mode = "random1"
    mode = "random2"
    # mode = "exhaustive1"
    # mode = "exhaustive2"

    if mode == "static":
        print("=== Static Mode ===")
        StaticMode(num_gpu, gpu_latency, link_latency, skip, shift, print_steps).simulate()
    elif mode == "random1":
        print("=== Random Mode 1 ===")
        RandomMode1(num_gpu, gpu_latency, link_latency, skip, print_steps).simulate()
    elif mode == "random2":
        print("=== Random Mode 2 ===")
        RandomMode2(num_gpu, gpu_latency, link_latency, skip, print_steps).simulate()
    elif mode == "exhaustive1":
        print("=== Exhaustive Mode 1 ===")
        ExhaustiveMode(num_gpu, gpu_latency, link_latency, skip, print_steps).simulate()
    elif mode == "exhaustive2":
        importance_weights = [10, 10, 1, 1]
        penalty_factor = 1
        max_skip = 4
        if len(importance_weights) != num_gpu:
            raise ValueError("Length of importance_weights must match num_gpu")
        print("=== Exhaustive Mode 2 ===")
        ExhaustivePenaltyMode(num_gpu, gpu_latency, link_latency, max_skip, importance_weights, penalty_factor, print_steps).simulate()