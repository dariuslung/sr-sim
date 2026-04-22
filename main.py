import random
import itertools

SIGCONSEC = -1

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
                    print(f"  GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)}")

            step_gpu_latency = max(step_gpu_latencies)
            step_link_latency = max(step_link_latencies)
            total_latency += step_gpu_latency + step_link_latency
            if self.print_steps:
                for gpu in self.gpu:
                    print(f"  GPU {gpu.rank}: {gpu.data}")
                print(f"  Step Latency: GPU {step_gpu_latency} + Link {step_link_latency}")

        # Final data state
        if not self.print_steps:
            print("\nFinal data at each GPU:")
            for gpu in self.gpu:
                print(f"GPU {gpu.rank}: {gpu.data}")

        # Total latency
        print(f"\nTotal latency: {total_latency}")

# Random Mode
# Each GPU randomly skips 'skip' chunks during the N-1 steps.
# Skipped GPU is idle during that step.
class RandomMode:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, print_steps, fixed_skip_indices=None):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip
        self.print_steps = print_steps
        self.fixed_skip_indices = fixed_skip_indices # Optional override
        self.gpu = [GPU(i, num_gpu) for i in range(num_gpu)]

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip chunks per GPU: {self.skip}")
        
        # Variables
        STEPS = self.num_gpu - 1
        total_latency = 0

        # Use fixed indices if provided, otherwise generate random ones
        if self.fixed_skip_indices:
            skip_indices = self.fixed_skip_indices
        else:
            skip_indices = [random.sample(range(self.num_gpu), self.skip) for _ in range(self.num_gpu)]
        # skip_indices = [[0], [1], [2], [3]]
        # skip_indices = [[3], [1], [2], [3]]
        # skip_indices = [[0], [1, 3], [2], [2, 3]]
        print(f"Skip indices per GPU: {skip_indices}")

        # Simulation Loop
        for step in range(STEPS):
            step_gpu_latencies = []
            step_link_latencies = []
            
            if self.print_steps:
                print(f"\n--- Step {step + 1} ---")
            
            # --- SENDER LOGIC ---
            for i in range(self.num_gpu):
                sender = self.gpu[i]
                receiver = self.gpu[(i + 1) % self.num_gpu]
                
                # Identify the chunk traveling on this link
                chunk_idx = (i - step) % self.num_gpu
                
                if step == 0 and chunk_idx in skip_indices[i]:
                    data_to_send = SIGCONSEC
                else:
                    data_to_send = sender.data[chunk_idx]
                receiver.buffer[chunk_idx] = data_to_send
                sender.data[chunk_idx] = None

                # --- LATENCY LOGIC ---
                if chunk_idx in skip_indices[receiver.rank]:
                    # Receive 'empty' signal
                    if receiver.buffer[chunk_idx] == SIGCONSEC:
                        step_gpu_latencies.append(0)
                        step_link_latencies.append(0)
                        if self.print_steps:
                            print(f"  GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)} ({receiver.buffer[chunk_idx]}) (Consecutive signal)")
                    # Skipped Reduction: Pay Link Latency, but 0 Compute Latency
                    else:
                        step_gpu_latencies.append(0)
                        step_link_latencies.append(self.link_latency[sender.rank])
                        if self.print_steps:
                            print(f"  GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)} ({receiver.buffer[chunk_idx]}) (No reduction)")
                else:
                    # Receive 'empty' signal
                    if receiver.buffer[chunk_idx] == SIGCONSEC:
                        step_gpu_latencies.append(0)
                        step_link_latencies.append(0)
                        if self.print_steps:
                            print(f"  GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)} ({receiver.buffer[chunk_idx]}) (No reduction)")

                    # Normal Operation: Pay Link + Compute Latency
                    else:
                        step_gpu_latencies.append(self.gpu_latency[receiver.rank])
                        step_link_latencies.append(self.link_latency[sender.rank])
                        if self.print_steps:
                            print(f"  GPU {sender.rank} -> GPU {receiver.rank} : Chunk {chr(ord('a') + chunk_idx)} ({receiver.buffer[chunk_idx]})")

                # --- RECEIVER LOGIC ---
                if chunk_idx in skip_indices[receiver.rank]:
                    # Skip: Overwrite with incoming data (ignore local contribution)
                    receiver.data[chunk_idx] = receiver.buffer[chunk_idx]
                else:
                    # Normal: Reduce (accumulate)
                    # If receiver buffer is None, meaning every GPU skipped this index in previous steps, can also skip reduction
                    if receiver.buffer[chunk_idx] != SIGCONSEC:
                        receiver.data[chunk_idx] += receiver.buffer[chunk_idx]
                receiver.buffer[chunk_idx] = None
            
            # Record latencies
            step_gpu_latency = max(step_gpu_latencies)
            step_link_latency = max(step_link_latencies)
            total_latency += step_gpu_latency + step_link_latency
            if self.print_steps:
                for gpu in self.gpu:
                    print(f"  GPU {gpu.rank}: {gpu.data}")
                print(f"  Step Latency: GPU {step_gpu_latency} + Link {step_link_latency}")

        # Final data state
        if not self.print_steps:
            print("\nFinal data at each GPU:")
            for gpu in self.gpu:
                print(f"GPU {gpu.rank}: {gpu.data}")

        # Total latency
        print(f"\nTotal latency: {total_latency}")

# Exhaustive Mode
# Each GPU can skip 'max_skip' chunks during the N-1 steps.
# Each skipped chunk incurs a penalty based on its importance weight.
# Exhaustive search to find the optimal skip configuration minimizing total latency + penalty.
class ExhaustiveMode:
    def __init__(self, num_gpu, gpu_latency, link_latency, max_skip, importance_weights, penalty_factor):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.max_skip = max_skip        # Max chunks ANY single GPU is allowed to skip
        self.weights = importance_weights # List of [Weight_Chunk0, Weight_Chunk1...]
        self.penalty = penalty_factor   # How much latency is 1 unit of "Importance" worth?

    def solve(self):
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
        print(f"Optimal Total Score: {best_score} (Latency: {min_latency} + Penalty: {min_penalty * self.penalty})")
        
        return best_config

    def evaluate_configuration(self, skip_config):
        steps = self.num_gpu - 1
        total_latency = 0
        total_penalty = 0

        # Calculate Penalty ONCE
        for gpu_id in range(self.num_gpu):
            for chunk_idx in skip_config[gpu_id]:
                total_penalty += self.weights[chunk_idx]

        def is_skipped(gpu_id, chunk_idx):
            return chunk_idx in skip_config[gpu_id]

        for step in range(steps):
            step_gpu_latencies = []
            step_link_latencies = []
            
            for sender_idx in range(self.num_gpu):
                # Identify chunk and receiver
                chunk_idx = (sender_idx - step) % self.num_gpu
                receiver_idx = (sender_idx + 1) % self.num_gpu
                
                # Check if the data circulating is "Zero/None"
                # This happens if the chunk's originator (Rank = chunk_idx) skipped it.
                # WRONGGGGGGGGGGGGGGG
                origin_skipped = is_skipped(chunk_idx, chunk_idx)

                # --- LOGIC BRANCHES ---
                
                # 1. Step 0 Sender Skip
                if step == 0 and is_skipped(sender_idx, chunk_idx):
                    step_gpu_latencies.append(0)
                    step_link_latencies.append(self.link_latency[sender_idx])

                # 2. Link Active (Step 0 Active OR Step > 0 Relay)
                else:
                    # GPU Compute Logic
                    if is_skipped(receiver_idx, chunk_idx):
                        # Case A: Receiver Skips -> Overwrite (Free)
                        step_gpu_latencies.append(0)
                    
                    elif origin_skipped:
                        # Case B: Receiver Active, but Data is None -> Add 0 (Free)
                        # This fixes the missing logic in your snippet
                        step_gpu_latencies.append(0)
                    
                    else:
                        # Case C: Receiver Active, Data Valid -> Reduce (Paid)
                        # NOTE: Use receiver_idx for GPU latency, not sender_idx
                        step_gpu_latencies.append(self.gpu_latency[receiver_idx])

                # Link Latency is always paid (Signal/Header overhead)
                step_link_latencies.append(self.link_latency[sender_idx])
                
            # Parallel Latency Calculation (Max of all parallel ops)
            if step_gpu_latencies and step_link_latencies:
                total_latency += max(step_gpu_latencies) + max(step_link_latencies)
            
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
    mode = "random"
    # mode = "exhaustive"

    if mode == "static":
        print("=== Static Mode ===")
        StaticMode(num_gpu, gpu_latency, link_latency, skip, shift, print_steps).simulate()
    elif mode == "random":
        print("=== Random Mode ===")
        RandomMode(num_gpu, gpu_latency, link_latency, skip, print_steps).simulate()
    elif mode == "exhaustive":
        importance_weights = [1, 1, 1, 1]
        penalty_factor = 1
        max_skip = num_gpu
        if len(importance_weights) != num_gpu:
            raise ValueError("Length of importance_weights must match num_gpu")
        print("=== Exhaustive Mode ===")
        best_config = ExhaustiveMode(num_gpu, gpu_latency, link_latency, max_skip, importance_weights, penalty_factor).solve()
        RandomMode(num_gpu, gpu_latency, link_latency, 0, print_steps, fixed_skip_indices=best_config).simulate()