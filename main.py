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
                # In real memory, this data is retained until overwritten, but for simulation this is for clarity
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
    StaticMode(num_gpu, gpu_latency, link_latency, skip, shift, print_steps).simulate()