import argparse
import random
import heapq
import itertools

class GPU:
    def __init__(self, name, num_gpu):
        self.name = name
        self.data = [name + str(i) for i in range(num_gpu)]
        self.buffer = [None] * num_gpu

class StaticSR:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, shift, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip
        self.shift = shift
        self.print_steps = print_steps
        # GPU as characters 'a' to 'z' (limited to 26 gpu)
        self.gpu = [GPU(chr(ord('a') + i), num_gpu) for i in range(num_gpu)]

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}.")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip steps: {self.skip}")
        print(f"Shift value: {self.shift}")
        
        # Ring AllReduce logic
        STEPS = self.num_gpu - 1
        total_latency = 0
        for step in range(STEPS - self.skip):
            per_step_gpu_latencies = []
            per_step_link_latencies = []
            if self.print_steps:
                print(f"\n--- Step {step + 1} ---")
            for i in range(self.num_gpu):
                # Shift index
                sender = self.gpu[i]
                receiver = self.gpu[(i + 1) % self.num_gpu]
                # Simulate sending data
                data_index = (i + self.shift - step) % self.num_gpu
                data_to_send = sender.data[data_index]
                sender.data[data_index] = None
                receiver.data[data_index] += data_to_send
                # Record latencies
                per_step_gpu_latencies.append(self.gpu_latency[i])
                per_step_link_latencies.append(self.link_latency[i])
                if self.print_steps:
                    print(f"GPU {sender.name} sends {data_to_send} to GPU {receiver.name}")

            per_step_gpu_latency = max(per_step_gpu_latencies)
            per_step_link_latency = max(per_step_link_latencies)
            total_latency += per_step_gpu_latency + per_step_link_latency
            if self.print_steps:
                print(f"Step {step + 1} latency: {per_step_gpu_latency} + {per_step_link_latency}")

        # Final data state
        print("\nFinal data at each GPU:")
        for gpu in self.gpu:
            print(f"GPU {gpu.name}: {gpu.data}")

        # Total latency
        print(f"Total latency: {total_latency}")

class RandSRAp1:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip
        self.print_steps = print_steps
        # GPU as characters 'a' to 'z' (limited to 26 GPUs)
        self.gpu = [GPU(chr(ord('a') + i), num_gpu) for i in range(num_gpu)]

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}.")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip steps: {self.skip}")
        
        # Ring AllReduce logic
        STEPS = self.num_gpu - 1
        total_latency = 0
        skip_indices = [[1], [1], [0], [1]]
        # skip_indices = [random.sample(range(self.num_gpu), self.skip) for _ in range(self.num_gpu)]
        print(f"Skip indices per GPU: {skip_indices}")
        for step in range(STEPS):
            per_step_gpu_latencies = []
            per_step_link_latencies = []
            if self.print_steps:
                print(f"\n--- Step {step + 1} ---")
            for i in range(self.num_gpu):
                # Shift index
                sender = self.gpu[i]
                receiver = self.gpu[(i + 1) % self.num_gpu]
                # Simulate sending data
                data_index = (i - step) % self.num_gpu
                if data_index in skip_indices[i]:
                    continue # GPU rests this step
                data_to_send = sender.data[data_index]
                receiver.buffer[data_index] = data_to_send
                # Record latencies
                per_step_gpu_latencies.append(self.gpu_latency[i])
                per_step_link_latencies.append(self.link_latency[i])
                if self.print_steps:
                    print(f"GPU {sender.name} sends {data_to_send} to GPU {receiver.name}")
            
            # After all sends, update receiver data from buffer
            for i in range(self.num_gpu):
                receiver = self.gpu[i]
                for j in range(self.num_gpu):
                    if receiver.buffer[j] is not None:
                        receiver.data[j] += receiver.buffer[j]
                        receiver.buffer[j] = None

            per_step_gpu_latency = max(per_step_gpu_latencies)
            per_step_link_latency = max(per_step_link_latencies)
            total_latency += per_step_gpu_latency + per_step_link_latency
            if self.print_steps:
                print(f"Step {step + 1} latency: {per_step_gpu_latency} + {per_step_link_latency}")
                for gpu in self.gpu:
                    print(f"GPU {gpu.name}: {gpu.data}")

        # Final data state
        if not self.print_steps:
            print("\nFinal data at each GPU:")
            for gpu in self.gpu:
                print(f"GPU {gpu.name}: {gpu.data}")

        # Total latency
        print(f"\nTotal latency: {total_latency}")

class RandSRAp2:
    def __init__(self, num_gpu, gpu_latency, link_latency, skip, print_steps):
        self.num_gpu = num_gpu
        self.gpu_latency = gpu_latency
        self.link_latency = link_latency
        self.skip = skip
        self.print_steps = print_steps
        # GPU as characters 'a' to 'z' (limited to 26 GPUs)
        self.gpu = [GPU(chr(ord('a') + i), num_gpu) for i in range(num_gpu)]

    def simulate(self):
        print(f"Number of GPUs: {self.num_gpu}.")
        print(f"GPU latencies: {self.gpu_latency}")
        print(f"Link latencies: {self.link_latency}")
        print(f"Skip steps: {self.skip}")
        
        # Ring AllReduce logic
        STEPS = self.num_gpu - 1
        total_latency = 0
        skip_indices = [[1], [1], [0], [1]]
        # skip_indices = [random.sample(range(self.num_gpu), self.skip) for _ in range(self.num_gpu)]
        print(f"Skip indices per GPU: {skip_indices}")
        for step in range(STEPS - self.skip):
            per_step_gpu_latencies = []
            per_step_link_latencies = []
            if self.print_steps:
                print(f"\n--- Step {step + 1} ---")
            for i in range(self.num_gpu):
                # Shift index
                sender = self.gpu[i]
                receiver = self.gpu[(i + 1) % self.num_gpu]
                # Simulate sending data
                data_index = (i - step) % self.num_gpu
                if data_index in skip_indices[i]:
                    data_index = (data_index + 1) % self.num_gpu  # Skip this index
                data_to_send = sender.data[data_index]
                receiver.buffer[data_index] = data_to_send
                # Record latencies
                per_step_gpu_latencies.append(self.gpu_latency[i])
                per_step_link_latencies.append(self.link_latency[i])
                if self.print_steps:
                    print(f"GPU {sender.name} sends {data_to_send} to GPU {receiver.name}")
            
            # After all sends, update receiver data from buffer
            for i in range(self.num_gpu):
                receiver = self.gpu[i]
                for j in range(self.num_gpu):
                    if receiver.buffer[j] is not None:
                        receiver.data[j] += receiver.buffer[j]
                        receiver.buffer[j] = None

            per_step_latency = max(per_step_gpu_latencies) + max(per_step_link_latencies)
            total_latency += per_step_latency
            if self.print_steps:
                print(f"Step {step + 1} latency: {per_step_latency}")
                for gpu in self.gpu:
                    print(f"GPU {gpu.name}: {gpu.data}")

        # Final data state
        if not self.print_steps:
            print("\nFinal data at each GPU:")
            for gpu in self.gpu:
                print(f"GPU {gpu.name}: {gpu.data}")

        # Total latency
        print(f"\nTotal latency: {total_latency}")


if __name__ == "__main__":
    # Get Ring AllReduce configuration
    parser = argparse.ArgumentParser(description="Ring AllReduce Simulation")
    parser.add_argument("-m", "--mode", help="operation mode", choices=["static", "random1", "random2", "exhaustive"], default="static")
    parser.add_argument("-n", "--num-gpu", help="number of GPUs", default=5, type=int)
    parser.add_argument("--gpu-latency", help="comma-separated GPU latencies", default=None)
    parser.add_argument("--link-latency", help="comma-separated link latencies", default=None)
    parser.add_argument("--skip", help="number of skip steps", default=0, type=int)
    parser.add_argument("--shift", help="shift value for starting skip index", default=0, type=int)
    parser.add_argument("--print-steps", help="print each step", action="store_true")
    
    # Process configuration
    num_gpu = parser.parse_args().num_gpu
    gpu_latency = list(map(int, parser.parse_args().gpu_latency.split(','))) if parser.parse_args().gpu_latency else [i + 1 for i in range(num_gpu)]
    link_latency = list(map(int, parser.parse_args().link_latency.split(','))) if parser.parse_args().link_latency else [i + num_gpu for i in gpu_latency]
    if len(gpu_latency) != num_gpu or len(link_latency) != num_gpu:
        raise ValueError("Length of gpu-latency and link-latency must match num-gpu")
    skip = parser.parse_args().skip
    shift = parser.parse_args().shift
    print_steps = parser.parse_args().print_steps
    mode = parser.parse_args().mode
    
    # Run simulation
    if mode == "static":
        StaticSR(num_gpu, gpu_latency, link_latency, skip, shift, print_steps).simulate()
    elif mode == "random1":
        RandSRAp1(num_gpu, gpu_latency, link_latency, skip, print_steps).simulate()
    elif mode == "random2":
        RandSRAp2(num_gpu, gpu_latency, link_latency, skip, print_steps).simulate()