from abc import ABC, abstractmethod
from random import random, seed
from typing import Callable, List
from scipy.stats import beta
import time


class MHSimulator(ABC):
    def __init__(self, n: int, target_density_func: Callable[[float], float]):
        self.N = n
        self.target_density = target_density_func
        seed()
        self.current_x = random()

    @abstractmethod
    def proposal_func(self) -> float:
        pass

    def runner(self) -> List[float]:
        results = []
        for _ in range(self.N):
            proposal_x = self.proposal_func()
            acceptance_ratio = self.target_density(proposal_x) / self.target_density(self.current_x)
            if random() < acceptance_ratio:
                self.current_x = proposal_x
            results.append(self.current_x)
        return results


class GenericMHSimulator(MHSimulator):
    def proposal_func(self) -> float:
        return self.current_x + 0.5 * (random() - 0.5)


class IndependentMHSimulator(MHSimulator):
    def proposal_func(self) -> float:
        return 0.5 * (random() - 0.5)


class RandomWalkMHSimulator(MHSimulator):
    def proposal_func(self) -> float:
        return self.current_x + (random() - 0.5)


class MHCreator(ABC):
    def __init__(self, n: int, target_density_func: Callable[[float], float]):
        self.N = n
        self.target_density = target_density_func

    @abstractmethod
    def create_simulator(self) -> MHSimulator:
        pass

    def operator_runner(self) -> List[float]:
        simulator = self.create_simulator()
        return simulator.runner()


class GenericMHSimulatorCreator(MHCreator):
    def create_simulator(self) -> MHSimulator:
        return GenericMHSimulator(self.N, self.target_density)


class IndependentMHSimulatorCreator(MHCreator):
    def create_simulator(self) -> MHSimulator:
        return IndependentMHSimulator(self.N, self.target_density)


class RandomWalkMHSimulatorCreator(MHCreator):
    def create_simulator(self) -> MHSimulator:
        return RandomWalkMHSimulator(self.N, self.target_density)


def client_code(creator: MHCreator):
    results = creator.operator_runner()
    for value in results:
        print(value)


if __name__ == "__main__":
    def target_density(x: float) -> float:
        a, b = 2.6, 6.3
        return beta.pdf(x, a,b)

    creator = GenericMHSimulatorCreator(100000, target_density)
    start_time = time.time()
    client_code(creator)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time) # 11.69 seconds at 100k, generic 
