from brain import BaseGeneticBrain
from constants import MODEL_INPUTS
import numpy as np
import os


class FlappyBrain(BaseGeneticBrain):

    def __init__(self, driver_id: int, epsilon: float):
        super().__init__(f"Flappy Brain {driver_id}")
        # weight initialization range of [-1.5, 1.5)
        self.w1 = 3 * np.random.random((MODEL_INPUTS, 1)) - 1.5
        self.driver_id = str(driver_id)
        self.epsilon = epsilon

    def forward(self, input_arr: np.array):
        # input shape is (1, num_input)
        pass1 = np.matmul(input_arr, self.w1)  # pass1 has shape of (1, 32)
        # sigmoid activation function with clipping to prevent overflow
        pass1 = np.clip(pass1, -500, 500)  # clip values to prevent exp overflow
        return 1 / (1 + np.exp(-pass1))

    def step(self, inputs: np.array, keys_pressed: list[bool]) -> bool:
        """
        - Given input from the simulation make a decision
        :param inputs: sensor input as a numpy array
        :return direction: int [0 - num_outputs)
        """
        return self.forward(inputs) > 0.5

    def cross_over_mutation(self, other, total_batch_size):
        """
        1. cross over self and other
        2. generate {total_batch_size - 2 (self and other)} number of mutations
        3. return the list of drivers as a numpy array, with the first two being the parents
        :param other:
        :return:
        """
        child_driver = self.cross_over(other)
        mutations = child_driver.mutate(total_batch_size - 2)
        return np.array([self, other] + mutations)

    def cross_over(self, other):
        """
        Single point crossover
        :param other:
        :return:
        """
        child_driver = FlappyBrain(
            "({} & {})".format(self.driver_id, other.driver_id), self.epsilon
        )
        cross_over_point = np.random.randint(self.w1.shape[1])
        child_driver.w1 = np.hstack(
            (self.w1[:, :cross_over_point], other.w1[:, cross_over_point:])
        )
        return child_driver

    def mutate(self, num_mutations):
        """
        :param num_mutations: the number of mutations to create
        :return: a list of FlappyBrain mutations
        """
        mutations = []
        for i in range(num_mutations):
            mutation = FlappyBrain(
                driver_id="{}_{}".format(self.driver_id, i), epsilon=self.epsilon
            )
            mutation_arr = self.generate_mutation_arr()  # generate mutations and masks
            # read following as add mutations to self.w1 at the positions where the mask is less than epsilon
            mutation.w1 = self.w1 + mutation_arr[0][0] * (
                mutation_arr[0][1] < self.epsilon
            )
            mutations.append(mutation)
        return mutations

    def generate_mutation_arr(self) -> list[tuple[np.array, np.array]]:
        """
        :return: list([(mutation array, mask array), ...])
        """
        # since it's only one layer, the mutation array and mask array are the same
        return [
            (np.random.random(self.w1.shape) - 0.5, np.random.random(self.w1.shape))
        ]

    @staticmethod
    def generate_brains(
        num_brains: int, epsilon: float, load_latest: bool = False
    ) -> np.array:
        """
        create a list of random FlappyBrains
        :param load_latest:
        :param epsilon:
        :param num_brains:
        :return:
        """
        if load_latest:
            parent = FlappyBrain(driver_id=0, epsilon=epsilon)
            parent.load_model(os.path.join("assets", "models"))
            mutations = parent.mutate(num_brains - 1)
            return np.array([parent] + mutations)
        return np.array(
            [
                FlappyBrain(driver_id=identifier, epsilon=epsilon)
                for identifier in range(num_brains)
            ]
        )

    def save_model(self, path):
        """
        - Save the brain of the agent to some file (or don't)
        :param path: the path to the model
        :return: None
        """
        path_name = os.path.join(path, "latest_genetic")
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        np.save(os.path.join(path_name, "w1"), self.w1)

    def load_model(self, path):
        """
        - Load the brain of the agent from some file (or don't)
        :param path: the path to the model
        :return: None
        """
        path_name = os.path.join(path, "latest_genetic")
        self.w1 = np.load(os.path.join(path_name, "w1.npy"))
