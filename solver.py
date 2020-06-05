import torch


class Model:
    def __init__(self, calculator, t_0):
        self.calculator = calculator
        self.t = t_0

    def loss(self):
        return self.calculator.dual_func_value(t)

class Solver:
    def __init__(self, calculator, t_0):
        self.model = Model(calculator, t_0)

    def solve(self, t_0=None, num_iters=1000):
        optimizer = optim.SGD(model.t, lr=0.01, momentum=0.9)
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = model.loss()
            loss.backward()
            optimizer.step()
