class ObjectiveFunction(object):
    def __init__(self):
        raise NotImplemented

    def __call__(self, x):
        raise NotImplemented


class DiscreteObjectiveFunction(ObjectiveFunction):
    def __init__(self, y_values):
        self.x_values = range(len(y_values))
        self.y_values = y_values

    def __call__(self, x_idx):
        return self.y_values[x_idx]


class ContinuousObjectiveFunction(ObjectiveFunction):
    def __init__(self):
        pass

    def __call__(self):
        pass

# TODO - make discrete and continuous functions that inherits this
#     - then create a domain-specific classes that inherits 
#     - either discrete/continuous class
