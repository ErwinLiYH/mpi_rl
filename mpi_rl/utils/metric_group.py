from matplotlib import pyplot as plt

class MetricGroup:
    def __init__(self, X_name, Y_names):
        self.X_name = X_name
        self.Y_names = Y_names

        self.X = []
        self.Y = {name: [] for name in Y_names}

    def add(self, X, **Y):
        # check is all Y provided
        assert set(Y.keys()) == set(self.Y_names), "Not all Y provided."
        # add data
        self.X.append(X)
        for name in self.Y_names:
            self.Y[name].append(Y[name])

    def plot(self, Y_name):
        plt.plot(self.X, self.Y[Y_name])
        plt.xlabel(self.X_name)
        plt.ylabel(Y_name)
        plt.show()

class MetricGroups(dict):
    # A dict in structure of {group_name: MetricGroup}
    
    # init as a empty dict
    def __init__(self):
        super().__init__()

    def add(self, X_name, X, **Y):
        if X_name not in self:
            self[X_name] = MetricGroup(X_name=X_name, Y_names=Y.keys())
        self[X_name].add(X, **Y)