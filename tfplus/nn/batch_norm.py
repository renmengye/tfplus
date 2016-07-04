from graph_builder import GraphBuilder


class BatchNorm(GraphBuilder):

    def __init__(self, n_out, phase_train, scope='bn', scope2='bn',
                 affine=True, init_beta=None, init_gamma=None, frozen=False):
        pass

    def init_var(self):
        pass

    def build(self, inp):
        return inp
        pass
    pass
