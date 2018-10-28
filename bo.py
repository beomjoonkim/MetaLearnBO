import numpy as np
import helper


class BO(object):
    def __init__(self, model, obj_fcn, acq_fcn, opt_n=1e4):
        self.model = model
        self.evaled_x = []
        self.evaled_y = []
        self.obj_fcn = obj_fcn
        self.acq_fcn = acq_fcn
        self.domain = self.obj_fcn.domain
        self.opt_n = opt_n

    def choose_next_point(self):
        self.model.update(self.evaled_x, self.evaled_y)
        x, acq_fcn_val = helper.global_minimize(self.acq_fcn,
                                                self.acq_fcn.fg,
                                                self.domain,
                                                self.opt_n)
        y = self.obj_fcn(x)
        self.evaled_x.append(x)
        self.evaled_y.append(y)
        return x, y

    def generate_evals(self, T):
        for i in range(T):
            yield self.choose_next_point()
