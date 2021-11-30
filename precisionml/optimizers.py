import numpy as np
import torch


class ConjugateGradients(torch.optim.Optimizer):
    """Implements the conjugate gradients optimization algorithm.
    
    Args:
        params (iterable): iterable of parameters to optimize
        search (bool, optional): whether to compute step size via 
            a line search. If False, compute step size by computing the
            Hessian-direction product (works great if loss is actually
            quadratic)
    
    Example:
        >>> optimizer = ConjugateGradients(model.parameters())
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    TODO: Implement a fancier line search algorithm. Currently I just do
        a grid search over many orders of magnitude and chose the lowest-loss
        step length.
    """
    
    def __init__(self, params, search=False):
        defaults = dict()
        super(ConjugateGradients, self).__init__(params, defaults)
        self.lr_range = [np.power(2.0, n) for n in range(-32, 10)]
        self.prev_grads = None
        self.prev_direction = None
        self.search = search
    
    def _dot_prod(self, vecs1, vecs2):
        """Helper function for computing dot product between two vectors, where
        the vector elements are spread out across a list of tensors.
        
        Args:
            vecs1, vecs2 (list[torch.Tensor]): vectors to take product of
        
        Returns:
            float
        """
        assert len(vecs1) == len(vecs2)
        assert all(vecs1[i].shape == vecs2[i].shape for i in range(len(vecs1)))
        result = 0
        for i in range(len(vecs1)):
            result += torch.sum(vecs1[i] * vecs2[i])
        return result

    def reset(self):
        """Resets conjugate gradients optimization such that the next step
        will be taken in precisely the direction of the gradient.
        """
        self.prev_grads = None
        self.prev_direction = None

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a CG optimization step.
        
        Args:
            closure (callable): An function that takes no arguments and
                evaluates the model's loss and returns it. Unlike in first-order
                methods, this argument is not optional.
        """
        assert closure is not None, "Must pass a closure to evaluate loss"

        params_with_grad = []
        d_p_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad.detach().clone())
        
        # save parameters of model
        initial_params_with_grad_data = [param.data.detach().clone() 
                    for param in params_with_grad]

        if self.prev_grads is None:
            direction = [-g for g in d_p_list]
        else:
            beta = self._dot_prod([d_p_list[i] - self.prev_grads[i] for i in range(len(d_p_list))], d_p_list) / self._dot_prod(self.prev_grads, self.prev_grads)
            direction = [- d_p_list[i] + beta * self.prev_direction[i] for i in range(len(d_p_list))]
        if self.search: # line search method
            best_loss = float('inf')
            best_params_with_grad_data = None
            for lr in self.lr_range:
                for i, param in enumerate(params_with_grad):
                    param.add_(-lr * direction[i]) # should this be a negative?
                loss = closure()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params_with_grad_data = [param.data.detach().clone()
                        for param in params_with_grad]
                for i, param in enumerate(params_with_grad):
                    param.data = initial_params_with_grad_data[i]
            # set model parameters to the best we found
            for i, param in enumerate(params_with_grad):
                param.data = best_params_with_grad_data[i]
        else: # step size computation under quadratic assumption
            for param in params_with_grad:
                param.grad = None
            sigma = 1e-5 # be very careful about this value
            for i, param in enumerate(params_with_grad):
                param.add_(sigma * direction[i])
            with torch.enable_grad():
                loss = closure()
                loss.backward()
            shifted_d_p_list = [param.grad.detach().clone()
                    for param in params_with_grad]
            H_d_list = [(shifted_d_p_list[i] - d_p_list[i]) / sigma for i in range(len(d_p_list))]
            epsilon = -self._dot_prod(d_p_list, direction) / self._dot_prod(direction, H_d_list)
            for i, param in enumerate(params_with_grad):
                param.data = initial_params_with_grad_data[i]
            for i, param in enumerate(params_with_grad):
                param.grad = d_p_list[i]
            for i, param in enumerate(params_with_grad):
                param.add_(epsilon * direction[i])
        # store grad and direction for next step
        self.prev_grads = [g.clone() for g in d_p_list]
        self.prev_direction = [d.clone() for d in direction]


# class GradientSearch(torch.optim.Optimizer):
#     def __init__(self, params):
#         defaults = dict()
#         super(GradientSearch, self).__init__(params, defaults)
#         self.lr_range = [np.power(2.0, n) for n in range(-32, 10)]
    
#     @torch.no_grad()
#     def step(self, closure=None):
#         assert closure is not None, "Must pass a closure to evaluate loss"

#         params_with_grad = []
#         d_p_list = []
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is not None:
#                     params_with_grad.append(p)
#                     d_p_list.append(p.grad)
#         # save parameters of model
#         initial_params_with_grad_data = [param.data.detach().clone() 
#                     for param in params_with_grad]
#         best_loss = float('inf')
#         best_params_with_grad_data = None
#         for lr in self.lr_range:
#             for i, param in enumerate(params_with_grad):
#                 param.add_(-lr * d_p_list[i])
#             loss = closure()
#             if loss.item() < best_loss:
#                 best_loss = loss.item()
#                 best_params_with_grad_data = [param.data.detach().clone()
#                     for param in params_with_grad]
#             for i, param in enumerate(params_with_grad):
#                 param.data = initial_params_with_grad_data[i]
#         # set model parameters to the best we found
#         for i, param in enumerate(params_with_grad):
#             param.data = best_params_with_grad_data[i]

