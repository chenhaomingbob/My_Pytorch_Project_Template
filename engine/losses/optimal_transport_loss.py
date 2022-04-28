# import ot

import torch
import torch.nn as nn


# os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class FeatureOptimalLoss(nn.Module):
    def __init__(self, epsilon=100, niter=10):
        super(FeatureOptimalLoss, self).__init__()
        # Todo
        #  Add the initial parameters to the config file

        ###########################################
        # self.batch_size = batch_size  # using max value
        self.epsilon = epsilon
        self.niter = niter  # iterations
        ###########################################

    def forward(self, output, target):
        """
            output: [batch_size, dims]
            target: [batch_size, dims]
            only one-dim.
        """
        batch_size = output.shape[0]

        return self._sinkhorn_loss_primal(output, target, self.epsilon, batch_size, self.niter)

    def _sinkhorn_loss_primal(self, x, y, epsilon, n, niter):
        """
            Given two emprical measures with n points each with locations x and y
            outputs an approximation of the OT cost with regularization parameter epsilon
            niter is the max. number of steps in sinkhorn loop
        """
        # The Sinkhorn algorithm takes as input three variables :
        C = self._squared_distances(x, y)  # Wasserstein cost function       把特征拉平后两两之间的cost

        # mu = Variable(1./ n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
        # nu = Variable(1./ n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)

        mu = torch.full([n, ], 1 / n).cuda()
        nu = torch.full([n, ], 1 / n).cuda()

        # Parameters of the Sinkhorn algorithm.
        # epsilon            = (.1)**2          # regularization parameter
        rho = 1  # (.5) **2          # unbalanced transport (See PhD Th. of Lenaic Chizat)
        tau = -.8  # nesterov-like acceleration

        lam = rho / (rho + epsilon)  # Update exponent

        # Elementary operations .....................................................................
        def ave(u, u1):
            "Barycenter subroutine, used by kinetic acceleration through extrapolation."
            return tau * u + (1 - tau) * u1

        def M(u, v):
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

        lse = lambda A: torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # slight modif to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * mu, 0. * nu, 0.
        actual_nits = 0

        for i in range(niter):
            u1 = u  # useful to check the update

            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
            v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
            # u = ave( u, lam * ( epsilon * ( torch.log(mu.unsqueeze(1)) - lse(M(u,v))   ) + u ) )
            # v = ave( v, lam * ( epsilon * ( torch.log(nu.unsqueeze(1)) - lse(M(u,v).t()) ) + v ) )
            err = (u - u1).abs().sum()

            actual_nits += 1
            if (err < 1e-1).data.cpu().numpy():
                break
        U, V = u, v
        Gamma = torch.exp(M(U, V))  # Eventual transport plan g = diag(a)*K*diag(b)
        cost = torch.sum(Gamma * C)  # Simplistic cost, chosen for readability in this tutorial

        return cost

    def _squared_distances(self, x, y):
        "Returns the matrix of $\|x_i-y_j\|^2$."
        x_col = x.unsqueeze(1)  # x.dimshuffle(0, 'x', 1)
        y_lin = y.unsqueeze(0)  # y.dimshuffle('x', 0, 1)
        c = torch.sum(torch.abs(x_col - y_lin), 2)
        return c

    def _adjust_parameters(self):
        need_adjust_parameters = {"niter": [1, 10, 100],
                                  "epsilon": [10 ** 4, 10 ** 3, 10 ** 2, 10],
                                  "batch_size": "max_value"}
        return need_adjust_parameters

    def _get_parameters_info(self):
        pass


# OTA = dict(
#     REG_WEIGHT=1.5,
#     SINKHORN_EPS=0.1,
#     SINKHORN_ITER=50,
#     TOP_CANDIDATES=20,
# )

class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


if __name__ == '__main__':
    x = torch.randn(16, 1024)
    x = x.cuda()
    y = torch.randn(16, 1024)
    y = y.cuda()
    net = FeatureOptimalLoss()
    net = net.cuda()
    loss = net(x, y)
    print(loss)
    loss = net(x, x)
    print(loss)
