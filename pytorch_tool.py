import numpy as np
import torch
from random import shuffle

EPS = 1e-2


def torch_sgd_linear(dots: torch.tensor, batch_size, start=None, lr=1e-6, epoch_limit=100, method="SGD", log=False,
                     dtype=torch.float64):
    if start is None:
        start = [0., 0.]
    n = len(dots)
    # looking for solution of linear regression y = ax + b, so need to search the a and b
    model = torch.nn.Linear(1, 1, dtype=dtype)
    model.weight.data = torch.tensor([[float(start[0])]], dtype=dtype)
    model.bias.data = torch.tensor([[float(start[1])]], dtype=dtype)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    if method == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif method == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif method == 'Nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    elif method == 'AdaGrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif method == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise RuntimeError("Unsupported operation")

    converged = False
    way = [start]

    for epoch in range(epoch_limit):
        if converged:
            break
        if log:
            print(f"Epoch {epoch + 1}\n-------------------------------")
        dots_perm = torch.randperm(n)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            indexes = dots_perm[i * batch_size: (i + 1) * batch_size]
            shuffled_batch = dots[indexes]
            x_batch, y_batch = shuffled_batch[:, 0], shuffled_batch[:, 1]
            y_batch_pred = model(x_batch.unsqueeze(-1))

            loss = loss_fn(y_batch_pred, y_batch.unsqueeze(-1))

            optimizer.zero_grad()
            loss.backward()
            if all(abs(param.grad.item()) < EPS for param in model.parameters()):
                converged = True
                break
            optimizer.step()
            ans_a, ans_b = model.weight.item(), model.bias.item()
            way.append((ans_a, ans_b))
            if log and i % 10 == 0:
                loss, current = loss.item(), (i + 1) * len(x_batch)
                print(f"loss: {loss:>7f}  [{current:>5d}/{n:>5d}]")
    return converged, np.array(way)
