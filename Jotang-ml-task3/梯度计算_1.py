import numpy as np
import torch

def MyGrad(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    a = x*y
    b = x*x
    c = a+b
    f = c*c

    df_dc = 2*c
    dc_da = 1
    dc_db = 1
    da_dx = y
    da_dy = x
    db_dx = 2*x

    df_dx = df_dc*dc_da*da_dx + df_dc*dc_db*db_dx
    df_dy = df_dc*dc_da*da_dy
    return f, df_dx, df_dy

def torch_grad(x_val, y_val):
    x = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_val, dtype=torch.float64, requires_grad=True)
    f = (x*y + x**2)**2
    f.backward()

    return f.item(), x.grad.item(), y.grad.item()
if __name__ == '__main__':
    x_val = 2.0
    y_val = 3.0
    np_f, np_dx, np_dy = MyGrad(x_val, y_val)
    tc_f, tc_dx, tc_dy = torch_grad(x_val, y_val)
    print(f"{np_f, np_dx, np_dy}\n{tc_f, tc_dx, tc_dy}")