import sys
import os
from torch.backends import cudnn
import random
import numpy as np
import tensorly as tl
import scipy as sp
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import optim, nn
from torch.utils.data import DataLoader

#from utils import NET, CustomDataset, draw, IdDataset
from tensorly import unfold, fold
import torch
import tensorly as tl
import tensorly.tenalg as tl_alg
#from tensorly.tenalg import khatri_rao

# 设置 Tensorly 的后端为 PyTorch
tl.set_backend('pytorch')

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class Model(object):
    def __init__(self, name='NCTF',device='cuda'):
        super().__init__()
        self.name = name
        self.device =device
        #self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def NCTF_torch(self, X, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone()
            R = D - A @ X @ B - mu * X
            P = R.clone()

            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break

                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P

            return X

        m = X.shape[0]
        d = X.shape[2]
        #print(m, m, d)
        # Initialization
        #rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff=1.15

        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float64)
        C = torch.rand((m, r), dtype=torch.float64)
        D = torch.rand((d, r), dtype=torch.float64)

        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float64)
        R2 = Y3 = P = F  = torch.zeros((d, r), dtype=torch.float64)

        A_1 = torch.zeros((r, r), dtype=torch.float64)
        A_2 = torch.zeros((r, r), dtype=torch.float64)

        X1 = unfold(X, 0)
        X2 = unfold(X, 1)
        X3 = unfold(X, 2)

        I = torch.eye(r, dtype=torch.float64)
        U = torch.diag(S_m.sum(1)) - S_m
        V = torch.diag(S_d.sum(1)) - S_d

        W = X.clone()
        X_old = X.clone()
        del X
        
        for i in range(max_iter):
            #print(f"iter\t{i + 1}\t\tbegin")
            # output_X_old = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)

            # Conjugate Gradient method (CG)
            O_1 = M.T @ M
            O_2 = D.T @ D
            A_1 = CG(torch.zeros_like(A_1), mu * O_1, O_1, mu * M.T @ S_m @ M, lam, 0.01, 200)
            A_2 = CG(torch.zeros_like(A_2), eta * O_2, O_2, eta * D.T @ S_d @ D, lam, 0.01, 200)

            # 1-slice update H,Z,M, R1,Y1,Y2
            MA_1 = M @ A_1
            H = (mu * S_m @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * S_m @ H @ A_1.T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(
                    E.T @ E + mu * A_1 @ H.T @ H @ A_1.T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m) + alpha * U) @ (rho_2 * M + Y2)

            # R1 = R1 + theta_1 * (M - H)
            # Y1 = Y1 + rho_1 * (M - C)
            # Y2 = Y2 + rho_2 * (M - Z)

            # theta_1 *= coff
            # rho_1 *= coff
            # rho_2 *= coff

            # 2-slice update C
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)

            # 3-slice update P,F,D ,R2,Y3
            DA_2 = D @ A_2
            P = (eta * S_d @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * S_d @ P @ A_2.T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * A_2 @ P.T @ P @ A_2.T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d) + beta * V) @ (rho_3 * D + Y3)

            # Update the Lagrange Multiplier
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)

            # Update rho and theta
            # theta_2 *= coff
            # rho_3 *= coff
            # theta_1 *= coff
            # rho_1 *= coff
            # rho_2 *= coff

            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
            #print(theta_1)

            # output_X = fold(D @ tl.tenalg.khatri_rao([M, C]).T, 2, X.shape)
            # err = torch.linalg.norm(output_X - output_X_old) / torch.linalg.norm(output_X_old)

            # output_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)
            # err = torch.linalg.norm(output_X - output_X_old) / torch.linalg.norm(output_X_old)
            #print(M.shape,C.shape,D.shape)

            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)
            #print(X_comp.shape)
            #print(W.shape)
            tmp = (1-W) * X_comp
            X_new = X_old + tmp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new

            #print(err)
            if err < tol:
                break

        #predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape).cpu().numpy()
        predict_X = X_comp.cpu().numpy()

        return predict_X, M, C, D

    def NCTF_torch_gpu(self, X, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone().to(self.device)
            R = D - A @ X @ B - mu * X
            P = R.clone()
    
            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break
    
                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P
    
            return X
    
        # Ensure all tensors are on the GPU
        X = X.to(self.device)
        S_m = S_m.to(self.device)
        S_d = S_d.to(self.device)
    
        m = X.shape[0]
        d = X.shape[2]
    
        # Initialization
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff = 1.15
    
        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float64, device=self.device)
        C = torch.rand((m, r), dtype=torch.float64, device=self.device)
        D = torch.rand((d, r), dtype=torch.float64, device=self.device)
    
        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float64, device=self.device)
        R2 = Y3 = P = F  = torch.zeros((d, r), dtype=torch.float64, device=self.device)
    
        A_1 = torch.zeros((r, r), dtype=torch.float64, device=self.device)
        A_2 = torch.zeros((r, r), dtype=torch.float64, device=self.device)
    
        X1 = unfold(X, 0).to(self.device)
        X2 = unfold(X, 1).to(self.device)
        X3 = unfold(X, 2).to(self.device)
    
        I = torch.eye(r, dtype=torch.float64, device=self.device)
        U = torch.diag(S_m.sum(1)) - S_m
        V = torch.diag(S_d.sum(1)) - S_d

        W = X.clone()
        X_old = X.clone()
        shape = X.shape
        del X
    
        for i in range(max_iter):
            # Conjugate Gradient method (CG)
            O_1 = M.T @ M
            O_2 = D.T @ D
            A_1 = CG(torch.zeros_like(A_1), mu * O_1, O_1, mu * M.T @ S_m @ M, lam, 0.01, 200)
            A_2 = CG(torch.zeros_like(A_2), eta * O_2, O_2, eta * D.T @ S_d @ D, lam, 0.01, 200)
    
            # 1-slice update H,Z,M, R1,Y1,Y2
            MA_1 = M @ A_1
            H = (mu * S_m @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * S_m @ H @ A_1.T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(E.T @ E + mu * A_1 @ H.T @ H @ A_1.T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m, device=self.device) + alpha * U) @ (rho_2 * M + Y2)
    
            # 2-slice update C
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)
    
            # 3-slice update P,F,D ,R2,Y3
            DA_2 = D @ A_2
            P = (eta * S_d @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * S_d @ P @ A_2.T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * A_2 @ P.T @ P @ A_2.T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d, device=self.device) + beta * V) @ (rho_3 * D + Y3)
    
            # Update the Lagrange Multiplier
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)
    
            # Update rho and theta
            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
    
            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).to(self.device)
            tmp = (1 - W) * X_comp
            X_new = X_old + tmp
            del tmp,X_comp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new
            del X_new
    
            if err < tol:
                break
    
        predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).cpu().numpy()

        return predict_X, M, C, D

    def NCTF_torch(self, X, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone()
            R = D - A @ X @ B - mu * X
            P = R.clone()

            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break

                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P

            return X

        m = X.shape[0]
        d = X.shape[2]
        #print(m, m, d)
        # Initialization
        #rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff=1.15

        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float64)
        C = torch.rand((m, r), dtype=torch.float64)
        D = torch.rand((d, r), dtype=torch.float64)

        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float64)
        R2 = Y3 = P = F  = torch.zeros((d, r), dtype=torch.float64)

        A_1 = torch.zeros((r, r), dtype=torch.float64)
        A_2 = torch.zeros((r, r), dtype=torch.float64)

        X1 = unfold(X, 0)
        X2 = unfold(X, 1)
        X3 = unfold(X, 2)

        I = torch.eye(r, dtype=torch.float64)
        U = torch.diag(S_m.sum(1)) - S_m
        V = torch.diag(S_d.sum(1)) - S_d

        W = X.clone()
        X_old = X.clone()
        del X
        
        for i in range(max_iter):
            #print(f"iter\t{i + 1}\t\tbegin")
            # output_X_old = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)

            # Conjugate Gradient method (CG)
            O_1 = M.T @ M
            O_2 = D.T @ D
            A_1 = CG(torch.zeros_like(A_1), mu * O_1, O_1, mu * M.T @ S_m @ M, lam, 0.01, 200)
            A_2 = CG(torch.zeros_like(A_2), eta * O_2, O_2, eta * D.T @ S_d @ D, lam, 0.01, 200)

            # 1-slice update H,Z,M, R1,Y1,Y2
            MA_1 = M @ A_1
            H = (mu * S_m @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * S_m @ H @ A_1.T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(
                    E.T @ E + mu * A_1 @ H.T @ H @ A_1.T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m) + alpha * U) @ (rho_2 * M + Y2)

            # R1 = R1 + theta_1 * (M - H)
            # Y1 = Y1 + rho_1 * (M - C)
            # Y2 = Y2 + rho_2 * (M - Z)

            # theta_1 *= coff
            # rho_1 *= coff
            # rho_2 *= coff

            # 2-slice update C
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)

            # 3-slice update P,F,D ,R2,Y3
            DA_2 = D @ A_2
            P = (eta * S_d @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * S_d @ P @ A_2.T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * A_2 @ P.T @ P @ A_2.T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d) + beta * V) @ (rho_3 * D + Y3)

            # Update the Lagrange Multiplier
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)

            # Update rho and theta
            # theta_2 *= coff
            # rho_3 *= coff
            # theta_1 *= coff
            # rho_1 *= coff
            # rho_2 *= coff

            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
            #print(theta_1)

            # output_X = fold(D @ tl.tenalg.khatri_rao([M, C]).T, 2, X.shape)
            # err = torch.linalg.norm(output_X - output_X_old) / torch.linalg.norm(output_X_old)

            # output_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)
            # err = torch.linalg.norm(output_X - output_X_old) / torch.linalg.norm(output_X_old)
            #print(M.shape,C.shape,D.shape)

            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)
            #print(X_comp.shape)
            #print(W.shape)
            tmp = (1-W) * X_comp
            X_new = X_old + tmp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new

            #print(err)
            if err < tol:
                break

        #predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape).cpu().numpy()
        predict_X = X_comp.cpu().numpy()

        return predict_X, M, C, D

    def NCTF_torch_gpu_1(self, X, W, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone().to(self.device)
            R = D - A @ X @ B - mu * X
            P = R.clone()
    
            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break
    
                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P
    
            return X
    
        # Ensure all tensors are on the GPU
        X = X.to(self.device)
        S_m = S_m.to(self.device)
        S_d = S_d.to(self.device)
        W = W.to(self.device)
    
        m = X.shape[0]
        d = X.shape[2]
    
        # Initialization
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff = 1.15
    
        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float64, device=self.device)
        C = torch.rand((m, r), dtype=torch.float64, device=self.device)
        D = torch.rand((d, r), dtype=torch.float64, device=self.device)
    
        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float64, device=self.device)
        R2 = Y3 = P = F  = torch.zeros((d, r), dtype=torch.float64, device=self.device)
    
        A_1 = torch.zeros((r, r), dtype=torch.float64, device=self.device)
        A_2 = torch.zeros((r, r), dtype=torch.float64, device=self.device)
    
        X1 = unfold(X, 0).to(self.device)
        X2 = unfold(X, 1).to(self.device)
        X3 = unfold(X, 2).to(self.device)
    
        I = torch.eye(r, dtype=torch.float64, device=self.device)
        U = torch.diag(S_m.sum(1)) - S_m
        V = torch.diag(S_d.sum(1)) - S_d

        #W = X.clone()
        X_old = X.clone()
        shape = X.shape
        del X
    
        for i in range(max_iter):
            # Conjugate Gradient method (CG)
            O_1 = M.T @ M
            O_2 = D.T @ D
            A_1 = CG(torch.zeros_like(A_1), mu * O_1, O_1, mu * M.T @ S_m @ M, lam, 0.01, 200)
            A_2 = CG(torch.zeros_like(A_2), eta * O_2, O_2, eta * D.T @ S_d @ D, lam, 0.01, 200)
    
            # 1-slice update H,Z,M, R1,Y1,Y2
            MA_1 = M @ A_1
            H = (mu * S_m @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * S_m @ H @ A_1.T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(E.T @ E + mu * A_1 @ H.T @ H @ A_1.T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m, device=self.device) + alpha * U) @ (rho_2 * M + Y2)
    
            # 2-slice update C
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)
    
            # 3-slice update P,F,D ,R2,Y3
            DA_2 = D @ A_2
            P = (eta * S_d @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * S_d @ P @ A_2.T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * A_2 @ P.T @ P @ A_2.T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d, device=self.device) + beta * V) @ (rho_3 * D + Y3)
    
            # Update the Lagrange Multiplier
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)
    
            # Update rho and theta
            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
    
            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).to(self.device)
            tmp = (1 - W) * X_comp
            X_new = X_old + tmp
            del tmp,X_comp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new
            del X_new
    
            if err < tol:
                break
    
        predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).cpu().numpy()

        return predict_X, M, C, D

    def NCTF_torch_splitSim(self, X, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone()
            R = D - A @ X @ B - mu * X
            P = R.clone()

            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break

                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P

            return X

        m = X.shape[0]
        d = X.shape[2]
        #print(m, m, d)
        W = X.clone()
        X_old = X.clone()
        # Initialization
        #rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff=1.15

        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float64)
        C = torch.rand((m, r), dtype=torch.float64)
        D = torch.rand((d, r), dtype=torch.float64)

        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float64)
        R2 = Y3 = P = F  = torch.zeros((d, r), dtype=torch.float64)

        # A_1 = torch.zeros((r, r), dtype=torch.float64)
        # A_2 = torch.zeros((r, r), dtype=torch.float64)

        # X1 = torch.tensor(unfold(X, 0), dtype=torch.float32)
        # X2 = torch.tensor(unfold(X, 1), dtype=torch.float32)
        # X3 = torch.tensor(unfold(X, 2), dtype=torch.float32)
        X1 = unfold(X, 0)
        X2 = unfold(X, 1)
        X3 = unfold(X, 2)

        I = torch.eye(r, dtype=torch.float64)

        U=[]
        for i in range(len(S_m)):
            S_mi=S_m[i]
            Ui = torch.diag(S_mi.sum(1)) - S_mi
            U.append(Ui)

        V=[]
        for i in range(len(S_d)):
            S_di=S_d[i]
            Vi = torch.diag(S_di.sum(1)) - S_di
            V.append(Vi)
        
        # U = torch.diag(S_m.sum(1)) - S_m
        # V = torch.diag(S_d.sum(1)) - S_d

        for k in range(max_iter):
            #print(f"iter\t{k + 1}\t\tbegin")
            #output_X_old = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)

            # Conjugate Gradient method (CG)
            O_1 = M.T @ M
            O_2 = D.T @ D
            
            A_1=[]
            for i in range(len(S_m)):
                A_1i = CG(torch.zeros((r, r), dtype=torch.float64), mu * O_1, O_1, mu * M.T @ S_m[i] @ M, lam, 0.01, 200)
                A_1.append(A_1i)
            
            A_2=[]
            for i in range(len(S_d)):
                A_2i = CG(torch.zeros((r, r), dtype=torch.float64), eta * O_2, O_2, eta * D.T @ S_d[i] @ D, lam, 0.01, 200)
                A_2.append(A_2i)

            # 1-slice update H,Z,M, R1,Y1,Y2
            MA_1 = M @ sum(A_1)
            H = (mu * sum(S_m) @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * sum(S_m) @ H @ sum(A_1).T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(
                    E.T @ E + mu * sum(A_1) @ H.T @ H @ sum(A_1).T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m) + alpha * sum(U)) @ (rho_2 * M + Y2)

            # R1 = R1 + theta_1 * (M - H)
            # Y1 = Y1 + rho_1 * (M - C)
            # Y2 = Y2 + rho_2 * (M - Z)
            # Y1 = Y1 + rho_1 * (C - M)
            # Y2 = Y2 + rho_2 * (Z - M)

            # theta_1 *= coff
            # rho_1 *= coff
            # rho_2 *= coff

            # 2-slice update C
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)

            # 3-slice update P,F,D ,R2,Y3
            DA_2 = D @ sum(A_2)
            P = (eta * sum(S_d) @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * sum(S_d) @ P @ sum(A_2).T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * sum(A_2) @ P.T @ P @ sum(A_2).T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d) + beta * sum(V)) @ (rho_3 * D + Y3)

            # Update the Lagrange Multiplier
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            #Y3 = Y3 + rho_3 * (F - D)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)

            # Update rho and theta
            # theta_2 *= coff
            # rho_3 *= coff
            # theta_1 *= coff
            # rho_1 *= coff
            # rho_2 *= coff

            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
            #print(theta_1)

            # output_X = fold(D @ tl.tenalg.khatri_rao([M, C]).T, 2, X.shape)
            # err = torch.linalg.norm(output_X - output_X_old) / torch.linalg.norm(output_X_old)

            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)
            #print(X_comp.shape)
            #print(W.shape)
            tmp = (1-W) * X_comp
            X_new = X_old + tmp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new

            #print(err)
            if err < tol:
                break

        #predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape).cpu().numpy()
        predict_X = X_comp.cpu().numpy()

        return predict_X, M, C, D

    def NCTF_torch_splitSim_gpu(self, X, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone()
            R = D - A @ X @ B - mu * X
            P = R.clone()
    
            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break
    
                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P
    
            return X
    
        # Transfer all tensors to the GPU
        X = X.to(self.device)
        S_m = [sm.to(self.device) for sm in S_m]
        S_d = [sd.to(self.device) for sd in S_d]
    
        m = X.shape[0]
        d = X.shape[2]
        # W = X.clone()
        # X_old = X.clone()
    
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff = 1.15
    
        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float64, device=self.device)
        C = torch.rand((m, r), dtype=torch.float64, device=self.device)
        D = torch.rand((d, r), dtype=torch.float64, device=self.device)
    
        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float64, device=self.device)
        R2 = Y3 = P = F = torch.zeros((d, r), dtype=torch.float64, device=self.device)
    
        X1 = unfold(X, 0).to(self.device)
        X2 = unfold(X, 1).to(self.device)
        X3 = unfold(X, 2).to(self.device)
    
        I = torch.eye(r, dtype=torch.float64, device=self.device)
    
        U = []
        for i in range(len(S_m)):
            S_mi = S_m[i]
            Ui = torch.diag(S_mi.sum(1)) - S_mi
            U.append(Ui)
    
        V = []
        for i in range(len(S_d)):
            S_di = S_d[i]
            Vi = torch.diag(S_di.sum(1)) - S_di
            V.append(Vi)

        W = X.clone()
        X_old = X.clone()
        shape = X.shape
        del X
        
        for k in range(max_iter):
            O_1 = M.T @ M
            O_2 = D.T @ D
    
            A_1 = []
            for i in range(len(S_m)):
                A_1i = CG(torch.zeros((r, r), dtype=torch.float64, device=self.device), mu * O_1, O_1, mu * M.T @ S_m[i] @ M, lam, 0.01, 200)
                A_1.append(A_1i)
    
            A_2 = []
            for i in range(len(S_d)):
                A_2i = CG(torch.zeros((r, r), dtype=torch.float64, device=self.device), eta * O_2, O_2, eta * D.T @ S_d[i] @ D, lam, 0.01, 200)
                A_2.append(A_2i)
    
            MA_1 = M @ sum(A_1)
            H = (mu * sum(S_m) @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * sum(S_m) @ H @ sum(A_1).T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(
                    E.T @ E + mu * sum(A_1) @ H.T @ H @ sum(A_1).T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m, device=self.device) + alpha * sum(U)) @ (rho_2 * M + Y2)
    
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)
    
            DA_2 = D @ sum(A_2)
            P = (eta * sum(S_d) @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * sum(S_d) @ P @ sum(A_2).T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * sum(A_2) @ P.T @ P @ sum(A_2).T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d, device=self.device) + beta * sum(V)) @ (rho_3 * D + Y3)
    
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)
    
            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
    
            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape)
            tmp = (1-W) * X_comp
            X_new = X_old + tmp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new
    
            if err < tol:
                break
    
        predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).cpu().numpy()
    
        return predict_X, M, C, D

    def NCTF_torch_gpu_float32(self, X, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone().to(self.device)
            R = D - A @ X @ B - mu * X
            P = R.clone()
    
            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break
    
                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P
    
            return X
            
        fix_seed(2024)
        # Ensure all tensors are on the GPU
        X = X.to(self.device)
        S_m = S_m.to(self.device)
        S_d = S_d.to(self.device)
    
        m = X.shape[0]
        d = X.shape[2]
    
        # Initialization
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff = 1.15
    
        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float32, device=self.device)
        C = torch.rand((m, r), dtype=torch.float32, device=self.device)
        D = torch.rand((d, r), dtype=torch.float32, device=self.device)
    
        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float32, device=self.device)
        R2 = Y3 = P = F  = torch.zeros((d, r), dtype=torch.float32, device=self.device)
    
        A_1 = torch.zeros((r, r), dtype=torch.float32, device=self.device)
        A_2 = torch.zeros((r, r), dtype=torch.float32, device=self.device)
    
        X1 = unfold(X, 0).to(self.device)
        X2 = unfold(X, 1).to(self.device)
        X3 = unfold(X, 2).to(self.device)
    
        I = torch.eye(r, dtype=torch.float32, device=self.device)
        U = torch.diag(S_m.sum(1)) - S_m
        V = torch.diag(S_d.sum(1)) - S_d

        W = X.clone()
        X_old = X.clone()
        shape = X.shape
        del X
        # S_m = S_m * 1e-10
        # S_d = S_d * 1e-10
    
        for i in range(max_iter):
            #print(f"iter\t{i + 1}\t\tbegin")
            # output_X_old = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, X.shape)
            # Conjugate Gradient method (CG)
            O_1 = M.T @ M
            O_2 = D.T @ D
            A_1 = CG(torch.zeros_like(A_1), mu * O_1, O_1, mu * M.T @ S_m @ M, lam, 0.01, 200)
            A_2 = CG(torch.zeros_like(A_2), eta * O_2, O_2, eta * D.T @ S_d @ D, lam, 0.01, 200)
    
            # 1-slice update H,Z,M, R1,Y1,Y2
            MA_1 = M @ A_1
            H = (mu * S_m @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * S_m @ H @ A_1.T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(E.T @ E + mu * A_1 @ H.T @ H @ A_1.T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m, device=self.device) + alpha * U) @ (rho_2 * M + Y2)
    
            # 2-slice update C
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)
    
            # 3-slice update P,F,D ,R2,Y3
            DA_2 = D @ A_2
            P = (eta * S_d @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * S_d @ P @ A_2.T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * A_2 @ P.T @ P @ A_2.T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d, device=self.device) + beta * V) @ (rho_3 * D + Y3)
    
            # Update the Lagrange Multiplier
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)
    
            # Update rho and theta
            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
    
            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).to(self.device)
            tmp = (1 - W) * X_comp
            X_new = X_old + tmp
            del tmp,X_comp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            #print(err)
            X_old = X_new
            del X_new
    
            if err < tol:
                break
    
        predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).cpu().numpy()
        return predict_X, M, C, D

    def NCTF_torch_gpu_float32_1(self, X, W, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone().to(self.device)
            R = D - A @ X @ B - mu * X
            P = R.clone()
    
            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break
    
                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P
    
            return X
    
        # Ensure all tensors are on the GPU
        X = X.to(self.device)
        S_m = S_m.to(self.device)
        S_d = S_d.to(self.device)
        W = W.to(self.device)
    
        m = X.shape[0]
        d = X.shape[2]
    
        # Initialization
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff = 1.15
    
        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float32, device=self.device)
        C = torch.rand((m, r), dtype=torch.float32, device=self.device)
        D = torch.rand((d, r), dtype=torch.float32, device=self.device)
    
        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float32, device=self.device)
        R2 = Y3 = P = F  = torch.zeros((d, r), dtype=torch.float32, device=self.device)
    
        A_1 = torch.zeros((r, r), dtype=torch.float32, device=self.device)
        A_2 = torch.zeros((r, r), dtype=torch.float32, device=self.device)
    
        X1 = unfold(X, 0).to(self.device)
        X2 = unfold(X, 1).to(self.device)
        X3 = unfold(X, 2).to(self.device)
    
        I = torch.eye(r, dtype=torch.float32, device=self.device)
        U = torch.diag(S_m.sum(1)) - S_m
        V = torch.diag(S_d.sum(1)) - S_d

        #W = X.clone()
        X_old = X.clone()
        shape = X.shape
        del X
        # S_m = S_m * 1e-10
        # S_d = S_d * 1e-10
    
        for i in range(max_iter):
            # Conjugate Gradient method (CG)
            O_1 = M.T @ M
            O_2 = D.T @ D
            A_1 = CG(torch.zeros_like(A_1), mu * O_1, O_1, mu * M.T @ S_m @ M, lam, 0.01, 200)
            A_2 = CG(torch.zeros_like(A_2), eta * O_2, O_2, eta * D.T @ S_d @ D, lam, 0.01, 200)
    
            # 1-slice update H,Z,M, R1,Y1,Y2
            MA_1 = M @ A_1
            H = (mu * S_m @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * S_m @ H @ A_1.T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(E.T @ E + mu * A_1 @ H.T @ H @ A_1.T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m, device=self.device) + alpha * U) @ (rho_2 * M + Y2)
    
            # 2-slice update C
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)
    
            # 3-slice update P,F,D ,R2,Y3
            DA_2 = D @ A_2
            P = (eta * S_d @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * S_d @ P @ A_2.T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * A_2 @ P.T @ P @ A_2.T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d, device=self.device) + beta * V) @ (rho_3 * D + Y3)
    
            # Update the Lagrange Multiplier
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)
    
            # Update rho and theta
            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
    
            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).to(self.device)
            tmp = (1 - W) * X_comp
            X_new = X_old + tmp
            del tmp,X_comp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new
            del X_new
    
            if err < tol:
                break
    
        predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).cpu().numpy()
        return predict_X, M, C, D
    
    def NCTF_torch_splitSim_gpu_float32(self, X, S_m, S_d, r=4, mu=0.125, alpha=0.1, eta=0.25, beta=0.1, lam=0.001, tol=1e-4, max_iter=500):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Conjugate Gradient method (CG) using PyTorch
        def CG(X_initial, A, B, D, mu, tol, max_iter):
            X = X_initial.clone()
            R = D - A @ X @ B - mu * X
            P = R.clone()
    
            for i in range(max_iter):
                R_norm = torch.trace(R @ R.T)
                Q = A @ P @ B + mu * P
                alpha = R_norm / torch.trace(Q @ P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = torch.norm(R)
                if err < tol:
                    break
    
                beta = torch.trace(R @ R.T) / R_norm
                P = R + beta * P
    
            return X
    
        # Transfer all tensors to the GPU
        X = X.to(self.device)
        S_m = [sm.to(self.device) for sm in S_m]
        S_d = [sd.to(self.device) for sd in S_d]
    
        m = X.shape[0]
        d = X.shape[2]
        # W = X.clone()
        # X_old = X.clone()
    
        rho_1 = rho_2 = rho_3 = theta_1 = theta_2 = 1e-6
        rho_1_max = rho_2_max = rho_3_max = theta_1_max = theta_2_max = 1e12
        coff = 1.15
    
        torch.manual_seed(2024)
        M = torch.rand((m, r), dtype=torch.float32, device=self.device)
        C = torch.rand((m, r), dtype=torch.float32, device=self.device)
        D = torch.rand((d, r), dtype=torch.float32, device=self.device)
    
        Y1 = Y2 = R1 = H = Z = torch.zeros((m, r), dtype=torch.float32, device=self.device)
        R2 = Y3 = P = F = torch.zeros((d, r), dtype=torch.float32, device=self.device)
    
        X1 = unfold(X, 0).to(self.device)
        X2 = unfold(X, 1).to(self.device)
        X3 = unfold(X, 2).to(self.device)
    
        I = torch.eye(r, dtype=torch.float32, device=self.device)
    
        U = []
        for i in range(len(S_m)):
            S_mi = S_m[i]
            Ui = torch.diag(S_mi.sum(1)) - S_mi
            U.append(Ui)
    
        V = []
        for i in range(len(S_d)):
            S_di = S_d[i]
            Vi = torch.diag(S_di.sum(1)) - S_di
            V.append(Vi)

        W = X.clone()
        X_old = X.clone()
        shape = X.shape
        del X
        
        for k in range(max_iter):
            O_1 = M.T @ M
            O_2 = D.T @ D
    
            A_1 = []
            for i in range(len(S_m)):
                A_1i = CG(torch.zeros((r, r), dtype=torch.float32, device=self.device), mu * O_1, O_1, mu * M.T @ S_m[i] @ M, lam, 0.01, 200)
                A_1.append(A_1i)
    
            A_2 = []
            for i in range(len(S_d)):
                A_2i = CG(torch.zeros((r, r), dtype=torch.float32, device=self.device), eta * O_2, O_2, eta * D.T @ S_d[i] @ D, lam, 0.01, 200)
                A_2.append(A_2i)
    
            MA_1 = M @ sum(A_1)
            H = (mu * sum(S_m) @ MA_1 + theta_1 * M + R1) @ torch.inverse(mu * MA_1.T @ MA_1 + theta_1 * I)
            E = tl.tenalg.khatri_rao([C, D])
            W_1 = torch.diag(0.5 * torch.linalg.norm(M.T, dim=1))
            M = (X1 @ E + mu * sum(S_m) @ H @ sum(A_1).T + rho_1 * C + theta_1 * H + rho_2 * Z - Y1 - R1 - Y2) @ \
                torch.inverse(
                    E.T @ E + mu * sum(A_1) @ H.T @ H @ sum(A_1).T + lam * W_1 * I + rho_1 * I + theta_1 * I + rho_2 * I)
            Z = torch.inverse(rho_2 * torch.eye(m, device=self.device) + alpha * sum(U)) @ (rho_2 * M + Y2)
    
            G = tl.tenalg.khatri_rao([M, D])
            C = (X2 @ G + rho_1 * M + Y1) @ torch.inverse(G.T @ G + rho_1 * I)
    
            DA_2 = D @ sum(A_2)
            P = (eta * sum(S_d) @ DA_2 + theta_2 * D + R2) @ torch.inverse(eta * DA_2.T @ DA_2 + theta_2 * I)
            J = tl.tenalg.khatri_rao([M, C])
            W_2 = torch.diag(0.5 * torch.linalg.norm(D.T, dim=1))
            D = (X3 @ J + eta * sum(S_d) @ P @ sum(A_2).T + theta_2 * P + rho_3 * F - R2 - Y3) @ \
                torch.inverse(J.T @ J + eta * sum(A_2) @ P.T @ P @ sum(A_2).T + lam * W_2 * I + theta_2 * I + rho_3 * I)
            F = torch.inverse(rho_3 * torch.eye(d, device=self.device) + beta * sum(V)) @ (rho_3 * D + Y3)
    
            R2 = R2 + theta_2 * (D - P)
            Y3 = Y3 + rho_3 * (D - F)
            R1 = R1 + theta_1 * (M - H)
            Y1 = Y1 + rho_1 * (M - C)
            Y2 = Y2 + rho_2 * (M - Z)
    
            theta_1 = min(theta_1_max, theta_1 * coff)
            theta_2 = min(theta_2_max, theta_2 * coff)
            rho_1 = min(rho_1_max, rho_1 * coff)
            rho_2 = min(rho_2_max, rho_2 * coff)
            rho_3 = min(rho_3_max, rho_3 * coff)
    
            X_comp = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape)
            tmp = (1-W) * X_comp
            X_new = X_old + tmp
            err = torch.linalg.norm(X_old - X_new) / torch.linalg.norm(X_old)
            X_old = X_new
    
            if err < tol:
                break
    
        predict_X = fold(M @ tl.tenalg.khatri_rao([C, D]).T, 0, shape).cpu().numpy()
    
        return predict_X, M, C, D
        
    def __call__(self):

        return getattr(self, self.name, None)


