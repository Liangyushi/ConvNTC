import random
import numpy as np
import tensorly as tl
import scipy as sp
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import optim, nn
from torch.utils.data import DataLoader

from utils import NET, CustomDataset, draw, IdDataset
from tensorly import unfold, fold
import torch
import tensorly as tl
import tensorly.tenalg as tl_alg


# from tensorly.tenalg import khatri_rao

# 设置 Tensorly 的后端为 PyTorch
tl.set_backend('numpy')


class Model(object):
    def __init__(self, name='TDRC'):
        super().__init__()
        self.name = name

    def CTF(self, X, S1, S2, r=51, mu=0.5, eta=0.2, alpha=0.5, beta=0.5, lam=0.001, tol=1e-6, max_iter=200):

        def CG(X_initial, A, B, D, mu, tol, max_iter):
            '''
            A: v U.T U
            B: V.T V
            D: v U.T O V
            '''
            X = X_initial
            R = D - A * X * B - mu * X
            P = np.array(R, copy=True)

            for i in range(max_iter):
                R_norm = np.trace(R * R.T)
                Q = A * P * B + mu * P
                alpha = R_norm / np.trace(Q * P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = np.linalg.norm(R)
                if err < tol:
                    # print("CG convergence: iter = %d" % i)
                    break

                beta = np.trace(R * R.T) / R_norm
                P = R + beta * P

            return X

        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]
        #print(m, d, t)

        # initialization
        rho_1 = 1
        rho_2 = 1
        xita = 1
        np.random.seed(2024)
        C = np.mat(np.random.rand(m, r))
        P = np.mat(np.random.rand(d, r))
        F = np.mat(np.random.rand(t, r))

        Y1 = 0
        Y2 = 0

        R1 = 0
        R2 = 0

        Z1 = 0
        Z2 = 0

        X1 = np.mat(tl.unfold(X, 0))
        X2 = np.mat(tl.unfold(X, 1))
        X3 = np.mat(tl.unfold(X, 2))

        I = np.eye(r)
        U = np.diagflat(S1.sum(1)) - S1
        V = np.diagflat(S2.sum(1)) - S2

        for i in range(max_iter):
            #print(f"iter\t{i + 1}\t\tbegin")
            output_X_old = tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, F])).T), 0, X.shape)

            O_1 = C.T * C
            O_2 = P.T * P
            M = CG(0, mu * O_1, O_1, mu * C.T * S1 * C, lam, 0.01, 200)
            N = CG(0, eta * O_2, O_2, eta * P.T * S2 * P, lam, 0.01, 200)

            C_P = np.mat(tl.tenalg.khatri_rao([C, P]))
            F = X3 * C_P * np.linalg.inv(C_P.T * C_P + 0.5 * lam * I)

            CM = C * M
            J1 = (mu * S1 * CM + rho_1 * C + R1) * np.linalg.inv(mu * CM.T * CM + rho_1 * I)
            # R1 = R1 + rho_1 * (C - J1)
            P_F = np.mat(tl.tenalg.khatri_rao([P, F]))
            D = np.diag(0.5 * np.linalg.norm(C.T, axis=1))
            C = (X1 * P_F + mu * S1 * J1 * M.T + rho_1 * J1 - R1 + xita * Z1 + Y1) * \
                np.linalg.inv(P_F.T * P_F + mu * M * J1.T * J1 * M.T + rho_1 * I + lam * I + lam * D * I + xita * I)
            Z1 = np.linalg.inv(xita * np.eye(m) + alpha * U) * (xita * C - Y1)  ## C(t) is not C(t+1)

            R1 = R1 + rho_1 * (C - J1)
            Y1 = Y1 + xita * (Z1 - C)
            rho_1 = rho_1 * 1.1

            PN = P * N
            J2 = (eta * S2 * PN + rho_2 * P + R2) * np.linalg.inv(eta * PN.T * PN + rho_2 * I)
            # R2 = R2 + rho_2 * (P - J2)
            C_F = np.mat(tl.tenalg.khatri_rao([C, F]))
            Q = np.diag(0.5 * np.linalg.norm(P.T, axis=1))
            P = (X2 * C_F + eta * S2 * J2 * N.T + rho_2 * J2 - R2 + xita * Z2 + Y2) * \
                np.linalg.inv(C_F.T * C_F + eta * N * J2.T * J2 * N.T + rho_2 * I + lam * I + lam * Q * I + xita * I)
            Z2 = np.linalg.inv(xita * np.eye(d) + beta * V) * (xita * P - Y2)

            R2 = R2 + rho_2 * (P - J2)
            Y2 = Y2 + xita * (Z2 - P)
            rho_2 = rho_2 * 1.1
            xita = xita * 1.1

            output_X = tl.fold(np.array(F * np.mat(tl.tenalg.khatri_rao([C, P])).T), 2, X.shape)
            err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
            #print(err)
            if err < tol:
                # print(i)
                break

        predict_X = np.array(tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, F])).T), 0, X.shape))

        return predict_X, C, P, F

    def TFAI_CP_within_mod(self, X, S_m, S_d, r=3, mu=0.125, eta=0.25, alpha=0.25, beta=1.0, lam=0.1, tol=1e-7,
                           max_iter=500, seed=2024):
        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]

        # initialization
        np.random.seed(2024)
        C = np.mat(np.random.rand(m, r))
        P = np.mat(np.random.rand(d, r))
        D = np.mat(np.random.rand(t, r))

        X_1 = np.mat(tl.unfold(X, 0))
        X_2 = np.mat(tl.unfold(X, 1))
        X_3 = np.mat(tl.unfold(X, 2))
        D_C = np.diagflat(S_m.sum(1))
        D_P = np.diagflat(S_d.sum(1))
        L_C = D_C - S_m
        L_P = D_P - S_d

        for i in range(max_iter):
            #print(f"iter\t{i}\tbegin")
            G = np.mat(tl.tenalg.khatri_rao([P, D]))
            output_X_old = tl.fold(np.array(C * G.T), 0, X.shape)

            C = np.mat(
                sp.linalg.solve_sylvester(np.array(alpha * L_C + lam * np.mat(np.eye(m))), np.array(G.T * G), X_1 * G))
            U = np.mat(tl.tenalg.khatri_rao([C, D]))
            P = np.mat(
                sp.linalg.solve_sylvester(np.array(beta * L_P + lam * np.mat(np.eye(d))), np.array(U.T * U), X_2 * U))
            B = np.mat(tl.tenalg.khatri_rao([C, P]))
            D = X_3 * B * np.linalg.inv(B.T * B + lam * np.eye(r))

            output_X = tl.fold(np.array(D * B.T), 2, X.shape)
            err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
            if err < tol:
                #print(i)
                break
        predict_X = tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape)
        return predict_X

    def TDRC_md(self, X, S_m, S_d, r=4, mu=0.125, eta=0.25, alpha=0.1, beta=0.1, lam=0.001, tol=1e-6, max_iter=500):
        '''
        S_d: S2
        S_m: S1
        '''

        def CG(X_initial, A, B, D, mu, tol, max_iter):

            X = X_initial
            R = D - A * X * B - mu * X
            P = np.array(R, copy=True)

            for i in range(max_iter):
                R_norm = np.trace(R * R.T)
                Q = A * P * B + mu * P
                alpha = R_norm / np.trace(Q * P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = np.linalg.norm(R)
                if err < tol:
                    # print("CG convergence: iter = %d" % i)
                    break

                beta = np.trace(R * R.T) / R_norm
                P = R + beta * P

            return X

        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]

        # initialization
        rho_1 = 1
        rho_2 = 1
        np.random.seed(2024)
        C = np.mat(np.random.rand(m, r))
        # P = np.mat(np.random.rand(d, r)) #第二维 疾病
        # D = np.mat(np.random.rand(t, r))
        P = np.mat(np.random.rand(t, r))  # 第三维 疾病
        D = np.mat(np.random.rand(d, r))  # 第二维

        Y_1 = 0
        Y_2 = 0

        X_1 = np.mat(tl.unfold(X, 0))
        X_2 = np.mat(tl.unfold(X, 1))
        X_3 = np.mat(tl.unfold(X, 2))

        for i in range(max_iter):
            #print(f"iter\t{i + 1}\t\tbegin")
            G = np.mat(tl.tenalg.khatri_rao([P, D]))
            output_X_old = tl.fold(np.array(C * G.T), 0, X.shape)

            O_1 = C.T * C
            O_2 = P.T * P

            M_2 = CG(0, alpha * O_1, O_1, alpha * C.T * S_m * C, lam, 0.01, 200)

            M_3 = CG(0, beta * O_2, O_2, beta * P.T * S_d * P, lam, 0.01, 200)

            K = np.mat(np.eye(r))

            F = C * M_2
            J = (alpha * S_m.T * F + rho_1 * C + Y_1) * np.linalg.inv(alpha * F.T * F + rho_1 * np.eye(r))
            Q = M_2 * J.T
            C = (X_1 * G + alpha * S_m * Q.T + rho_1 * J - Y_1) * np.linalg.inv(
                G.T * G + alpha * Q * Q.T + rho_1 * np.eye(r))

            R = P * M_3
            W = (beta * S_d.T * R + rho_2 * P + Y_2) * np.linalg.inv(beta * R.T * R + rho_2 * np.eye(r))
            Y_1 = Y_1 + rho_1 * (C - J)
            rho_1 = rho_1 * 1.1

            U = np.mat(tl.tenalg.khatri_rao([C, D]))
            Z = M_3 * W.T
            P = (X_3 * U + beta * S_d * Z.T + rho_2 * W - Y_2) * np.linalg.inv(
                U.T * U + beta * Z * Z.T + rho_2 * np.eye(r))
            Y_2 = Y_2 + rho_2 * (P - W)
            rho_2 = rho_2 * 1.1

            B = np.mat(tl.tenalg.khatri_rao([C, P]))
            D = X_2 * B * np.linalg.inv(B.T * B + lam * np.eye(r))

            output_X = tl.fold(np.array(D * B.T), 2, X.shape)
            err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
            # print(err)
            if err < tol:
                # print(i)
                break

        predict_X = np.array(tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape))

        return predict_X

    def TDRC_mm(self, X, S_m, S_d, r=4, mu=0.125, eta=0.25, alpha=0.1, beta=0.1, lam=0.001, tol=1e-6, max_iter=500):
        '''
        S_d: S2
        S_m: S1
        '''

        def CG(X_initial, A, B, D, mu, tol, max_iter):

            X = X_initial
            R = D - A * X * B - mu * X
            P = np.array(R, copy=True)

            for i in range(max_iter):
                R_norm = np.trace(R * R.T)
                Q = A * P * B + mu * P
                alpha = R_norm / np.trace(Q * P.T)
                X = X + alpha * P
                R = R - alpha * Q
                err = np.linalg.norm(R)
                if err < tol:
                    # print("CG convergence: iter = %d" % i)
                    break

                beta = np.trace(R * R.T) / R_norm
                P = R + beta * P

            return X

        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]

        # initialization
        rho_1 = 1
        rho_2 = 1
        np.random.seed(2024)
        C = np.mat(np.random.rand(m, r))
        P = np.mat(np.random.rand(d, r))
        D = np.mat(np.random.rand(t, r))

        Y_1 = 0
        Y_2 = 0

        X_1 = np.mat(tl.unfold(X, 0))
        X_2 = np.mat(tl.unfold(X, 1))
        X_3 = np.mat(tl.unfold(X, 2))

        for i in range(max_iter):
            #print(f"iter\t{i + 1}\t\tbegin")
            G = np.mat(tl.tenalg.khatri_rao([P, D]))
            output_X_old = tl.fold(np.array(C * G.T), 0, X.shape)

            O_1 = C.T * C
            O_2 = P.T * P

            M_2 = CG(0, alpha * O_1, O_1, alpha * C.T * S_m * C, lam, 0.01, 200)

            M_3 = CG(0, beta * O_2, O_2, beta * P.T * S_d * P, lam, 0.01, 200)

            K = np.mat(np.eye(r))

            F = C * M_2
            J = (alpha * S_m.T * F + rho_1 * C + Y_1) * np.linalg.inv(alpha * F.T * F + rho_1 * np.eye(r))
            Q = M_2 * J.T
            C = (X_1 * G + alpha * S_m * Q.T + rho_1 * J - Y_1) * np.linalg.inv(
                G.T * G + alpha * Q * Q.T + rho_1 * np.eye(r))

            R = P * M_3
            W = (beta * S_d.T * R + rho_2 * P + Y_2) * np.linalg.inv(beta * R.T * R + rho_2 * np.eye(r))
            Y_1 = Y_1 + rho_1 * (C - J)
            rho_1 = rho_1 * 1.1

            U = np.mat(tl.tenalg.khatri_rao([C, D]))
            Z = M_3 * W.T
            P = (X_2 * U + beta * S_d * Z.T + rho_2 * W - Y_2) * np.linalg.inv(
                U.T * U + beta * Z * Z.T + rho_2 * np.eye(r))
            Y_2 = Y_2 + rho_2 * (P - W)
            rho_2 = rho_2 * 1.1

            B = np.mat(tl.tenalg.khatri_rao([C, P]))
            D = X_3 * B * np.linalg.inv(B.T * B + lam * np.eye(r))

            output_X = tl.fold(np.array(D * B.T), 2, X.shape)
            err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
            # print(err)
            if err < tol:
                # print(i)
                break

        predict_X = np.array(tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape))

        return predict_X

    def CP(self, X, S_d=None, S_m=None, r=30, mu=0.5, alpha=0.5, eta=0.2, beta=0.5, lam=0.001, tol=1e-4, max_iter=100):
        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]

        np.random.seed(2024)
        C = np.mat(np.random.rand(m, r))
        P = np.mat(np.random.rand(d, r))
        D = np.mat(np.random.rand(t, r))

        X_1 = np.mat(tl.unfold(X, 0))
        X_2 = np.mat(tl.unfold(X, 1))
        X_3 = np.mat(tl.unfold(X, 2))

        for i in range(max_iter):
            #print(f"iter\t{i}\t\tbegin")
            output_X_old = tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape)

            B = np.mat(tl.tenalg.khatri_rao([C, P]))
            D = X_3 * B * np.linalg.inv(B.T * B)

            U = np.mat(tl.tenalg.khatri_rao([C, D]))
            P = X_2 * U * np.linalg.inv(U.T * U)

            G = np.mat(tl.tenalg.khatri_rao([P, D]))
            C = X_1 * G * np.linalg.inv(G.T * G)

            output_X = tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape)
            err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)

            if err < tol:
                break

        predict_X = np.array(tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape))

        return predict_X

    def CTF_DNN(self, X, S1, S2, r=4, mu=0.125, eta=0.25, alpha=0.1, beta=0.1, lam=0.001, tol=1e-6, max_iter=500,
                net={"epoch": 300, "batch_size": 1000, "hids_size": [256, 256, 128], "lr": 0.0001, "weight_decay": 5e-4,
                     'k': 300}):
        def CTF(X, S1, S2, r=4, mu=0.125, eta=0.25, alpha=0.1, beta=0.1, lam=0.001, tol=1e-6, max_iter=500):

            def CG(X_initial, A, B, D, mu, tol, max_iter):
                '''
                A: v U.T U
                B: V.T V
                D: v U.T O V
                '''
                X = X_initial
                R = D - A * X * B - mu * X
                P = np.array(R, copy=True)

                for i in range(max_iter):
                    R_norm = np.trace(R * R.T)
                    Q = A * P * B + mu * P
                    alpha = R_norm / np.trace(Q * P.T)
                    X = X + alpha * P
                    R = R - alpha * Q
                    err = np.linalg.norm(R)
                    if err < tol:
                        # print("CG convergence: iter = %d" % i)
                        break

                    beta = np.trace(R * R.T) / R_norm
                    P = R + beta * P

                return X

            m = X.shape[0]
            d = X.shape[1]
            t = X.shape[2]

            # initialization
            rho_1 = 1
            rho_2 = 1
            xita = 1
            np.random.seed(2024)
            C = np.mat(np.random.rand(m, r))
            P = np.mat(np.random.rand(d, r))
            F = np.mat(np.random.rand(t, r))

            Y1 = 0
            Y2 = 0

            R1 = 0
            R2 = 0

            Z1 = 0
            Z2 = 0

            X1 = np.mat(tl.unfold(X, 0))
            X2 = np.mat(tl.unfold(X, 1))
            X3 = np.mat(tl.unfold(X, 2))

            I = np.eye(r)
            U = np.diagflat(S1.sum(1)) - S1
            V = np.diagflat(S2.sum(1)) - S2

            for i in range(max_iter):
                #print(f"iter\t{i + 1}\t\tbegin")
                output_X_old = tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, F])).T), 0, X.shape)

                O_1 = C.T * C
                O_2 = P.T * P

                M = CG(0, mu * O_1, O_1, mu * C.T * S1 * C, lam, 0.01, 200)

                N = CG(0, eta * O_2, O_2, eta * P.T * S2 * P, lam, 0.01, 200)

                C_P = np.mat(tl.tenalg.khatri_rao([C, P]))
                F = X3 * C_P * np.linalg.inv(C_P.T * C_P + 0.5 * lam * I)

                CM = C * M
                J1 = (mu * S1 * CM + rho_1 * C + R1) * np.linalg.inv(mu * CM.T * CM + rho_1 * I)
                R1 = R1 + rho_1 * (C - J1)
                P_F = np.mat(tl.tenalg.khatri_rao([P, F]))
                D = np.diag(0.5 * np.linalg.norm(C.T, axis=1))
                C = (X1 * P_F + mu * S1 * J1 * M.T + rho_1 * J1 - R1 + xita * Z1 + Y1) * \
                    np.linalg.inv(P_F.T * P_F + mu * M * J1.T * J1 * M.T + rho_1 * I + lam * I + lam * D * I + xita * I)
                Z1 = np.linalg.inv(xita * np.eye(m) + alpha * U) * (xita * C - Y1)  ## C(t) is not C(t+1)
                Y1 = Y1 + xita * (Z1 - C)
                rho_1 = rho_1 * 1.1

                PN = P * N
                J2 = (eta * S2 * PN + rho_2 * P + R2) * np.linalg.inv(eta * PN.T * PN + rho_2 * I)
                R2 = R2 + rho_2 * (P - J2)
                C_F = np.mat(tl.tenalg.khatri_rao([C, F]))
                Q = np.diag(0.5 * np.linalg.norm(P.T, axis=1))
                P = (X2 * C_F + eta * S2 * J2 * N.T + rho_2 * J2 - R2 + xita * Z2 + Y2) * \
                    np.linalg.inv(
                        C_F.T * C_F + eta * N * J2.T * J2 * N.T + rho_2 * I + lam * I + lam * Q * I + xita * I)
                Z2 = np.linalg.inv(xita * np.eye(d) + beta * V) * (xita * P - Y2)
                Y2 = Y2 + xita * (Z2 - P)
                rho_2 = rho_2 * 1.1
                xita = xita * 1.1

                output_X = tl.fold(np.array(F * np.mat(tl.tenalg.khatri_rao([C, P])).T), 2, X.shape)
                err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
                # print(err)
                if err < tol:
                    # print(i)
                    break

            return C, P, F

        def set_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        set_seed(2024)

        C, P, F = CTF(X, S1, S2, r, mu, eta, alpha, beta, lam, tol, max_iter)
        print('CTF end!')
        # Y = net['label']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        Y = np.array(X, copy=True)
        x, y, z = Y.shape
        #print(Y.shape)
        C = torch.FloatTensor(C)
        P = torch.FloatTensor(P)
        F = torch.FloatTensor(F)
        print(C.shape,P.shape,F.shape)
        F = F.unsqueeze(0).expand(y, z, r)
        P = P.unsqueeze(1).expand(y, z, r)
        print(C.shape,P.shape,F.shape)
        PF = torch.cat((P, F), dim=2)
        print(PF.shape)
        PF = PF.reshape(y * z, 2 * r)
        print(PF.shape)
        PF = PF.unsqueeze(0).expand(x, y * z, 2 * r)
        print(PF.shape)
        C = C.unsqueeze(1).expand(x, y * z, r)
        print(C.shape)
        CPF = torch.cat((C, PF), dim=2)
        print(CPF.shape)
        CPF = CPF.reshape(x * y * z, 3 * r)
        print(CPF.shape)
        inputCPF = CPF
        print(inputCPF.shape)
        label = torch.FloatTensor(Y).to(device)
        label = label.reshape(x * y * z)
        print(label.shape)
        print(label)

        inputCPF_train = inputCPF[:len(label) // 5 * 4]
        inputCPF_test = inputCPF[len(label) // 5 * 4:]
        label_train = label[:len(label) // 5 * 4]
        label_test = label[len(label) // 5 * 4:]

        print(inputCPF_train.shape,label_train.shape)
        print(inputCPF_test.shape,label_test.shape)
        del C,P,F,PF,CPF

        dataset = CustomDataset(inputCPF_train, label_train)
        dataloader = DataLoader(dataset, batch_size=net['batch_size'], shuffle=True)
        del inputCPF_train, dataset, label_train

        print('Start Net:')
        dnn = NET(input_size=3 * r, hids_size=net['hids_size']).to(device)
        optimizer = optim.Adam(
            dnn.parameters(),
            lr=net['lr'],
            weight_decay=net['weight_decay']
        )
        criterion = nn.BCEWithLogitsLoss()

        loss_train_list = []
        loss_test_list = []
        for epoch in range(net['epoch']):
            #print(f"epoch\t{epoch + 1}\t\tbegin")
            dnn.train()
            loss_train_list_batch = []
            for batch, (input, y) in enumerate(dataloader):
                optimizer.zero_grad()

                output = dnn(input.to(device))
                output = torch.squeeze(output)
                loss = criterion(output, y.to(device))
                loss.backward()
                optimizer.step()

                # print(f"epoch\t{epoch + 1}\t\tbatch\t{batch + 1}\t\tloss\t{loss.item()}")
                loss_train_list_batch.append(loss.item())

            loss_train_list.append(np.mean(loss_train_list_batch))
            # print(f"{epoch+1}\t", end='')

            dnn.eval()
            with torch.no_grad():
                loss_test_list_batch = []
                for batch, (input, y) in enumerate(
                        DataLoader(CustomDataset(inputCPF_test, label_test), batch_size=net['batch_size'])):
                    output = dnn(input.to(device))
                    output = torch.squeeze(output)
                    loss = criterion(output, y.to(device))
                    loss_test_list_batch.append(loss.item())
                loss_test_list.append(np.mean(loss_test_list_batch))

        # ------------------------------------------------------------------------------------------------------------------
        # dnn.eval()
        # with torch.no_grad():
        #     predict_X = torch.empty(len(inputCPF))
        #     for batch, inputId in enumerate(DataLoader(IdDataset(len(inputCPF)), batch_size=net['batch_size'])):
        #         # print(f"epoch\t{epoch+1}\t\tbatch\t{batch+1}\t\tval")
        #         output = dnn(inputCPF[inputId])
        #         output = torch.squeeze(output)
        #         output = torch.sigmoid(output)
        #         predict_X[inputId] = output.to('cpu')
        #     predict_X = predict_X.reshape(Y.shape)
        #     predict_X = np.array(predict_X)
        #
        # metrics_tensor = np.zeros((1, 7))
        # for i in range(10):
        #     metrics_tensor = metrics_tensor + cv_tensor_model_evaluate(net['label'], predict_X, net['train_index'], i)
        #
        # result = np.around(metrics_tensor / 10, decimals=4)
        # print('\t'.join(map(str, result[0])))

        # ---------------------------------------------------------------------------------------------------------------------------

        #draw(loss_train_list, loss_test_list, str(net['k']) + '-loss.png')
        print('DNN Train end!')
        print('CTF-DDT Predicting!')
        dnn.eval()
        with torch.no_grad():
            predict_X = torch.empty(len(inputCPF))
            for batch, inputId in enumerate(DataLoader(IdDataset(len(inputCPF)), batch_size=net['batch_size'])):
                # print(f"batch\t{batch + 1}\t\tval")
                output = dnn(inputCPF[inputId].to(device))
                output = torch.squeeze(output)
                output = torch.sigmoid(output)
                predict_X[inputId] = output.to('cpu')
            predict_X = predict_X.reshape(Y.shape)
            predict_X = np.array(predict_X)

        return predict_X

    def __call__(self):

        return getattr(self, self.name, None)
