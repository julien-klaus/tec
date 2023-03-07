# @author: Julien Klaus

from timeit import default_timer as timer


import numpy as np
import torch
import tensorflow as tf


if __name__ == "__main__":

    
    for s in [25, 50, 100, 150, 200]:

        I,J,K,L,M,N = s, s, s, 10, 10, 10

        A = np.random.rand(I,J,K)
        Z = np.random.rand(L,M,N)
        B = np.random.rand(L,I)
        C = np.random.rand(M,J)
        D = np.random.rand(N,K)

        n_experiments = 5

        #### baseline
        tic = timer()
        r_for_loops = 0.0
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    s = 0.0
                    for l in range(L):
                        for m in range(M):
                            for n in range(N):
                                s += Z[l][m][n]*B[l][i]*C[m][j]*D[n][k]
                    r_for_loops += (A[i][j][k] - s)**2
        r_for_loops = r_for_loops**(0.5)
        toc = timer()
        t_baseline = (toc-tic) / n_experiments
        print(f"Baseline computed in {((toc-tic)/n_experiments):.5f}s")

        
        #### NumPy
        tic = timer()
        r_numpy = np.sqrt(np.einsum('ijk->',(A-np.einsum('lmn,li,mj,nk->ijk',Z,B,C,D))**2))
        toc = timer()
        t_numpy = (toc-tic) / n_experiments
        print(f"NumPy computed in {((toc-tic)/n_experiments):.5f}s")

        
        #### PyTorch
        tA = torch.Tensor(A)
        tZ = torch.Tensor(Z)
        tB = torch.Tensor(B)
        tC = torch.Tensor(C)
        tD = torch.Tensor(D)
        tic = timer()
        r_torch = torch.sqrt(torch.einsum('ijk->',(tA-torch.einsum('lmn,li,mj,nk->ijk',tZ,tB,tC,tD))**2))
        toc = timer()
        t_pytorch = (toc-tic) / n_experiments
        print(f"PyTorch computed in {((toc-tic)/n_experiments):.5f}s")


        #### TensorFlow
        tfA = tf.convert_to_tensor(A)
        tfZ = tf.convert_to_tensor(Z)
        tfB = tf.convert_to_tensor(B)
        tfC = tf.convert_to_tensor(C)
        tfD = tf.convert_to_tensor(D)
        tic = timer()
        r_tensorflow = tf.sqrt(tf.einsum('ijk->',(tfA-tf.einsum('lmn,li,mj,nk->ijk',tfZ,tfB,tfC,tfD))**2))
        toc = timer()
        t_tensorflow = (toc-tic) / n_experiments
        print(f"TensorFlow computed in {((toc-tic)/n_experiments):.5f}s")


        # correctness proof
        assert np.allclose(r_for_loops, r_numpy, r_torch, r_tensorflow)

        # print results
        print("Speed-Ups for s =", s, "(Baseline Python, NumPy, TensorFlow, PyTorch):", end=" ")
        print(", ".join([str(int(i)) for i in t_baseline / np.array([t_baseline, t_numpy, t_tensorflow, t_pytorch])]))