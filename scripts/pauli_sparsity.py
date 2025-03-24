import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import qiskit


eps = np.finfo(float).eps


def pauli_representation(A):
    if A.size == 1:
        B = np.diag([A, 1.0])
    else:
        N = A.shape[0]
        n = int(np.ceil(np.log2(N)))
        B = np.eye(2**n)
        B[:N, :N] = A
    C = qiskit.quantum_info.SparsePauliOp.from_operator(B, atol=eps, rtol=eps)
    return C


def pauli_truncate_abstol(A, tol):
    abscoeffs = np.abs(A.coeffs)
    inds = (abscoeffs > tol)
    return qiskit.quantum_info.SparsePauliOp(A.paulis[inds], A.coeffs[inds])


def pauli_truncate_reltol(A, tol):
    abscoeffs2 = np.abs(A.coeffs)**2
    inds = np.argsort(-abscoeffs2)
    ratios = np.cumsum(abscoeffs2[inds]) / np.sum(abscoeffs2)
    if ratios.size < 2 or ratios[-2] < 1.0 - tol:
        cutoff = ratios.size
    else:
        cutoff = np.argmin(ratios >= 1 - tol) + 1
    return qiskit.quantum_info.SparsePauliOp(A.paulis[inds[:cutoff]],
                                             A.coeffs[inds[:cutoff]])


def add_distribution(ax, num_qubit, num_term, shift, width, color, label):
    for n in range(num_qubit.min(), num_qubit.max() + 1):
        inds = (num_qubit == n)
        yavg = num_term[inds].mean()
        ymin = num_term[inds].min()
        ymax = num_term[inds].max()

        yq1, yq3 = np.quantile(num_term[inds], [0.25, 0.75])

        if n == num_qubit.min():
            ax.plot([n + shift, n + shift], [ymin, ymax], color=color,
                    linewidth=1, label=label)
        else:
            ax.plot([n + shift, n + shift], [ymin, ymax], color=color,
                    linewidth=1)

        xcoords = [n + shift - 0.5 * width, n + shift + 0.5 * width]
        ax.plot(xcoords, [ymin, ymin], color=color, linewidth=1.0)
        ax.plot(xcoords, [ymax, ymax], color=color, linewidth=1.0)
        ax.plot(xcoords, [yavg, yavg], color=color, linewidth=1.5)

        ax.add_patch(mpl.patches.Rectangle((n + shift - 0.4 * width, yq1),
                                           width=0.8 * width,
                                           height=yq3 - yq1,
                                           linewidth=0.5,
                                           edgecolor=color,
                                           facecolor='none'))


def pauli_sparsity(data_dir):
    data_file = os.path.join(data_dir, 'pauli_sparsity.txt')
    plot_file = os.path.join(data_dir, 'pauli_sparsity.pdf')

    if os.path.exists(data_file):
        data = np.loadtxt(data_file, dtype=int)
        size = data[:, 0]
        num_qubit = data[:, 1]
        num_term = data[:, 2:]
    else:
        df = pd.read_csv(os.path.join(data_dir, 'hessian_sizes.txt'),
                         sep="\\s+",
                         names=["stem", "size"])
        num_hessian = df.shape[0]

        size = np.zeros((num_hessian,), dtype=np.int_)
        num_qubit = np.zeros((num_hessian,), dtype=np.int_)
        num_term = np.zeros((num_hessian, 7), dtype=np.int_)

        reporting_interval = (num_hessian + 19) // 20

        with open(data_file, 'w') as f:
            for i in range(num_hessian):
                size[i] = df['size'][i]
                A = np.loadtxt(os.path.join(data_dir,
                                            'solver_logs',
                                            df['stem'][i] + '_A.txt'))

                B = pauli_representation(A)
                num_qubit[i] = B.num_qubits
                num_term[i, 0] = B.size

                C = pauli_truncate_abstol(B, 1.0e-15)
                num_term[i, 1] = C.size
                C = pauli_truncate_abstol(B, 1.0e-09)
                num_term[i, 2] = C.size
                C = pauli_truncate_abstol(B, 1.0e-03)
                num_term[i, 3] = C.size

                C = pauli_truncate_reltol(B, 1.0e-15)
                num_term[i, 4] = C.size
                C = pauli_truncate_reltol(B, 1.0e-12)
                num_term[i, 5] = C.size
                C = pauli_truncate_reltol(B, 1.0e-09)
                num_term[i, 6] = C.size

                f.write(f'{size[i]} {num_qubit[i]} {num_term[i, 0]} ' +
                        f'{num_term[i, 1]} {num_term[i, 2]} ' +
                        f'{num_term[i, 3]} {num_term[i, 4]} ' +
                        f'{num_term[i, 5]} {num_term[i, 6]}\n')

                if (i + 1) % reporting_interval == 0:
                    percentage = 100.0 * (i + 1) / num_hessian
                    print(f'{percentage:6.2f}% done')

            if num_hessian % reporting_interval != 0:
                print('100.00% done')

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    n = np.arange(num_qubit.min(), num_qubit.max() + 1)
    M = (4**n + 2**n) / 2
    nn = np.reshape(np.vstack((n - 0.5, n + 0.5)), shape=(2 * n.size,),
                    order='F')
    MM = np.reshape(np.vstack((M, M)), shape=(2 * M.size,), order='F')
    ax.semilogy(nn, MM, base=2, color='tab:gray',
                label=r'$M = \frac{1}{2} (4^n + 2^n)$')
    for i in range(n.size):
        count = np.sum(num_qubit == n[i])
        ax.text(n[i], 1.1 * M[i], str(count), ha='center', va='bottom')

    add_distribution(ax, num_qubit, num_term[:, 0], shift=0.0, width=0.15,
                     color='tab:blue', label='no truncation')
    add_distribution(ax, num_qubit, num_term[:, 3], shift=-0.2, width=0.15,
                     color='tab:orange',
                     label=r'truncate($\tau_{\rm abs} = 10^{-3}$)')
    add_distribution(ax, num_qubit, num_term[:, 5], shift=0.2, width=0.15,
                     color='tab:green',
                     label=r'truncate($\tau_{\rm rel} = 10^{-12}$)')
    ax.set_xticks(np.arange(num_qubit.min(), num_qubit.max() + 1))
    ax.set_xlabel(r'Number of Qubits, $n$')
    ax.set_ylabel(r'Number of Pauli Strings, $M$')
    ax.legend(loc='best')

    plt.tight_layout()
    fig.savefig(plot_file)
    plt.close(fig)


if __name__ == '__main__':
    plt.rc('text', usetex=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='directory containing simulation results')
    args = parser.parse_args()

    pauli_sparsity(args.data_dir)
