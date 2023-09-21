import numpy as np
import numpy.ma as ma


def snf(mtrx: np.array) -> np.array:
    """
    Returns the Smith Normal Form of a matrix of integer entries, which makes the linear transform 'diagonal' under base
    changes.
    """
    if mtrx.dtype != int:
        print(f'Please give a matrix with integer entries!')
        return
    shape = mtrx.shape
    num_loop = min(shape[0], shape[1])
    for loop in range(num_loop):    # Iterate along diagonal
        while (mtrx[loop+1:, loop] != 0).any() or (mtrx[loop, loop+1:] != 0).any():
            # Keep applying Gauss elimination, but also Euclidean arithmetic as we are working on integers.
            i, j = np.unravel_index(ma.masked_where(mtrx[loop:, loop:] == 0, np.abs(mtrx[loop:, loop:])).argmin(), mtrx[loop:, loop:].shape)
            mtrx[[i+loop, loop]] = mtrx[[loop, i+loop]]
            mtrx[:, [j+loop, loop]] = mtrx[:, [loop, j+loop]]
            for k in range(loop+1, shape[0]):
                mtrx[k] -= mtrx[k, loop]//mtrx[loop, loop]*mtrx[loop]
            for k in range(loop+1, shape[1]):
                mtrx[:, k] -= mtrx[loop, k]//mtrx[loop, loop]*mtrx[:, loop]
            if (mtrx[loop+1:, loop] == 0).all() and (mtrx[loop, loop+1:] == 0).all():
                new_entry = False
                m = 0
                for m in range(loop+1, shape[0]):
                    for n in range(loop+1, shape[1]):
                        if mtrx[m, n] % mtrx[loop, loop] != 0:
                            new_entry = True
                            break
                    else:
                        continue
                    break
                if new_entry:
                    mtrx[loop] += mtrx[m]
                else:
                    if mtrx[loop, loop] < 0:
                        mtrx[loop] = -mtrx[loop]
    return mtrx


def rank_smith(mtrx: np.array) -> int:
    """
    Computes the rank of a diagonal matrix.
    """
    diag = mtrx.diagonal()
    return len(diag[diag != 0])


def rank(mtrx: np.array) -> int:
    """
    Computes the rank of a matrix.
    """
    smith = snf(mtrx)
    return rank_smith(smith)


def homology(mtrx_in: np.array = None, mtrx_out: np.array = None) -> np.array:
    """
    Returns the homology of a chain complex, i.e. two composable linear maps where the image of the incoming matrix
    lies in the kernel of the outgoing matrix.
    See https://eric-bunch.github.io/blog/calculating_homology_of_simplicial_complex, part Calculating Homology using
    Smith Normal Form, a formular is there.
    """
    if type(mtrx_in) != np.ndarray and type(mtrx_out) != np.ndarray:  # Case 1: both trivial maps
        return None
    if type(mtrx_in) != np.ndarray:  # Case 2: only incoming is trivial
        return np.zeros((len(mtrx_out[1]) - rank(mtrx_out)), dtype=int)
    smith = snf(mtrx_in)
    diag_in = smith.diagonal()
    torsion = diag_in[diag_in > 1]
    rank_in = rank_smith(smith)
    if type(mtrx_out) != np.ndarray:  # Case 3: only outgoing is trivial
        free = np.zeros((mtrx_in.shape[0] - rank_in), dtype=int)
        return np.hstack((torsion, free))
    else:  # Case 4: both non-trivial
        if mtrx_in.shape[0] != mtrx_out.shape[1]:
            print(f'The linear maps are not composable!')
            return None
        free = np.zeros((mtrx_in.shape[0] - rank_in - rank(mtrx_out)), dtype=int)
        return np.hstack((torsion, free))
