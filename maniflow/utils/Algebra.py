import numpy as np
import numpy.ma as ma


def snf(mtrx: np.array) -> np.array:
    if mtrx.dtype != int:
        print(f'Please give a matrix with integer entries!')
        return
    shape = mtrx.shape
    num_loop = min(shape[0], shape[1])
    for loop in range(num_loop):
        while (mtrx[loop+1:, loop] != 0).any() or (mtrx[loop, loop+1:] != 0).any():
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


def rank(mtrx: np.array) -> int:
    smith = snf(mtrx)
    diag = smith.diagonal()
    return diag.argmin()


def homology(mtrx_in: np.array, mtrx_out: np.array) -> np.array:
    if mtrx_in.shape[0] != mtrx_out.shape[1]:
        print(f'The linear maps are not composable!')
        return None
    diag_in = snf(mtrx_in).diagonal()
    torsion = diag_in[:diag_in.argmin()]
    free = np.zeros((mtrx_in.shape[0]-rank(mtrx_in)-rank(mtrx_out)), dtype=int)
    return np.hstack((torsion, free))
