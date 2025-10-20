import qutip as qt

def dual_rail_H(
    N1=10, N2=10, Nc=20,
    w1=5.0, w2=5.2, wc=7.0,           # ω_λ
    k1=-0.25, k2=-0.25, kc=0.0,       # Kerr α_λ (renamed: k*)
    g1=0.050, g2=0.050,               # g_j
    g12=0.010                         # g_12
):

    a1_op = qt.destroy(N1)
    a2_op = qt.destroy(N2)
    ac_op = qt.destroy(Nc)

    I1, I2, Ic = qt.qeye(N1), qt.qeye(N2), qt.qeye(Nc)

    b1  = qt.tensor(a1_op, I2, Ic)
    b2  = qt.tensor(I1, a2_op, Ic)
    bc  = qt.tensor(I1, I2, ac_op)

    b1d, b2d, bcd = b1.dag(), b2.dag(), bc.dag()

    H_loc  = w1 * (b1d * b1) + (k1/2.0) * (b1d * b1d * b1 * b1)
    H_loc += w2 * (b2d * b2) + (k2/2.0) * (b2d * b2d * b2 * b2)
    H_loc += wc * (bcd * bc)
    if kc != 0.0:
        H_loc += (kc/2.0) * (bcd * bcd * bc * bc) 

    H_1c = g1 * (b1d*bc + b1*bcd - b1d*bcd - b1*bc)
    H_2c = g2 * (b2d*bc + b2*bcd - b2d*bcd - b2*bc)
    H_12 = g12 * (b1d*b2 + b1*b2d - b1d*b2d - b1*b2)

    return H_loc + H_1c + H_2c + H_12


if __name__ == "__main__":
    H = dual_rail_H(N1=8, N2=8, Nc=20, w1=5.0, w2=5.3, wc=7.2, k1=-0.25, k2=-0.27, kc=0.0,
                    g1=0.06, g2=0.055, g12=0.012)
    print("Shape:", H.shape)
    print("Hermitian:", (H - H.dag()).norm() < 1e-12)
