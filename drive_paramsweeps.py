import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import time


def get_dims_from_ops(ops):
    dims_left = ops[0].dims[0]
    def flatten_dims(dl):
        if len(dl) == 0: raise ValueError("Empty dims.")
        if isinstance(dl[0], list) or isinstance(dl[0], tuple):
            return [int(x[0]) for x in dl]
        else:
            return [int(x) for x in dl]
    Ns = flatten_dims(dims_left)
    if len(Ns) != 3: raise ValueError(f"Expected 3 subsystem dims, got {Ns}")
    N1, N2, Nc = Ns
    return N1, N2, Nc


def dual_rail_H(
    N1=10, N2=10, Nc=20,
    omega_1=5.0, omega_2=5.2, omega_c=7.0,
    k1=-0.25, k2=-0.25, kc=0.0,
    g1=0.050, g2=0.050,
    g12=0.010
):
    a1_op = qt.destroy(N1)
    a2_op = qt.destroy(N2)
    ac_op = qt.destroy(Nc)
    I1, I2, Ic = qt.qeye(N1), qt.qeye(N2), qt.qeye(Nc)
    b1 = qt.tensor(a1_op, I2, Ic)
    b2 = qt.tensor(I1, a2_op, Ic)
    bc = qt.tensor(I1, I2, ac_op)
    b1_dag, b2_dag, bc_dag = b1.dag(), b2.dag(), bc.dag()
    H_loc = omega_1 * (b1_dag * b1) + (k1/2.0) * (b1_dag * b1_dag * b1 * b1)
    H_loc += omega_2 * (b2_dag * b2) + (k2/2.0) * (b2_dag * b2_dag * b2 * b2)
    H_loc += omega_c * (bc_dag * bc)
    if kc != 0.0:
        H_loc += (kc/2.0) * (bc_dag * bc_dag * bc * bc)
    H_1c = g1 * (b1_dag*bc + b1*bc_dag - b1_dag*bc_dag - b1*bc)
    H_2c = g2 * (b2_dag*bc + b2*bc_dag - b2_dag*bc_dag - b2*bc)
    H_12 = g12 * (b1_dag*b2 + b1*b2_dag - b1_dag*b2_dag - b1*b2)
    return H_loc + H_1c + H_2c + H_12


def make_H(N1, N2, Nc, pars, use_rwa=True):
    w1, w2, wc = pars["omega_1"], pars["omega_2"], pars["omega_c"]
    k1, k2, kc = pars["k1"], pars["k2"], pars["kc"]
    g1, g2, g12 = pars["g1"], pars["g2"], pars["g12"]
    a1, a2, ac = qt.destroy(N1), qt.destroy(N2), qt.destroy(Nc)
    I1, I2, Ic = qt.qeye(N1), qt.qeye(N2), qt.qeye(Nc)
    b1 = qt.tensor(a1, I2, Ic)
    b2 = qt.tensor(I1, a2, Ic)
    bc = qt.tensor(I1, I2, ac)
    b1d, b2d, bcd = b1.dag(), b2.dag(), bc.dag()
    H = (w1*b1d*b1 + (k1/2)*b1d*b1d*b1*b1
       + w2*b2d*b2 + (k2/2)*b2d*b2d*b2*b2
       + wc*bcd*bc)
    if kc != 0.0:
        H += (kc/2)*bcd*bcd*bc*bc
    if use_rwa:
        H += g1*(b1d*bc + b1*bcd) + g2*(b2d*bc + b2*bcd) + g12*(b1d*b2 + b1*b2d)
    else:
        H += g1*(b1d*bc + b1*bcd - b1d*bcd - b1*bc)
        H += g2*(b2d*bc + b2*bcd - b2d*bcd - b2*bc)
        H += g12*(b1d*b2 + b1*b2d - b1d*b2d - b1*b2)
    return H, (b1, b2, bc)


def eig_and_labels(H, ops, max_states=120):
    N_hilbert = H.shape[0]
    n_eig = min(max_states, N_hilbert)
    
    evals, evecs = H.eigenstates(eigvals=n_eig)
    b1, b2, bc = ops
    n1_op, n2_op, nc_op = b1.dag()*b1, b2.dag()*b2, bc.dag()*bc
    labels = []
    for psi in evecs:
        labels.append((qt.expect(n1_op, psi), qt.expect(n2_op, psi), qt.expect(nc_op, psi)))
    return np.array(evals), evecs, np.array(labels)

def _find_state_ramped(evals, labels, q='q1', q_occ=0.0, n_c=0.0,
                       tol_q_start=0.25, tol_c_start=0.25, steps=5, inc=0.15):
    qidx = 0 if q == 'q1' else 1
    tol_q, tol_c = tol_q_start, tol_c_start
    for _ in range(steps):
        good = np.where((np.abs(labels[:, qidx] - q_occ) < tol_q) &
                        (np.abs(labels[:, 2]   - n_c)   < tol_c))[0]
        if len(good) > 0:
            return int(good[np.argmin(evals[good])])
        tol_q += inc
        tol_c += inc
    return None

def cavity_freq_given_qubit_state(evals, labels, qubit='q1', q_state='g', n=0):
    targ_q = 0.0 if q_state == 'g' else 1.0
    i_n   = _find_state_ramped(evals, labels, q=qubit, q_occ=targ_q, n_c=n)
    i_np1 = _find_state_ramped(evals, labels, q=qubit, q_occ=targ_q, n_c=n+1)
    if i_n is None or i_np1 is None:
        return None
    return float(evals[i_np1] - evals[i_n])

def dispersive_shifts(H, ops):
    N_hilbert = H.shape[0]
    n_eig = min(120, N_hilbert)
    
    evals, evecs, labels = eig_and_labels(H, ops, max_states=n_eig)
    wcg1 = cavity_freq_given_qubit_state(evals, labels, qubit='q1', q_state='g', n=0)
    wce1 = cavity_freq_given_qubit_state(evals, labels, qubit='q1', q_state='e', n=0)
    wcg2 = cavity_freq_given_qubit_state(evals, labels, qubit='q2', q_state='g', n=0)
    wce2 = cavity_freq_given_qubit_state(evals, labels, qubit='q2', q_state='e', n=0)
    chi1 = 0.5*(wce1 - wcg1) if (wcg1 is not None and wce1 is not None) else np.nan
    chi2 = 0.5*(wce2 - wcg2) if (wcg2 is not None and wce2 is not None) else np.nan
    return chi1, chi2, dict(wcg1=wcg1, wce1=wce1, wcg2=wcg2, wce2=wce2)


def sweep_drive_spectrum_ss(H_rwa, ops, pars, wd_span, eps=0.01, 
                            kappa_c=1e-3, gamma1=2e-4, gamma2=2e-4,
                            gphi1=1e-4, gphi2=1e-4, drive_on='c'):
    nc_list, P_c_list = [], []
    
    b1, b2, bc = ops
    n1_op, n2_op, nc_op = b1.dag()*b1, b2.dag()*b2, bc.dag()*bc

    w1, w2, wc = pars["omega_1"], pars["omega_2"], pars["omega_c"]
    k1, k2, kc = pars["k1"], pars["k2"], pars["kc"]
    g1, g2, g12 = pars["g1"], pars["g2"], pars["g12"]

    
    H_kerr = (k1/2.0) * n1_op * (n1_op - 1)
    H_kerr += (k2/2.0) * n2_op * (n2_op - 1)
    if kc != 0.0:
        H_kerr += (kc/2.0) * nc_op * (nc_op - 1)
    
    H_int = g1*(b1.dag()*bc + b1*bc.dag()) 
    H_int += g2*(b2.dag()*bc + b2*bc.dag())
    H_int += g12*(b1.dag()*b2 + b1*b2.dag())
    
    H_drive = 0
    if drive_on == 'c':
        H_drive = (eps/2.0) * (bc + bc.dag())
    elif drive_on == 'q1':
        H_drive = (eps/2.0) * (b1 + b1.dag())
    else:
        H_drive = (eps/2.0) * (b2 + b2.dag())

    H_static = H_kerr + H_int + H_drive

    c_ops = [
        np.sqrt(kappa_c)*bc,
        np.sqrt(gamma1)*b1,
        np.sqrt(gamma2)*b2,
        np.sqrt(2*gphi1)*n1_op,
        np.sqrt(2*gphi2)*n2_op
    ]

    for wd in wd_span:
        H_det = ((w1 - wd) * n1_op 
               + (w2 - wd) * n2_op 
               + (wc - wd) * nc_op)
        
        H_ss = H_static + H_det
        
        rho_ss = qt.steadystate(H_ss, c_ops)
        
        nc_ss = qt.expect(nc_op, rho_ss)
        
        nc_list.append(nc_ss)
        P_c_list.append(kappa_c * nc_ss * wd)

    return np.array(nc_list), np.array(P_c_list)



def sweep_detuning_and_compute_chi(N1, N2, Nc, base_pars, use_rwa=True,
                                   detuning_span=np.linspace(-1.0, +1.0, 21),
                                   qubit='q1'):
    chis = []
    deltas = []
    for d in detuning_span:
        pars = dict(base_pars)
        if qubit == 'q1':
            pars["omega_1"] = base_pars["omega_c"] + d
        else:
            pars["omega_2"] = base_pars["omega_c"] + d
        H, ops = make_H(N1, N2, Nc, pars, use_rwa=use_rwa)
        chi1, chi2, _ = dispersive_shifts(H, ops)
        chis.append(chi1 if qubit == 'q1' else chi2)
        deltas.append(d)
    return np.array(deltas), np.array(chis, dtype=float)


if __name__ == "__main__":
    t_start = time.time()
    
    base_pars = dict(
        omega_1=5.0, omega_2=5.3, omega_c=7.2,
        k1=-0.25, k2=-0.27, kc=0.0,
        g1=0.06, g2=0.055, g12=0.012
    )

    N1, N2, Nc = 3, 3, 10
    print(f"Running with dimensions N1={N1}, N2={N2}, Nc={Nc} (N_hilbert = {N1*N2*Nc})")


    H_full_orig = dual_rail_H(N1, N2, Nc, **base_pars)
    print("Original builder -> Shape:", H_full_orig.shape, "Hermitian:", (H_full_orig - H_full_orig.dag()).norm() < 1e-12)

    print("\nCalculating dispersive shifts...")
    t_chi_start = time.time()
    H_rwa, ops = make_H(N1, N2, Nc, base_pars, use_rwa=True)
    chi1, chi2, chi_info = dispersive_shifts(H_rwa, ops)
    if np.isfinite(chi1) and np.isfinite(chi2):
        print(f"Numerical dispersive shifts (RWA): chi1={chi1:.6f} GHz, chi2={chi2:.6f} GHz")
    else:
        print("Numerical dispersive shifts (RWA): chi1=", chi1, " chi2=", chi2)
    print("Conditional cavity freqs (GHz):", chi_info)
    print(f"...done in {time.time() - t_chi_start:.2f}s")


    detuning_span = np.linspace(-1.0, +1.0, 11)
    wd_span = np.linspace(base_pars["omega_c"] - 0.2, base_pars["omega_c"] + 0.2, 21)
    
    print("\nSweeping detuning vs. chi...")
    t_chi_sweep_start = time.time()
    d_q1, chi_q1 = sweep_detuning_and_compute_chi(N1, N2, Nc, base_pars, use_rwa=True,
                                                  detuning_span=detuning_span, qubit='q1')
    d_q2, chi_q2 = sweep_detuning_and_compute_chi(N1, N2, Nc, base_pars, use_rwa=True,
                                                  detuning_span=detuning_span, qubit='q2')
    print(f"...done in {time.time() - t_chi_sweep_start:.2f}s")

    plt.figure()
    plt.plot(d_q1, chi_q1, marker='o', label=r'$\chi_1(\Delta_1)$')
    plt.plot(d_q2, chi_q2, marker='s', label=r'$\chi_2(\Delta_2)$')
    plt.axhline(0, c='k', ls=':', lw=1)
    plt.xlabel('Detuning Δ (GHz)  [Δ = ω_q - ω_c]')
    plt.ylabel('χ (GHz)')
    plt.title(f'Dispersive shifts vs detuning (RWA, N={N1*N2*Nc})')
    plt.legend()
    plt.tight_layout()

    print("\nSweeping drive frequency vs. steady state...")
    t_drive_start = time.time()
    
    nc_vs_wd, P_c_vs_wd = sweep_drive_spectrum_ss(
        H_rwa, ops, base_pars, wd_span, 
        eps=0.01, kappa_c=1e-3, gamma1=2e-4, gamma2=2e-4,
        gphi1=1e-4, gphi2=1e-4, drive_on='c'
    )
    
    
    print(f"...done in {time.time() - t_drive_start:.2f}s")

    plt.figure()
    plt.plot(wd_span, nc_vs_wd, marker='.')
    plt.xlabel('Drive frequency ω_d (GHz)')
    plt.ylabel('<n_c> (steady-state)')
    plt.title('Driven cavity steady-state photons vs ω_d')
    plt.tight_layout()

    plt.figure()
    plt.plot(wd_span, P_c_vs_wd, marker='.')
    plt.xlabel('Drive frequency ω_d (GHz)')
    plt.ylabel('P_c ~ κ <n_c> ω_d (arb. units)')
    plt.title('Power to cavity bath vs ω_d')
    plt.tight_layout()

    t_end = time.time()
    print(f"\nTotal script execution time: {t_end - t_start:.2f} seconds")
    plt.show()