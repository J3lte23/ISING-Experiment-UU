import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

#Parameters
n = 2
N = n**2
kB = 1.0

T = 1.0
steps = 81
n_repeats = 40

n_therm = 50000
n_measure = 100000
stride = N

#Scan coupling around the expected critical value
J_c_exact = 0.5 * kB * T * np.log(1 + np.sqrt(2))
J_values = np.linspace(0.9, 1.1, steps) * J_c_exact


def total_energy(state, J):
    return -J * np.sum(
        state * (
            np.roll(state, -1, axis=0)
            + np.roll(state, -1, axis=1)
        )
    )


def change_spin(state, E, T, J, rng):
    x = rng.integers(n)
    y = rng.integers(n)

    neighbours = (
        state[(x + 1) % n, y]
        + state[(x - 1) % n, y]
        + state[x, (y + 1) % n]
        + state[x, (y - 1) % n]
    )

    dE = 2 * J * state[x, y] * neighbours

    if dE <= 0 or rng.random() < np.exp(-dE / (kB * T)):
        state[x, y] *= -1
        E += dE

    return E


def simulate_one(task):
    J, seed = task
    rng = np.random.default_rng(seed)

    state = 2 * rng.integers(0, 2, size=(n, n)) - 1
    E = total_energy(state, J)

    #Thermalization
    for i in range(n_therm):
        E = change_spin(state, E, T, J, rng)

    energies = []
    magnetizations = []

    #Measurements
    for step in range(n_measure):
        E = change_spin(state, E, T, J, rng)

        if (step + 1) % stride == 0:
            energies.append(E)
            magnetizations.append(np.mean(state))

    energies = np.asarray(energies)
    magnetizations = np.asarray(magnetizations)

    mean_E = np.mean(energies) / N
    mean_abs_m = np.mean(np.abs(magnetizations))

    #Heat capacity per spin, in units of kB
    C = (
        np.mean(energies**2) - np.mean(energies)**2
    ) / (N * kB * T**2)

    return J, mean_E, mean_abs_m, C


def main():
    seed_sequence = np.random.SeedSequence(100)
    seeds = seed_sequence.generate_state(
        steps * n_repeats, dtype=np.uint64
    )

    tasks = [
        (J, int(seeds[i * n_repeats + rep]))
        for i, J in enumerate(J_values)
        for rep in range(n_repeats)
    ]

    n_workers = min(os.cpu_count() or 1, len(tasks))

    print(f"Fixed temperature T = {T}")
    print(f"Exact critical coupling J_c = {J_c_exact:.8f}")
    print(f"Using {n_workers} worker processes")
    print(f"Running {len(tasks)} independent simulations")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(simulate_one, tasks))

    energy = np.zeros(steps)
    energy_err = np.zeros(steps)
    mag = np.zeros(steps)
    mag_err = np.zeros(steps)
    heat_cap = np.zeros(steps)
    heat_cap_err = np.zeros(steps)

    for i in range(steps):
        group = results[i * n_repeats:(i + 1) * n_repeats]

        e = np.array([r[1] for r in group])
        m = np.array([r[2] for r in group])
        c = np.array([r[3] for r in group])

        energy[i] = np.mean(e)
        energy_err[i] = np.std(e, ddof=1) / np.sqrt(n_repeats)

        mag[i] = np.mean(m)
        mag_err[i] = np.std(m, ddof=1) / np.sqrt(n_repeats)

        heat_cap[i] = np.mean(c)
        heat_cap_err[i] = np.std(c, ddof=1) / np.sqrt(n_repeats)

    #Finite-size estimate: coupling at maximum measured heat capacity
    i_peak = np.argmax(heat_cap)
    J_c_est = J_values[i_peak]

    print(f"Estimated J_c from heat-capacity peak = {J_c_est:.8f}")
    print(f"Exact J_c = {J_c_exact:.8f}")
    print(f"Difference = {J_c_est - J_c_exact:.8f}")

    J_red = J_values / J_c_exact

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].errorbar(
        J_red, energy, yerr=energy_err,
        fmt="o", capsize=3, label="Monte Carlo"
    )
    axes[0].axvline(1, color="red", linestyle="--", label=r"Exact $J_c$")
    axes[0].set(
        xlabel=r"$J/J_c$", ylabel=r"$\langle E\rangle/N$",
        title="Energy"
    )
    axes[0].legend(fontsize=8)

    axes[1].errorbar(
        J_red, mag, yerr=mag_err,
        fmt="o", capsize=3, label="Monte Carlo"
    )
    axes[1].axvline(1, color="red", linestyle="--", label=r"Exact $J_c$")
    axes[1].set(
        xlabel=r"$J/J_c$",
        title="Absolute magnetization"
    )
    axes[1].legend(fontsize=8)

    axes[2].errorbar(
        J_red, heat_cap, yerr=heat_cap_err,
        fmt="o", capsize=3, label="Monte Carlo"
    )
    axes[2].axvline(
        1, color="red", linestyle="--", label=r"Exact $J_c$"
    )
    axes[2].axvline(
        J_c_est / J_c_exact, color="black", linestyle=":",
        label=r"Heat-capacity peak"
    )
    axes[2].set(
        xlabel=r"$J/J_c$", ylabel=r"$C/(Nk_B)$",
        title="Heat capacity"
    )
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from numba import njit


#Parameters
kB = 1

#Coupling scan: fixed temperature
T_fixed = 1
J_c_exact = 0.5 * kB * T_fixed * np.log(1 + np.sqrt(2))

#Temperature scan: fixed coupling
J_fixed = 1
T_c_exact = 2 * J_fixed / (
    kB * np.log(1 + np.sqrt(2.0))
)

#Lattice sizes
L_values = [16, 24, 32, 48, 64, 96]

#Number of scan points
steps = 41

#Coupling values around the critical coupling
J_values = np.linspace(0.94, 1.06, steps) * J_c_exact

#Temperature values around the critical temperature
T_values = np.linspace(0.94, 1.06, steps) * T_c_exact

#Monte Carlo parameters
n_repeats = 20
n_therm = 2000
n_measure = 3000
stride = 5

n_workers = min(os.cpu_count() or 1, 8)


#Numba-compiled Monte Carlo simulation for efficient running
@njit(cache=True)
def simulate_one_numba(
    L, J, T, seed, n_therm, n_measure, stride, kB
):
    np.random.seed(seed)

    N = L**2

    #Random initial configuration
    state = 2*np.random.randint(0, 2, (L,L)) - 1

    #Initial total energy, each bond counted once
    E = 0

    for x in range(L):
        for y in range(L):
            E += -J * state[x, y] * (
                state[(x + 1) % L, y]
                + state[x, (y + 1) % L]
            )

    # Metropolis probabilities
    p4 = np.exp(-4.0 * J / (kB * T))
    p8 = np.exp(-8.0 * J / (kB * T))

    #Thermalization sweeps
    for sweep in range(n_therm):
        for _ in range(N):
            x = np.random.randint(L)
            y = np.random.randint(L)

            neighbours = (
                state[(x + 1) % L, y]
                + state[(x - 1) % L, y]
                + state[x, (y + 1) % L]
                + state[x, (y - 1) % L]
            )

            dE = 2.0 * J * state[x, y] * neighbours

            if dE <= 0.0:
                state[x, y] *= -1
                E += dE
            elif dE == 4.0 * J:
                if np.random.random() < p4:
                    state[x, y] *= -1
                    E += dE
            else:
                if np.random.random() < p8:
                    state[x, y] *= -1
                    E += dE

    #Measurements
    n_samples = n_measure // stride

    energies = np.empty(n_samples)
    magnetizations = np.empty(n_samples)

    sample = 0

    for sweep in range(n_measure):
        for _ in range(N):
            x = np.random.randint(L)
            y = np.random.randint(L)

            neighbours = (
                state[(x + 1) % L, y]
                + state[(x - 1) % L, y]
                + state[x, (y + 1) % L]
                + state[x, (y - 1) % L]
            )

            dE = 2.0 * J * state[x, y] * neighbours

            if dE <= 0.0:
                state[x, y] *= -1
                E += dE
            elif dE == 4.0 * J:
                if np.random.random() < p4:
                    state[x, y] *= -1
                    E += dE
            else:
                if np.random.random() < p8:
                    state[x, y] *= -1
                    E += dE

        if (sweep + 1) % stride == 0:
            energies[sample] = E

            total_m = 0
            for x in range(L):
                for y in range(L):
                    total_m += state[x, y]

            magnetizations[sample] = total_m / N
            sample += 1


    #Calculating necessary observables
    e_mean = np.mean(energies)
    e2_mean = np.mean(energies**2)

    m_mean = np.mean(magnetizations)
    m2 = np.mean(magnetizations**2)
    m4 = np.mean(magnetizations**4)

    energy_per_spin = e_mean / N
    abs_magnetization = np.mean(np.abs(magnetizations))

    #Heat capacity per spin, in units of kB
    heat_capacity = (
        e2_mean - e_mean**2
    ) / (N * kB * T**2)

    #Zero-field susceptibility per spin
    susceptibility = (
        N * (m2 - m_mean**2)
    ) / (kB * T)

    #Binder cumulant
    binder = np.nan
    if m2 > 0.0:
        binder = 1.0 - m4 / (3.0 * m2**2)

    return (
        L,
        energy_per_spin,
        abs_magnetization,
        heat_capacity,
        susceptibility,
        binder,
    )


def simulate_one(task):
    L, J, T, seed = task

    seed = 321

    return simulate_one_numba(
        L, J, T, seed,
        n_therm, n_measure, stride, kB
    )




observable_names = ["energy", "mag", "heat", "chi", "binder"]


def run_scan(scan_values, mode, fixed_value):
    seed_sequence = np.random.SeedSequence(
        100 if mode == "J" else 200
    )

    seeds = seed_sequence.generate_state(
        len(L_values) * len(scan_values) * n_repeats,
        dtype=np.uint64,
    )

    tasks = []
    idx = 0

    for L in L_values:
        for value in scan_values:
            for _ in range(n_repeats):

                if mode == "J":
                    J = value
                    T = fixed_value
                else:
                    J = fixed_value
                    T = value

                tasks.append((L, J, T, int(seeds[idx])))
                idx += 1

    print(f"\nRunning {mode}-scan")
    print(f"Lattice sizes: {L_values}")
    print(f"Scan points: {len(scan_values)}")
    print(f"Repeats per point: {n_repeats}")
    print(f"Independent simulations: {len(tasks)}")
    print(f"Worker processes: {min(n_workers, len(tasks))}")

    with ProcessPoolExecutor(
        max_workers=min(n_workers, len(tasks))
    ) as pool:
        results = list(pool.map(simulate_one, tasks))

    print(f"{mode}-scan completed.")

    data = {
        L: {
            name: np.zeros(len(scan_values))
            for name in observable_names
        }
        for L in L_values
    }

    errors = {
        L: {
            name: np.zeros(len(scan_values))
            for name in observable_names
        }
        for L in L_values
    }

    for iL, L in enumerate(L_values):
        for i_value in range(len(scan_values)):

            start = (
                iL * len(scan_values) + i_value
            ) * n_repeats

            group = results[start:start + n_repeats]

            for name, column in zip(
                observable_names, range(1, 6)
            ):
                values = np.array(
                    [result[column] for result in group],
                    dtype=float,
                )

                valid = values[np.isfinite(values)]

                if len(valid) > 0:
                    data[L][name][i_value] = np.mean(valid)
                else:
                    data[L][name][i_value] = np.nan

                if len(valid) > 1:
                    errors[L][name][i_value] = (
                        np.std(valid, ddof=1)
                        / np.sqrt(len(valid))
                    )
                else:
                    errors[L][name][i_value] = np.nan

    return data, errors


def interpolate_crossing(x, y1, y2):
    difference = y1 - y2

    for i in range(len(x) - 1):

        if not (
            np.isfinite(difference[i])
            and np.isfinite(difference[i + 1])
        ):
            continue

        if difference[i] == 0:
            return x[i]

        if difference[i] * difference[i + 1] < 0:
            return x[i] - difference[i] * (
                x[i + 1] - x[i]
            ) / (
                difference[i + 1] - difference[i]
            )

    return np.nan

def main():

    print("=" * 60)
    print("2D SQUARE-LATTICE ISING MODEL")
    print("=" * 60)

    print(f"Exact critical coupling at T={T_fixed}: "
          f"{J_c_exact:.10f}")

    print(f"Exact critical temperature at J={J_fixed}: "
          f"{T_c_exact:.10f}")


    data_J, errors_J = run_scan(
        J_values, mode="J", fixed_value=T_fixed
    )

    L_ref = 20
    heat = data_J[L_ref]["heat"]

    peak_idx = np.nanargmax(heat)
    J_peak = J_values[peak_idx]

    if 0 < peak_idx < len(J_values) - 1:
        x_fit = J_values[peak_idx - 1:peak_idx + 2]
        y_fit = heat[peak_idx - 1:peak_idx + 2]

        if np.all(np.isfinite(y_fit)):
            coeff = np.polyfit(x_fit, y_fit, 2)

            if coeff[0] < 0:
                J_vertex = -coeff[1] / (2.0 * coeff[0])

                if x_fit[0] <= J_vertex <= x_fit[-1]:
                    J_peak = J_vertex

    print(f"\nHeat-capacity peak for L={L_ref}:")
    print(f"J_peak = {J_peak:.10f}")
    print(f"Difference from exact = "
          f"{J_peak - J_c_exact:+.8f}")

    print("\nBinder cumulant crossings:")
    crossings = []

    for L1, L2 in zip(L_values[:-1], L_values[1:]):

        crossing = interpolate_crossing(
            J_values,
            data_J[L1]["binder"],
            data_J[L2]["binder"],
        )

        if np.isfinite(crossing):
            crossings.append(crossing)

            print(
                f"L={L1:2d}, {L2:2d}: "
                f"J_cross={crossing:.10f}, "
                f"difference={crossing - J_c_exact:+.8f}"
            )
        else:
            print(f"L={L1:2d}, {L2:2d}: no crossing found")

    if crossings:
        print(
            "Mean crossing estimate = "
            f"{np.mean(crossings):.10f}"
        )

    J_red = J_values / J_c_exact

    colors = plt.cm.viridis(
        np.linspace(0, 1, len(L_values))
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for color, L in zip(colors, L_values):

        axes[0, 0].errorbar(
            J_red, data_J[L]["heat"],
            yerr=errors_J[L]["heat"],
            fmt="o-", ms=3, capsize=2,
            color=color, label=f"L={L}",
        )

        axes[0, 1].errorbar(
            J_red, data_J[L]["chi"],
            yerr=errors_J[L]["chi"],
            fmt="o-", ms=3, capsize=2,
            color=color, label=f"L={L}",
        )

        axes[1, 0].errorbar(
            J_red, data_J[L]["binder"],
            yerr=errors_J[L]["binder"],
            fmt="o-", ms=3, capsize=2,
            color=color, label=f"L={L}",
        )

        axes[1, 1].errorbar(
            J_red, data_J[L]["mag"],
            yerr=errors_J[L]["mag"],
            fmt="o-", ms=3, capsize=2,
            color=color, label=f"L={L}",
        )

    for ax in axes.flat:
        ax.axvline(
            1.0, color="red", linestyle="--",
            label=r"Exact $J_c$",
        )
        ax.set_xlabel(r"$J/J_c$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel(r"$C/(Nk_B)$")
    axes[0, 0].set_title("Heat capacity")

    axes[0, 1].set_ylabel(r"$\chi$")
    axes[0, 1].set_title("Magnetic susceptibility")

    axes[1, 0].set_ylabel(r"$U_4$")
    axes[1, 0].set_title("Binder cumulant")

    axes[1, 1].set_ylabel(r"$\langle |m| \rangle$")
    axes[1, 1].set_title("Absolute magnetization")

    fig.suptitle(
        f"Coupling scan at T={T_fixed}",
        fontsize=14,
    )
    fig.tight_layout()

    data_T, errors_T = run_scan(
        T_values, mode="T", fixed_value=J_fixed
    )
    
    
    T_red = T_values / T_c_exact

    fig_T, axes_T = plt.subplots(1, 2, figsize=(12, 5))

    for color, L in zip(colors, L_values):

        axes_T[0].errorbar(
            T_red,
            data_T[L]["heat"],
            yerr=errors_T[L]["heat"],
            fmt="o-",
            ms=3,
            capsize=2,
            color=color,
            label=f"L={L}",
        )

        axes_T[1].errorbar(
            T_red,
            data_T[L]["mag"],
            yerr=errors_T[L]["mag"],
            fmt="o-",
            ms=3,
            capsize=2,
            color=color,
            label=f"L={L}",
        )

    for ax in axes_T:
        ax.axvline(
            1.0,
            color="red",
            linestyle="--",
            label=r"Exact $T_c$",
        )
        ax.set_xlabel(r"$T/T_c$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes_T[0].set_ylabel(r"$C/(Nk_B)$")
    axes_T[0].set_title("Heat capacity versus temperature")

    axes_T[1].set_ylabel(r"$\langle |m| \rangle$")
    axes_T[1].set_title("Absolute magnetization versus temperature")

    fig_T.suptitle(
        f"Temperature scan at J={J_fixed}",
        fontsize=14,
    )
    fig_T.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
