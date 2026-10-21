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
