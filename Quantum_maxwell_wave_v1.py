# Quantum_maxwell_wave 



import random
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
COUPLING_GRID = (0.20, 0.45, 0.75, 1.10, 1.50)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < 30:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def extract_coupled_fields(
    draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_curr = draws[2:-1].astype(np.float64) / MAX_NUM
    e_curr = (draws[2:-1] - draws[1:-2]).astype(np.float64) / MAX_NUM
    e_prev = (draws[1:-2] - draws[:-3]).astype(np.float64) / MAX_NUM
    b_field = e_curr - e_prev
    y_next = draws[3:].astype(np.int64)
    return x_curr, e_curr, b_field, y_next


def fit_field_propagator(
    x_train: np.ndarray,
    e_train: np.ndarray,
    b_train: np.ndarray,
    y_train: np.ndarray,
    reg_lambda: float = 1e-2,
) -> np.ndarray:
    phi = np.hstack([
        np.ones((len(x_train), 1), dtype=np.float64),
        x_train,
        e_train,
        b_train,
        np.sin(np.pi * e_train),
        np.cos(np.pi * b_train),
    ])
    n_features = int(phi.shape.__getitem__(1))
    n_samples = len(y_train)
    target_dist = np.zeros((n_samples, NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    grid = np.arange(1, MAX_NUM + 1, dtype=np.float64)[None, :]
    for pos in range(NUM_POSITIONS):
        y_col = y_train[:, pos][:, None].astype(np.float64)
        envelope = np.exp(-0.5 * ((grid - y_col) / 1.25) ** 2)
        envelope /= np.sum(envelope, axis=1, keepdims=True)
        target_dist[:, pos, :] = envelope

    y_flat = target_dist.reshape(n_samples, NUM_POSITIONS * MAX_NUM)
    gram = phi.T @ phi + reg_lambda * np.eye(n_features, dtype=np.float64)
    weights = np.linalg.solve(gram, phi.T @ y_flat)
    return weights


def predict_field_amplitudes(
    weights: np.ndarray,
    x_step: np.ndarray,
    e_step: np.ndarray,
    b_step: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi_step = np.concatenate([
        np.array([1.0], dtype=np.float64),
        x_step,
        e_step,
        b_step,
        np.sin(np.pi * e_step),
        np.cos(np.pi * b_step),
    ])
    raw_pred = (phi_step @ weights).reshape(NUM_POSITIONS, MAX_NUM)
    e_amp = np.maximum(raw_pred, 1e-8)
    e_amp /= np.sum(e_amp, axis=1, keepdims=True)

    # Fazni profil po brojevima (1..39) izveden iz indukcionog polja b_step i brzine e_step
    num_axis = np.linspace(-1.0, 1.0, MAX_NUM, dtype=np.float64)[None, :]
    b_phase = np.pi * (b_step[:, None] * num_axis + 0.5 * e_step[:, None])
    return e_amp, b_phase


def evolve_maxwell_quantum_register(
    e_prob: np.ndarray,
    b_phase: np.ndarray,
    b_scalar: float,
    coupling: float,
) -> np.ndarray:
    init_state = np.zeros(STATE_DIM, dtype=np.complex128)
    amps = np.sqrt(e_prob) * np.exp(1j * b_phase)
    init_state[1 : MAX_NUM + 1] = amps
    norm = float(np.linalg.norm(init_state))
    if norm <= 1e-15:
        raise ValueError("Nulta norma kvantnog talasnog stanja.")
    init_state /= norm

    qc = QuantumCircuit(NUM_QUBITS)
    qc.initialize(init_state, range(NUM_QUBITS))

    for q in range(NUM_QUBITS):
        qc.rz(coupling * np.pi * b_scalar * (q + 1) / NUM_QUBITS, q)
        qc.ry(coupling * 0.5 * np.pi * b_scalar, q)

    for q in range(NUM_QUBITS - 1):
        qc.rxx(coupling * 0.35, q, q + 1)
        qc.rzz(coupling * 0.35, q, q + 1)

    sv = Statevector.from_instruction(qc)
    full_probs = sv.probabilities()
    valid_probs = full_probs[1 : MAX_NUM + 1]
    valid_probs = np.maximum(valid_probs, 1e-12)
    valid_probs /= np.sum(valid_probs)

    if abs(float(np.sum(valid_probs)) - 1.0) > 1e-6:
        raise ValueError("Greska u normalizaciji izlazne kvantne raspodele.")
    return valid_probs


def decode_combination(q_probs: np.ndarray) -> list[int]:
    available = np.ones(MAX_NUM, dtype=bool)
    selected = []
    for pos in range(NUM_POSITIONS):
        masked = np.where(available, q_probs[pos], -1.0)
        best_idx = int(np.argmax(masked))
        selected.append(best_idx + 1)
        available[best_idx] = False
    return sorted(selected)


def evaluate_range(
    weights: np.ndarray,
    x_arr: np.ndarray,
    e_arr: np.ndarray,
    b_arr: np.ndarray,
    y_arr: np.ndarray,
    start_idx: int,
    end_idx: int,
    coupling: float,
) -> float:
    total_nll = 0.0
    count = 0
    for t in range(start_idx, end_idx):
        e_prob, b_phase = predict_field_amplitudes(weights, x_arr[t], e_arr[t], b_arr[t])
        for pos in range(NUM_POSITIONS):
            q_dist = evolve_maxwell_quantum_register(
                e_prob[pos],
                b_phase[pos],
                float(b_arr[t, pos]),
                coupling,
            )
            target_num = int(y_arr[t, pos])
            p = max(float(q_dist[target_num - 1]), 1e-12)
            total_nll -= np.log(p)
        count += 1
    return total_nll / max(count, 1)


def process_csv(file_path: Path) -> None:
    draws = load_draws(file_path)
    x_all, e_all, b_all, y_all = extract_coupled_fields(draws)
    n_samples = len(y_all)

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    weights_train = fit_field_propagator(
        x_all[:train_end],
        e_all[:train_end],
        b_all[:train_end],
        y_all[:train_end],
    )

    best_coupling = float(next(iter(COUPLING_GRID)))
    best_val_nll = float("inf")

    for c_val in COUPLING_GRID:
        val_nll = evaluate_range(
            weights_train,
            x_all,
            e_all,
            b_all,
            y_all,
            train_end,
            val_end,
            float(c_val),
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_coupling = float(c_val)

    weights_train_val = fit_field_propagator(
        x_all[:val_end],
        e_all[:val_end],
        b_all[:val_end],
        y_all[:val_end],
    )
    test_nll = evaluate_range(
        weights_train_val,
        x_all,
        e_all,
        b_all,
        y_all,
        val_end,
        n_samples,
        best_coupling,
    )

    weights_full = fit_field_propagator(x_all, e_all, b_all, y_all)

    x_last = draws[-1].astype(np.float64) / MAX_NUM
    e_last = (draws[-1] - draws[-2]).astype(np.float64) / MAX_NUM
    e_prev = (draws[-2] - draws[-3]).astype(np.float64) / MAX_NUM
    b_last = e_last - e_prev

    e_next, b_next = predict_field_amplitudes(weights_full, x_last, e_last, b_last)
    final_q_probs = np.zeros((NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    for pos in range(NUM_POSITIONS):
        final_q_probs[pos] = evolve_maxwell_quantum_register(
            e_next[pos],
            b_next[pos],
            float(b_last[pos]),
            best_coupling,
        )

    next_comb = decode_combination(final_q_probs)
    if len(set(next_comb)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in next_comb):
        raise RuntimeError(f"Nelegalna NEXT kombinacija: {next_comb}")

    print(f"CSV: {file_path.name}")
    print(f"Validacija NLL (omega={best_coupling}): {best_val_nll:.4f}")
    print(f"Test NLL: {test_nll:.4f}")
    print(f"NEXT: {next_comb}")


def main() -> None:
    print("Metoda: Quantum Maxwell Coupled-Field Wave Interference (7x6 qubita)")
    for csv_path in FILES:
        process_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Quantum Maxwell Coupled-Field Wave Interference (7x6 qubita)

CSV: loto7_4718_k90.csv
Validacija NLL (omega=0.2): 20.7665
Test NLL: 20.6828
NEXT: [2, x, 18, y, 22, z, 38]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (omega=0.2): 20.8893
Test NLL: 20.9336
NEXT: [3, x, 15, y, 27, z, 38]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (omega=0.2): 21.1380
Test NLL: 20.7905
NEXT: [2, x, 16, y, 28, z, 38]
"""



"""
James Clerk Maxwell 
electricity, magnetism and light 
James Clerk Maxwell — elektromagnetizam, talasi i svetlost
Maxwellov elektromagnetizam, spregnuta polja i talasna dinamika


	1. Spregnuta polja i fazna interferencija (Maxwellove jednačine): U Maxwellovoj teoriji promena jednog polja (\(\mathbf{E}\)) indukuje drugo polje (\(\mathbf{B}\)), stvarajući talasnu propagaciju kroz prostor i vreme. U Loto 7/39 nizu to se može modelovati kao spregnuta evolutivna dinamika gde razlike između uzastopnih kola ("brzina" promene brojeva) i ubrzanje tih promena generišu talasnu funkciju verovatnoće za naredno kolo.
	2. Kvantna talasna interferencija i Hamiltonijan evolucija (Qiskit): Svetlost i elektromagnetni talasi prirodno odgovaraju unitarnoj evoluciji kvantnog stanja \(U(t) = e^{-iHt}\) u Qiskitu. Na 7 registara po 6 qubita (39 stanja), spregnute rotacije (RXX, RYY, RZZ ili CRZ između susednih qubita i pozicija) simuliraju talasno prostiranje i konstruktivnu/destruktivnu interferenciju amplituda, čime se bez klasifikatora i frekvencija dobija raspodela za sledeće kolo.
	3. Maxwell-Boltzmannova raspodela energije (statistička fizika): Pored elektromagnetizma, Maxwell je postavio temelj statističke mehanike gde se verovatnoća stanja određuje preko energija konfiguracije (\(p_i \propto e^{-E_i / T}\)), što se direktno uči iz istorijskih prelaza kao energetski pejzaž za kvantne amplitude.

Koncept spregnutih polja i talasne interferencije u unitarnoj kvantnoj evoluciji daje potpuno drugačiji matematički mehanizam od prethodne kernel gustine, a prirodno koristi kvantne faze i amplitude u Qiskitu.

rincip Maxwellovih spregnutih polja i talasne interferencije za Loto 7/39.

• Spregnuta polja (\(\mathbf{E}\) i \(\mathbf{B}\) dinamika): Iz hronološkog niza kola za svaku poziciju \(p \in \{1\dots7\}\) računaju se dva spregnuta polja: prvo polje ("brzina" promene broja \(\Delta x_t = x_t - x_{t-1}\)) i drugo polje ("rotacija / indukcija" promene \(\Delta^2 x_t = \Delta x_t - \Delta x_{t-1}\)). Regresijom prelaza na istorijskim podacima uči se spregnuta matrica propagacije između ovih polja za svaki broj \(1\dots39\).
• Kvantna talasna interferencija (Qiskit 1.4.4, 7x6 qubita): Svaki od 7 registara (po 6 qubita, 64 bazna stanja, stanja \(1\dots39\) nose brojeve) prolazi kroz unitarnu talasnu evoluciju sastavljenu od faznog kodiranja polja \(\mathbf{E}\) (RZ rotacije), indukovanog polja \(\mathbf{B}\) (RY rotacije) i spregnutih talasnih interakcija (RXX i RZZ kapije između susednih qubita) koje proizvode konstruktivnu i destruktivnu interferenciju kvantnih amplituda.
• Hronološka validacija i test: Parametar talasne sprege \(\omega \) bira se na hronološkoj validaciji (70% trening, 15% validacija), proverava na odvojenom testu (15%), a završni model se uči na celoj istoriji bez curenja podataka.
"""
