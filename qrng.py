# =============================================================================
# QUANTUM RANDOM NUMBER GENERATOR (QRNG) USING QISKIT
# Mini Project — Quantum Technologies using Qiskit Internship
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import math
import statistics
import random   # used ONLY in classical comparison — not for quantum generation


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CORE QRNG ENGINE
# Generates true random bits using quantum superposition and measurement.
#
# Quantum Principle:
#   A qubit initialised to |0⟩ and passed through a Hadamard (H) gate enters
#   equal superposition:  |ψ⟩ = (|0⟩ + |1⟩) / √2
#   Measurement collapses it to 0 or 1 with exactly 50% probability each —
#   a source of irreducible, non-deterministic randomness.
# ─────────────────────────────────────────────────────────────────────────────

class QuantumRandomBitGenerator:
    """
    Generates random bits using a Hadamard gate + measurement circuit.
    Each bit corresponds to one quantum measurement in the computational basis.
    """

    def __init__(self, n_qubits=8):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits per circuit shot.
            Each shot produces n_qubits bits simultaneously.
        """
        self.n_qubits = n_qubits
        self.backend = AerSimulator()

    def _build_circuit(self):
        """Constructs the QRNG circuit: H gate on every qubit, then measure all."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)                     # Hadamard: |0⟩ → (|0⟩ + |1⟩)/√2
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def generate_bits(self, total_bits=64):
        """
        Generates a specified number of random bits by running the circuit
        in batches of n_qubits per shot.

        Returns
        -------
        list[int]  —  list of 0s and 1s of length total_bits
        """
        shots_needed = math.ceil(total_bits / self.n_qubits)
        qc = self._build_circuit()
        compiled = transpile(qc, self.backend)
        job = self.backend.run(compiled, shots=shots_needed)
        counts = job.result().get_counts()

        bits = []
        for bitstring, freq in counts.items():
            # Qiskit returns bitstrings in little-endian order; reverse for MSB-first
            for _ in range(freq):
                bits.extend([int(b) for b in reversed(bitstring)])

        return bits[:total_bits]

    def get_circuit_diagram(self):
        """Returns an ASCII text diagram of the QRNG circuit."""
        qc = self._build_circuit()
        return qc.draw(output='text')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RANDOM NUMBER FORMATTER
# Converts raw quantum bits into various usable formats.
# ─────────────────────────────────────────────────────────────────────────────

class QRNGFormatter:
    """
    Converts a stream of quantum random bits into different number formats:
    integers, floats, binary strings, and hex strings.
    """

    @staticmethod
    def bits_to_integer(bits, n_bits=8):
        """
        Packs bits into unsigned integers using n_bits per integer.

        Example (8-bit):  [1,0,1,1,0,0,1,0] → 0b10110010 = 178
        """
        if len(bits) < n_bits:
            raise ValueError(f"Need at least {n_bits} bits; got {len(bits)}")
        integers = []
        for i in range(0, len(bits) - n_bits + 1, n_bits):
            chunk = bits[i:i + n_bits]
            value = int(''.join(str(b) for b in chunk), 2)
            integers.append(value)
        return integers

    @staticmethod
    def bits_to_float(bits, n_bits=32):
        """
        Maps a raw integer derived from n_bits into [0.0, 1.0).

        value / 2^n_bits  →  uniform float in [0, 1)
        """
        if len(bits) < n_bits:
            raise ValueError(f"Need at least {n_bits} bits for float; got {len(bits)}")
        integers = QRNGFormatter.bits_to_integer(bits, n_bits=n_bits)
        max_val = 2 ** n_bits
        return [v / max_val for v in integers]

    @staticmethod
    def bits_to_binary_strings(bits, n_bits=8):
        """Returns zero-padded binary strings of length n_bits."""
        integers = QRNGFormatter.bits_to_integer(bits, n_bits=n_bits)
        return [format(v, f'0{n_bits}b') for v in integers]

    @staticmethod
    def bits_to_hex_strings(bits, n_bits=8):
        """Returns uppercase hexadecimal strings."""
        integers = QRNGFormatter.bits_to_integer(bits, n_bits=n_bits)
        hex_digits = n_bits // 4
        return [format(v, f'0{hex_digits}X') for v in integers]

    @staticmethod
    def bits_to_range(bits, low, high, n_bits=16):
        """
        Generates random integers uniformly in [low, high] using rejection sampling.

        Why rejection sampling?
            Modulo bias arises when the range size does not evenly divide 2^n_bits.
            Rejection sampling discards values that would bias the distribution.
        """
        span = high - low + 1
        max_unbiased = (2 ** n_bits // span) * span
        results = []
        i = 0
        integers = QRNGFormatter.bits_to_integer(bits, n_bits=n_bits)
        for val in integers:
            if val < max_unbiased:
                results.append(low + (val % span))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STATISTICAL VALIDATOR
# Verifies that generated numbers exhibit the statistical properties expected
# of a truly random source: uniform distribution, correct entropy, and no bias.
# ─────────────────────────────────────────────────────────────────────────────

class RandomnessValidator:
    """
    Statistical tests to verify the quality of quantum random output.

    Tests implemented:
      1. Bit-frequency test        — P(bit=1) should be ≈ 0.50
      2. Chi-squared uniformity    — integer distribution should be flat
      3. Shannon entropy           — should approach log2(2^n_bits) = n_bits
      4. Runs test (basic)         — counts alternating runs of 0s and 1s
    """

    def __init__(self, bits):
        self.bits = bits

    def bit_frequency_test(self):
        """Tests whether proportion of 1-bits is close to 0.5."""
        n = len(self.bits)
        ones = sum(self.bits)
        proportion = ones / n
        # Expected std dev for proportion under null hypothesis (p=0.5)
        expected_std = math.sqrt(0.25 / n)
        z_score = abs(proportion - 0.5) / expected_std
        passed = z_score < 1.96   # 95% confidence interval
        return {
            "test": "Bit Frequency",
            "ones": ones,
            "zeros": n - ones,
            "proportion_ones": round(proportion, 4),
            "z_score": round(z_score, 4),
            "passed": passed
        }

    def chi_squared_test(self, integers, n_bits=8):
        """
        Chi-squared goodness-of-fit test for uniform distribution.
        Expected frequency = total_count / num_bins
        χ² = Σ (observed - expected)² / expected
        """
        num_bins = min(16, 2 ** n_bits)   # use 16 bins for tractability
        bin_size = (2 ** n_bits) // num_bins
        observed = [0] * num_bins
        for v in integers:
            bin_idx = min(v // bin_size, num_bins - 1)
            observed[bin_idx] += 1
        n = len(integers)
        expected = n / num_bins
        chi2 = sum((o - expected) ** 2 / expected for o in observed)
        # Critical value for 95% confidence, df = num_bins - 1
        # Using lookup for df=15: 24.996
        critical = 24.996
        passed = chi2 < critical
        return {
            "test": "Chi-Squared Uniformity",
            "chi2_statistic": round(chi2, 4),
            "critical_value_95pct": critical,
            "degrees_of_freedom": num_bins - 1,
            "passed": passed
        }

    def shannon_entropy(self, integers, n_bits=8):
        """
        Computes Shannon entropy of the integer distribution.
        H = -Σ p_i × log2(p_i)
        For a perfectly uniform distribution over 2^n_bits values: H = n_bits.
        """
        from collections import Counter
        counts = Counter(integers)
        n = len(integers)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
        max_entropy = n_bits  # perfect uniform distribution
        efficiency = (entropy / max_entropy) * 100
        passed = efficiency > 85.0   # accept if >85% of theoretical max
        return {
            "test": "Shannon Entropy",
            "entropy_bits": round(entropy, 4),
            "max_possible": max_entropy,
            "efficiency_pct": round(efficiency, 2),
            "passed": passed
        }

    def runs_test(self):
        """
        Basic runs test: counts consecutive sequences of identical bits.
        Too few or too many runs indicate non-randomness.
        """
        runs = 1
        for i in range(1, len(self.bits)):
            if self.bits[i] != self.bits[i - 1]:
                runs += 1
        n = len(self.bits)
        n1 = sum(self.bits)
        n0 = n - n1
        # Expected runs and variance under null hypothesis
        if n0 == 0 or n1 == 0:
            return {"test": "Runs Test", "passed": False, "note": "All bits identical"}
        expected_runs = (2 * n0 * n1) / n + 1
        variance = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n ** 2 * (n - 1))
        z = (runs - expected_runs) / math.sqrt(max(variance, 1e-10))
        passed = abs(z) < 1.96
        return {
            "test": "Runs Test",
            "observed_runs": runs,
            "expected_runs": round(expected_runs, 2),
            "z_score": round(z, 4),
            "passed": passed
        }

    def run_all(self, integers, n_bits=8):
        results = [
            self.bit_frequency_test(),
            self.chi_squared_test(integers, n_bits),
            self.shannon_entropy(integers, n_bits),
            self.runs_test()
        ]
        return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CLASSICAL RNG COMPARISON
# Compares quantum randomness to Python's Mersenne Twister PRNG
# to illustrate why quantum randomness is fundamentally different.
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalRNGComparison:
    """
    Generates classical pseudo-random numbers using Python's random module
    (Mersenne Twister, MT19937) and subjects them to the same statistical tests
    for a side-by-side comparison.
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def generate_bits(self, total_bits):
        return [self.rng.randint(0, 1) for _ in range(total_bits)]

    def generate_integers(self, count, n_bits=8):
        return [self.rng.randint(0, 2 ** n_bits - 1) for _ in range(count)]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: APPLICATION DEMONSTRATIONS
# Practical applications of QRNG: cryptographic keys, secure passphrases,
# dice simulation, and Monte Carlo estimation of π.
# ─────────────────────────────────────────────────────────────────────────────

class QRNGApplications:
    """
    Demonstrates real-world applications of the quantum random number generator.
    """

    def __init__(self, formatter):
        self.formatter = formatter

    def generate_cryptographic_key(self, bits, key_length_bits=128):
        """Generates a cryptographic-quality key as a hex string."""
        integers = self.formatter.bits_to_integer(bits, n_bits=8)
        key_bytes = key_length_bits // 8
        key = ''.join(format(v, '02X') for v in integers[:key_bytes])
        return key

    def generate_passphrase(self, bits, word_list=None):
        """
        Generates a random passphrase by selecting words using quantum bits.
        Uses a built-in minimal word list if none provided.
        """
        if word_list is None:
            word_list = [
                "alpha", "bravo", "charlie", "delta", "echo",
                "foxtrot", "golf", "hotel", "india", "juliet",
                "kilo", "lima", "mike", "november", "oscar",
                "papa", "quebec", "romeo", "sierra", "tango",
                "uniform", "victor", "whiskey", "xray", "yankee", "zulu"
            ]
        indices = self.formatter.bits_to_range(bits, 0, len(word_list) - 1, n_bits=8)
        words = [word_list[i] for i in indices[:4]]
        return '-'.join(words)

    def simulate_dice(self, bits, sides=6, rolls=10):
        """Simulates dice rolls using quantum randomness."""
        results = self.formatter.bits_to_range(bits, 1, sides, n_bits=8)
        return results[:rolls]

    def monte_carlo_pi(self, bits, n_points=50):
        """
        Estimates π using Monte Carlo integration with quantum random points.

        Method: Sample (x,y) pairs uniformly in [0,1]².
                Count how many fall inside the unit circle: x² + y² ≤ 1.
                π ≈ 4 × (points inside circle) / (total points)
        """
        floats = self.formatter.bits_to_float(bits, n_bits=16)
        # Need pairs of floats
        pairs = [(floats[i], floats[i+1]) for i in range(0, len(floats)-1, 2)]
        inside = sum(1 for x, y in pairs if x**2 + y**2 <= 1.0)
        total = len(pairs)
        if total == 0:
            return None, 0, 0
        pi_estimate = 4 * inside / total
        return pi_estimate, inside, total


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DISPLAY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)

def print_sub(title):
    print(f"\n  ── {title} ──")

def print_stat_result(result):
    status = "PASS ✓" if result["passed"] else "FAIL ✗"
    print(f"\n  [{status}]  {result['test']}")
    for k, v in result.items():
        if k not in ("test", "passed"):
            print(f"            {k:<25}: {v}")

def ascii_histogram(integers, n_bins=8, width=30, n_bits=8):
    """Renders a simple ASCII histogram of the integer distribution."""
    bin_size = (2 ** n_bits) // n_bins
    bins = [0] * n_bins
    for v in integers:
        idx = min(v // bin_size, n_bins - 1)
        bins[idx] += 1
    max_count = max(bins)
    print(f"\n  Distribution Histogram ({n_bits}-bit integers, {n_bins} bins):")
    print(f"  {'Range':<14} {'Count':>6}  Bar")
    print(f"  {'-'*14} {'-'*6}  {'-'*width}")
    for i, count in enumerate(bins):
        lo = i * bin_size
        hi = lo + bin_size - 1
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {lo:>3}–{hi:<9}   {count:>5}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 62)
    print("  QUANTUM RANDOM NUMBER GENERATOR (QRNG) USING QISKIT")
    print("  Quantum Technologies using Qiskit — Mini Project")
    print("█" * 62)

    # ── Step 1: Initialise QRNG engine ──────────────────────────
    print_section("STEP 1: QRNG CIRCUIT INITIALISATION")
    qrng_engine = QuantumRandomBitGenerator(n_qubits=8)
    print(f"\n  Backend         : AerSimulator (statevector-compatible)")
    print(f"  Qubits per shot : {qrng_engine.n_qubits}")
    print(f"  Gate applied    : Hadamard (H) on all qubits")
    print(f"  Measurement     : Computational basis (Z-basis)")
    print(f"\n  Circuit Diagram:")
    print(qrng_engine.get_circuit_diagram())

    # ── Step 2: Generate quantum random bits ────────────────────
    print_section("STEP 2: QUANTUM BIT GENERATION")
    total_bits = 512
    print(f"\n  Generating {total_bits} quantum random bits...")
    q_bits = qrng_engine.generate_bits(total_bits=total_bits)
    print(f"  Generated       : {len(q_bits)} bits")
    print(f"  First 64 bits   : {''.join(str(b) for b in q_bits[:64])}")
    print(f"  Ones count      : {sum(q_bits)}")
    print(f"  Zeros count     : {len(q_bits) - sum(q_bits)}")

    # ── Step 3: Format outputs ───────────────────────────────────
    print_section("STEP 3: FORMATTED RANDOM NUMBERS")
    formatter = QRNGFormatter()

    q_ints_8  = formatter.bits_to_integer(q_bits, n_bits=8)
    q_ints_16 = formatter.bits_to_integer(q_bits, n_bits=16)
    q_floats  = formatter.bits_to_float(q_bits, n_bits=32)
    q_hex     = formatter.bits_to_hex_strings(q_bits, n_bits=8)
    q_binstr  = formatter.bits_to_binary_strings(q_bits, n_bits=8)
    q_range   = formatter.bits_to_range(q_bits, 1, 100, n_bits=16)

    print_sub("8-bit Unsigned Integers (0–255), first 16 values")
    print(f"  {q_ints_8[:16]}")

    print_sub("16-bit Unsigned Integers (0–65535), first 8 values")
    print(f"  {q_ints_16[:8]}")

    print_sub("32-bit Floats in [0.0, 1.0), first 8 values")
    print(f"  {[round(f, 6) for f in q_floats[:8]]}")

    print_sub("Hex Strings (8-bit), first 16 values")
    print(f"  {q_hex[:16]}")

    print_sub("Binary Strings (8-bit), first 8 values")
    print(f"  {q_binstr[:8]}")

    print_sub("Integers in [1, 100] via rejection sampling, first 16 values")
    print(f"  {q_range[:16]}")

    # Distribution histogram
    ascii_histogram(q_ints_8, n_bins=8, n_bits=8)

    # ── Step 4: Statistical validation ──────────────────────────
    print_section("STEP 4: STATISTICAL VALIDATION — QUANTUM RNG")
    validator_q = RandomnessValidator(q_bits)
    results_q = validator_q.run_all(q_ints_8, n_bits=8)
    for r in results_q:
        print_stat_result(r)

    # ── Step 5: Classical RNG comparison ────────────────────────
    print_section("STEP 5: CLASSICAL PRNG COMPARISON (Mersenne Twister)")
    classical = ClassicalRNGComparison(seed=None)   # no fixed seed — as fair as possible
    c_bits = classical.generate_bits(total_bits)
    c_ints = classical.generate_integers(len(q_ints_8), n_bits=8)

    validator_c = RandomnessValidator(c_bits)
    results_c = validator_c.run_all(c_ints, n_bits=8)
    for r in results_c:
        print_stat_result(r)

    # ── Step 6: Quantum vs Classical summary ────────────────────
    print_section("STEP 6: QUANTUM vs CLASSICAL — COMPARISON SUMMARY")
    q_pass = sum(1 for r in results_q if r["passed"])
    c_pass = sum(1 for r in results_c if r["passed"])
    print(f"\n  {'Test':<30} {'Quantum':>10} {'Classical':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*12}")
    for rq, rc in zip(results_q, results_c):
        q_stat = "PASS ✓" if rq["passed"] else "FAIL ✗"
        c_stat = "PASS ✓" if rc["passed"] else "FAIL ✗"
        print(f"  {rq['test']:<30} {q_stat:>10} {c_stat:>12}")
    print(f"\n  Tests Passed — Quantum: {q_pass}/4   Classical: {c_pass}/4")
    print(f"\n  KEY DISTINCTION:")
    print(f"  Quantum RNG derives randomness from wavefunction collapse —")
    print(f"  a physical process with no hidden state and no seed.")
    print(f"  Classical PRNG (MT19937) is deterministic: given the same")
    print(f"  seed, it produces an identical sequence every time.")

    # ── Step 7: Applications ─────────────────────────────────────
    print_section("STEP 7: QRNG APPLICATIONS")
    apps = QRNGApplications(formatter)

    # Need more bits for applications
    extra_bits = qrng_engine.generate_bits(total_bits=1024)

    print_sub("Application 1: 128-bit Cryptographic Key")
    key = apps.generate_cryptographic_key(extra_bits, key_length_bits=128)
    print(f"  Key (hex) : {key}")
    print(f"  Length    : {len(key) * 4} bits  ({len(key)} hex chars)")

    print_sub("Application 2: Quantum-Generated Passphrase")
    passphrase = apps.generate_passphrase(extra_bits)
    print(f"  Passphrase: {passphrase}")

    print_sub("Application 3: Dice Simulation (d6, 10 rolls)")
    rolls = apps.simulate_dice(extra_bits, sides=6, rolls=10)
    print(f"  Rolls     : {rolls}")
    print(f"  Sum       : {sum(rolls)}  |  Mean: {sum(rolls)/len(rolls):.2f}")

    print_sub("Application 4: Monte Carlo Estimation of π")
    pi_est, inside, total = apps.monte_carlo_pi(extra_bits, n_points=80)
    print(f"  Quantum π ≈ {pi_est:.5f}  (actual: 3.14159)")
    print(f"  Points inside circle : {inside} / {total}")
    print(f"  Error               : {abs(pi_est - math.pi):.5f}")

    # ── Step 8: Entropy and security summary ─────────────────────
    print_section("STEP 8: ENTROPY & SECURITY ANALYSIS")
    ent_result = validator_q.shannon_entropy(q_ints_8, n_bits=8)
    print(f"\n  Shannon Entropy     : {ent_result['entropy_bits']:.4f} bits/symbol")
    print(f"  Theoretical Maximum : {ent_result['max_possible']} bits/symbol")
    print(f"  Entropy Efficiency  : {ent_result['efficiency_pct']:.2f}%")
    print(f"\n  Security Notes:")
    print(f"  • Output is non-deterministic — no seed, no internal state")
    print(f"  • Suitable for OTP (One-Time Pad) key generation")
    print(f"  • Suitable for IV/nonce generation in AES-GCM, ChaCha20")
    print(f"  • Not suitable for direct use without post-processing in")
    print(f"    hardware QRNG with device imperfections (Toeplitz hashing")
    print(f"    or Von Neumann extractor recommended for physical devices)")

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  EXECUTION COMPLETE")
    print("═" * 62)
    print(f"  Quantum bits generated  : {total_bits + 1024}")
    print(f"  8-bit integers produced : {len(q_ints_8)}")
    print(f"  Statistical tests passed: {q_pass}/4")
    print(f"  Applications demonstrated: 4")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
