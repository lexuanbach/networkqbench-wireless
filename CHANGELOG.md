# Software release 1.2.0

NetworkQBench-Wireless integrates quality, complete circuit resources, deadlines, applied actions, and operational energy in one benchmark. This software version does not imply publication of an earlier paper.

- Schema 1.2 adds workload, incumbent, encoding, depth, noise, controller, transition, and event identities.
- Adds matched uniform controls, structured phase encoding, verified quadratization, p=2 transfer, exact noisy distributions, previous-placement incumbents, and hypothetical sleep states.
- Uses benefit-aware event accounting and concurrent classical backup with training-only policy selection.
- Preserves separately measured records and unknown historical energy; adds complete frozen-output reproduction and portable plotting.
- The 12-qubit noisy calculation uses a bundled exact native density kernel independently checked against Aer 0.17.2. Smaller noisy cases run in Aer.
