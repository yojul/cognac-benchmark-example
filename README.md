## Cognac Benchmark Example

**Simple implementation of multi-agent reinforcement learning (MARL) algorithms for benchmarking purposes on [🥃COGNAC](https://github.com/yojul/cognac) network environments.**

[COGNAC](https://github.com/yojul/cognac) is a collection of MARL environment with network structure.

---

## Features

* **Lightweight**: Minimal dependencies for quick setup.
* **Modular**: Easy-to-extend structure for adding new algorithms or environments.
* **Benchmark-ready**: Standardized scripts to evaluate and compare MARL performance.

---

## Getting Started

### Prerequisites

* Python 3.12 or above
* `pip` package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yojul/cognac-benchmark-example.git
   cd cognac-benchmark-example
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running Experiments

All benchmark scripts are located in the `algos/` directory. They are directly adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl) implementations and coded as self-contained single file implementation of standard MARL algorithms. To run a standard experiment:
All hyperparameters should be directly specified within the python file in the **_Args_** section.

```bash
python algos/benchmark_ippo.py
```

---

## Project Structure

```
├── algos/                # MARL algorithm implementations and benchmark scripts
│   ├── benchmark_centralized_ppo.py  # Centralized Proximal Policy Optimization, used as a baselines.
│   ├── benchmark_idqn.py             # Simple Independent Deep Q-Learning
│   ├── benchmark_ippo.py             # Simple Independent Proximal Policy Optimization
│   ├── benchmark_ippo_continuous.py  # Simple Independent Proximal Policy Optimization handling continuous action space (use for Multi-commodity flow problem)
│   ├── benchmark_mappo.py            # Multi-Agent PPO algorithm introduced by (Yu & al., 2022)
│   └── benchmark_qmix.py             # Q-Mix algorithm introduced by (Rashid & al., 2018)
├── requirements.txt      # Python dependencies
├── LICENSE.txt           # Apache-2.0 License
└── README.md             # Project overview and instructions
```

---

## Contributing

Contributions are welcome! To propose a new feature or algorithm:

---

## License

This project is licensed under the Apache-2.0 License. See [LICENSE.txt](LICENSE.txt) for details.

---

## Contact

For questions or feedback, please open an issue or contact the maintainer:

* **Repo Owner**: Jules Sintes, INRIA Paris, DIENS, École Normale Supérieure, PSL University.
* **Email**: [jules.sintes@inria.fr](mailto:jules.sintes@inria.fr)
