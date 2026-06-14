# The Zero-Cost Deterministic Compiler (India Runs Hackathon)

This system rejects the standard "real-time LLM API wrapper" approach in favor of a highly optimized, two-stage architecture designed for massive scale and absolute privacy. 

## 🏗️ Architecture
1. **Offline ETL Compiler (`prep_data.py`)**: Runs outside the 5-minute execution window. It parses raw JSON, eradicates chronological and salary honeypots, extracts behavioral scalars, and batches texts into a local `bge-small-en-v1.5` sentence-transformer. Results are serialized into an embedded SQLite database.
2. **Execution Engine (`rank.py`)**: The runtime script. It bypasses slow Python loops by utilizing a single hardware-accelerated NumPy matrix dot-product to rank 100,000 vectors in milliseconds.

## 🚀 How to Reproduce (Stage 3 Compliance)
This system operates entirely on CPU with **Network Access DISABLED**, strictly adhering to the compute limits.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Data:** Ensure `candidates.jsonl.gz` is in the `data/` folder.
3. **Compile Database (Pre-computation):**
   ```bash
   python prep_data.py
   ```

4. **Execute Ranker (The 5-Minute Window):**
   ```bash
   python rank.py
   ```

*Output `team_submission.csv` will be generated in the root directory.*
