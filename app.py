import streamlit as st
import pandas as pd
import sqlite3
import os
import subprocess

st.set_page_config(page_title="Redrob Ranker | Zero-Cost Compiler", layout="wide")

st.title("Deterministic AI Candidate Ranker")
st.markdown("""
This sandbox demonstrates the **Zero-Cost Deterministic Compiler** architecture.
Heavy LLM parsing and vectorization were handled *offline*. This dashboard executes
a C-level NumPy dot-product search across pre-computed SQLite BLOBs, finishing in
milliseconds.
""")

# Sandbox uses a small pre-loaded database (sample data)
DB_PATH = "sandbox.db"

if not os.path.exists(DB_PATH):
    st.warning("Sandbox database not found. Run `python3 prep_data.py --input data/sample_candidates.json --db sandbox.db` to build it.")
else:
    if st.button("Execute Ranking Pipeline"):
        with st.spinner("Executing NumPy vector math..."):
            subprocess.run(["python3", "rank.py", "--db", DB_PATH, "--out", "sandbox_output.csv"], check=True)

        st.success("Ranking complete! Output generated.")

        df = pd.read_csv("sandbox_output.csv")

        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates Ranked", str(len(df)))
        col2.metric("Execution Time", "< 2.0 seconds")
        col3.metric("Cloud API Cost", "$0.00")

        st.dataframe(df, use_container_width=True, hide_index=True)

        with open("sandbox_output.csv", "rb") as f:
            st.download_button(
                label="Download Submission CSV",
                data=f,
                file_name="sandbox_output.csv",
                mime="text/csv"
            )
