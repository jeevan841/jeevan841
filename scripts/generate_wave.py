import os
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USERNAME = os.environ["USERNAME"]
TOKEN = os.environ["GITHUB_TOKEN"]

QUERY = """
query($login: String!) {
  user(login: $login) {
    contributionsCollection {
      contributionCalendar {
        weeks {
          contributionDays {
            date
            contributionCount
          }
        }
      }
    }
  }
}
"""

resp = requests.post(
    "https://api.github.com/graphql",
    json={"query": QUERY, "variables": {"login": USERNAME}},
    headers={"Authorization": f"Bearer {TOKEN}"},
)
resp.raise_for_status()
weeks = resp.json()["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]

days = [d["contributionCount"] for w in weeks for d in w["contributionDays"]]
counts = np.array(days, dtype=float)

window = 7
rolling = np.convolve(counts, np.ones(window) / window, mode="valid")

fft_vals = np.fft.rfft(counts - counts.mean())
fft_mag = np.abs(fft_vals)
freqs = np.fft.rfftfreq(len(counts), d=1)

fig, axes = plt.subplots(2, 1, figsize=(10, 5))

axes[0].plot(rolling, color="#39d353", linewidth=1.5)
axes[0].set_title("Commit activity — 7-day rolling average", color="white")
axes[0].set_facecolor("#0d1117")

axes[1].plot(freqs[1:], fft_mag[1:], color="#58a6ff", linewidth=1.2)
axes[1].set_title("Commit rhythm — frequency spectrum (FFT)", color="white")
axes[1].set_facecolor("#0d1117")

fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

plt.tight_layout()
os.makedirs("dist", exist_ok=True)
plt.savefig("dist/commit-wave.svg", format="svg", transparent=False)
