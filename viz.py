import json, matplotlib.pyplot as plt

with open("schedule.json") as fh:
    sched = json.load(fh)

# group by core
cores = {}
for s in sched:
    cores.setdefault(s["core_id"], []).append(s)

fig, ax = plt.subplots(figsize=(10, 0.6 * len(cores) + 1))

yticks, ylabels = [], []
for i, (cid, tasks) in enumerate(sorted(cores.items())):
    yticks.append(i)
    ylabels.append(f"Core {cid}")
    for t in tasks:
        ax.broken_barh([(t["start"], t["finish"] - t["start"])],
                       (i - 0.4, 0.8))

ax.set_xlabel("Time")
ax.set_yticks(yticks, ylabels)
ax.set_title("Robust list-schedule (tasks coloured automatically)")
plt.tight_layout()
plt.show()
