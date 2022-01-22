from pathlib import Path

# mk dir for data
Path("../data").mkdir(parents=True, exist_ok=True)
Path("../data/neighborFinder").mkdir(parents=True, exist_ok=True)
# mk dir for output
Path("../output").mkdir(parents=True, exist_ok=True)
Path("../output/results").mkdir(parents=True, exist_ok=True)
Path("../output/logs").mkdir(parents=True, exist_ok=True)
