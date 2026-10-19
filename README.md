# MSML642 HW3: TurtleBot3 Autonomous Navigation

Abigail Basener

MSML642 - HW3 - README

10/17/2026

My Part 2 policy is a single ROS 2 node, `code/Part2/hw3_policy.py`. It has classes for `Phase`, `Planner`, `occupancyGrid`, `Gate`, `GateTracker`, `Observation`, and `PolicyExample`. These all track and interpret the data, then plan based on rules and the tracked information. It runs on the default evaluator, just like the reference policy.

---

## File structure

```
MSML642_Abigail_Basener_HW3
├── README.md
├── MSML642_Abigail_Basener_HW3
├── code/
│   └── hw3_policy.py
└── results/
    ├── part1/
    │   ├── reference_classical_maze_a.json
    │   ├── reference_classical_maze_a.log
    │   ├── reference_classical_maze_a.txt
    │   ├── reference_classical_maze_a.png
    │   ├── reference_classical_maze_a_Terminal.png
    │   ├── reference_classical_maze_b.json
    │   ├── reference_classical_maze_b.log
    │   ├── reference_classical_maze_b.txt
    │   ├── reference_classical_maze_b.png
    │   └── reference_classical_maze_b_Terminal.png
    ├── part2/
    │   ├── my_policy_maze_a.json
    │   ├── my_policy_maze_a.log
    │   ├── my_policy_maze_a.txt
    │   ├── my_policy_maze_a.png
    │   ├── my_policy_maze_a_Terminal.png
    │   ├── my_policy_maze_a.webm
    │   ├── my_policy_maze_b.json
    │   ├── my_policy_maze_b.log
    │   ├── my_policy_maze_b.txt
    │   ├── my_policy_maze_b.png
    │   ├── my_policy_maze_b_Terminal.png
    │   └── my_policy_maze_b.webm
    └── bonus/
        ├── my_policy_maze_c.json
        ├── my_policy_maze_c.log
        ├── my_policy_maze_c.txt
        ├── my_policy_maze_c.png
        ├── my_policy_maze_c_Terminal.png
        └── my_policy_maze_c.webm
```

File types:

| Ending | Contents |
|---|---|
| `.json` | Evaluator report for the run |
| `.log` | Full evaluator terminal output for the run |
| `.txt` | Copied terminal output from the run, including the interface check |
| `.png` | Screenshot of Gazebo/RViz at the end of the run |
| `_Terminal.png` | Screenshot of the terminal: interface check and final result |
| `.webm` | Screen recording of the full run (Part 2 only) |

---

## Requirements

**Software**
- The MSML642 HW3 **v2** course package.
- Docker (the course's `msml642-hw3:humble` image: ROS 2 Humble + Gazebo).
- No extra packages.

---

## Setup (once)

1. Extract the course package. All commands below are run from the extracted course folder.
2. Copy this submission's `code/Part2` folder into the course folder, so the policy is at:

   ```
   <course folder>/Part2/hw3_policy.py
   ```

3. Build the Docker image:

   ```bash
   docker build -f docker/Dockerfile.humble -t msml642-hw3:humble .
   ```

---

## How to run

Each run uses two terminals and a fresh simulator. Replace `maze_a` with `maze_b` or `maze_c` in **both** terminals to run another map.

**Terminal 1: start the simulator**

```bash
cd <course folder>
docker run --rm -it --name msml642-hw3 -p 127.0.0.1:6080:6080 -v "$PWD":/workspace -e COURSE_WORLD=maze_a msml642-hw3:humble
```

Wait for `HW3 humble maze_a running`. To watch, open `http://localhost:6080/vnc.html` and click **Connect**.

**Terminal 2: check interfaces and run the policy**

```bash
docker exec -it msml642-hw3 bash
```

Then inside the container:

```bash
source /opt/hw3/ros2_ws/install/setup.bash
python3 /opt/hw3/course_tools/check_interfaces.py
python3 /opt/hw3/course_tools/evaluate_policy.py --world maze_a --report /workspace/my_policy_maze_a.json -- python3 /workspace/Part2/hw3_policy.py 2>&1 | tee /workspace/my_policy_maze_a.log
```

---

## Notes

- The same code and parameters are used on every map. All tunable parameters are constants at the top of `hw3_policy.py`, and their values are listed in the report.
- `DEBUG_MAP` (top of the file) is `False` in the submitted version. Setting it to `True` makes the policy save debug map images to `Part2/debug/` during a run.