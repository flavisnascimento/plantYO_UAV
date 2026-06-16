<<<<<<< HEAD
# MRS Computer Vision Examples

## C++

* [Edge Detector](./cpp/edge_detector) - Comprehensive C++ ROS Example with OpenCV Edge detector

## Python

* [Blob Detector](./python/blob_detector) - Simple Python ROS Example with OpenCV Blob detector

# Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=======
# Blob Detector (UAV Planter)

This package contains computer vision and control logic for a UAV planter system, designed to run within the MRS UAV System simulation environment.

## 1. Requirements

*   **OS:** Ubuntu 20.04 LTS
*   **ROS Version:** Noetic Ninjemys
*   **Core Dependency:** [MRS UAV System](https://github.com/ctu-mrs/mrs_uav_system) (Make sure the MRS simulation environment is installed and working).

## 2. Installation

These instructions assume you already have a ROS workspace created. If not, replace `catkin_ws` with your workspace name (e.g., `rma2025_ws`).

### Step 1: Clone the repository
Navigate to the `src` folder of your catkin workspace and clone this repository.

```bash
cd ~/catkin_ws/src
git clone <PASTE_YOUR_GIT_REPO_URL_HERE>
```

### Step 2: Install dependencies
It is good practice to ensure all dependencies defined in `package.xml` are installed.

```bash
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
```

### Step 3: Build the package
Compile the workspace to register the new package.

```bash
cd ~/catkin_ws
catkin build
# OR if you use standard catkin_make:
# catkin_make
```

### Step 4: Source the workspace
Don't forget to source your workspace so ROS can find the `plantYO_UAV` package and its launch files.

```bash
source ~/catkin_ws/devel/setup.bash
```
*(Tip: Add this line to your `~/.bashrc` if you haven't already)*

## 3. Usage

This package uses `tmuxinator` (via a shell script wrapper) to launch the simulation, the drone core, the computer vision node, and Rviz simultaneously.

### Start the simulation
Navigate to the package folder and run the start script:

```bash
roscd plantYO_UAV/tmux
./start.sh
```

### Stop the simulation
To kill all sessions and close the windows safely:

```bash
./kill.sh
```

## 4. Configuration

*   **Vision Logic:** The main detection logic is located in `scripts/hgs_planter_node.py`.
*   **Session Layout:** The tmux window layout is defined in `tmux/session.yml`.
*   **Rviz:** Visualization config is in `config/planter_rviz.rviz`.

---

### Troubleshooting

**"Package not found" error:**
If `start.sh` fails claiming it cannot find the package paths:
1. Ensure you have run `catkin build`.
2. Ensure you have sourced your workspace (`source ~/catkin_ws/devel/setup.bash`).
3. Ensure the folder name in `src` matches the package name defined in `package.xml`.
>>>>>>> plantyo/main
