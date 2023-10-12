<img  align="right" width="150" height="150"  src="/docs/source/logo.png">

# Revolve2-Multi
Welcome to the Revolve2-Multi Tutorial, it goes along with Dimitri Kachler's thesis, "Predator-Prey Dynamics in an Open-Ended Environment".
This readme will guide you along the steps to be able to recreate the experiments found in the paper.

#### 0. The main folder where our code for the experiment is found at:

/revolve2-multi/examples/sanity3P/

The main files to look at are:

optimize.py

optimizer.py

genotype.py

#### 1. First you want to create a virtual environment with Python version 3.8 as the primary Python distribution.
Then you want to create a virtual environment to serve as a capsule for all the libraries:

#### python3.8 -m virtualenv .venv

Then, activate the virtual environment:
#### source .venv/bin/activate

#### 2. Then you want to clone this repository to a directory, we will refer to the directory as "/yourdir/"

#### 3. In the virtual environment, we install the necessary libraries:

pip install /yourdir/revolve2-multi/core

pip install /yourdir/revolve2-multi/standard_resources/

pip install /yourdir/revolve2-multi/genotypes/cppnwin

pip install /yourdir/revolve2-multi/runners/mujoco

#### 4. We change our directory to the experiment folder, this is also where all the data will be:

cd /yourdir/revolve2-multi/examples/sanity3P/

#### 5. The experiment runs by activating optimize.py. You should receive preliminary warnings, at which point the experiment is running:

python3 optimize.py

#### 6. After 6500 seconds (instead of 6000 just to be safe), the experiment will automatically stop and the data will be stored in two files: "countries.csv" and "deathBorn.csv".

#### 7. Copy and rename both files according to the experiment run, i.e. "countries4.csv" and upload both files to the Deepnote project to visualize the data:

https://deepnote.com/workspace/oasis-4a75-9d2a5e51-2ae1-4a9f-9e63-b0cb3d7f90e0/project/VisualizeMetrics-6e3b7bcb-ad31-4c62-8372-a7e756152eb6/notebook/visualizeWalls-592bf1878e7447f59ea271b009d4575f

#### 8. After all 10 runs have been completed and uploaded to Deepnote, the visualization may begin



## Documentation For Revolve2
[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/)
