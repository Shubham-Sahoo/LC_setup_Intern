# Safety Envelope Tracking using Light Curtains

## Installation

#### 1. Clone repository
```bash
git clone --recursive git@github.com:siddancha/LC-SET.git
cd LC-SET
```
#### 2. Install python package dependencies:
Note: this repository uses Python 3.6+.
- [simpy](https://simpy.readthedocs.io/en/latest/)
- [fire](https://github.com/google/python-fire)
- [easydict](https://pypi.org/project/easydict/)
- [sacred](https://sacred.readthedocs.io/en/stable/)
- [munch](https://pypi.org/project/munch/)
- [stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/)

```bash
pip install simpy fire easydict sacred munch stable-baselines3
```

#### 3. Set up the light curtain simulator package `lcsim`.
```bash
cd lcsim
mkdir cmake-build-release && cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release .. && make
cd ../..
```

#### 4. Install cpp utils.
```bash
cd cpp
mkdir cmake-build-release && cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release .. && make
cd ../..
```

#### 5. Set up environment variables
Note: `$DATADIR/synthia` should be the location of the [SYNTHIA-AL](http://synthia-dataset.net/downloads/) dataset.
```bash
# Ideally add these to your .bashrc file
export DATADIR=/path/to/data/dir
export PYTHONPATH=$PYTHONPATH:/path/to/LC-SET:/path/to/LC-SET/lcsim/python:/path/to/LC-SET/cpp/cmake-build-release
```

#### 6. Set up the dataset
```bash
python data/synthia.py create_video_infos_file --split=mini_train  # creates video info file in $DATADIR 
```

## Running the visualizer
In separate windows, run:

#### 1. Backend server
```bash
python ./kittiviewer/backend/set_baseline.py main # or set_velocity.py
```

#### 2. Frontend server
```bash
cd ./kittiviewer/frontend
python -m http.server
```

#### 3. Open `http://127.0.0.1:8000/` in browser

