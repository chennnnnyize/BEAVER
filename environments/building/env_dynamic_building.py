import numpy as np
from environments.building.env_building import BuildingEnv_2d, BuildingEnv_3d, BuildingEnv_9d
from environments.building.utils_building import compute_RC_tables
from scipy.linalg import expm
from numpy.linalg import inv

class BuildingEnv_DR_2d(BuildingEnv_2d):
    """
    An extended environment for testing dynamics generalization.
    Allows dynamic sampling of U-Wall parameters and recomputes the RC network.
    """
    def __init__(self, Parameter, *args, **kwargs):
        # Store current task (U_wall) and structure info for re-computation
        self.U_wall = list(Parameter['U_Wall'])
        self.roomnum = Parameter['roomnum']
        self.building_structure = Parameter['building_structure']
        self.original_Parameter = Parameter.copy()

        super().__init__(Parameter, *args, **kwargs)

        # Index-to-name mapping for U_Wall elements
        self.dyn_ind_to_name = {
            0: 'intwall',
            1: 'floor',
            2: 'outwall',
            3: 'roof',
            4: 'ceiling',
            5: 'groundfloor',
            6: 'window'
        }

        # Task-related metadata
        self.original_task = np.copy(self.get_task())
        self.nominal_values = np.copy(self.original_task)
        self.task_dim = len(self.original_task)

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.set_task_search_bounds()

    def get_task(self):
        """Returns the current U_Wall values as a task vector."""
        return np.array(self.U_wall)

    def set_task(self, *task):
        """Sets a new U_Wall vector and recomputes dynamics."""
        task = np.array(task).flatten()
        assert len(task) == 7
        self.U_wall = list(task)
        self.recompute_RC()

    def recompute_RC(self):
        """
        Recomputes RCtable, connectmap, weightCmap, and nonlinear terms
        based on the current U_Wall task.
        """
        self.RCtable, self.connectmap, self.weightCmap, self.nonlinear = compute_RC_tables(
            self.U_wall,
            self.roomnum,
            self.building_structure,
            full_occ=0,
            AC_map=self.acmap,
            max_power=self.maxpower,
            shgc=0.252,
            shgc_weight=0.01,
            ground_weight=0.5
        )

        # Recompute system dynamics matrices A_d and B_d
        Amatrix = self.RCtable[:, :-1]
        diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
        np.fill_diagonal(Amatrix, np.diag(diagvalue))
        Amatrix += self.nonlinear * self.OCCU_COEF9 / self.roomnum
        Bmatrix = self.weightCmap.T
        Bmatrix[2] = self.connectmap[:, -1] * self.RCtable[:, -1]
        Bmatrix = Bmatrix.T
        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix

    def get_search_bounds_mean(self, index):
        """
        Returns reasonable min/max bounds for each dynamic parameter.
        Used for task sampling.
        """
        bounds_mean = {
            'intwall': (0.774, 6.299),
            'floor': (0.386, 3.145),
            'outwall': (0.269, 2.191),
            'roof': (0.160, 1.304),
            'ceiling': (0.386, 3.145),
            'groundfloor': (0.386, 3.145),
            'window': (1.950, 3.622)
        }
        return bounds_mean[self.dyn_ind_to_name[index]]

    def set_task_search_bounds(self):
        """Initializes min/max bounds for each U_Wall parameter."""
        for i in range(self.task_dim):
            low, high = self.get_search_bounds_mean(i)
            self.min_task[i] = low
            self.max_task[i] = high

    def sample_task(self):
        """Samples a random U_Wall task vector."""
        return np.random.uniform(self.min_task, self.max_task)

    def set_random_task(self):
        """Samples and sets a random U_Wall task."""
        task = self.sample_task()
        self.set_task(*task)

    def reset(self, *args, **kwargs):
        """Resets the environment with a new random dynamics task."""
        self.set_random_task()
        #print(self.A_d)
        return super().reset(*args, **kwargs)
    
    def reset_with_task(self, task, *args, **kwargs):
        """
        Reset the environment with a fixed U_wall setting (instead of random sampling).
        task: 1D array or list of length 7
        """
        self.set_task(*task)
        return super().reset(*args, **kwargs)

    def step(self, action):
        """Steps the environment with the given action (unchanged from base)."""
        return super().step(action)

class BuildingEnv_DR_3d(BuildingEnv_3d):
    """
    An extended environment for testing dynamics generalization.
    Allows dynamic sampling of U-Wall parameters and recomputes the RC network.
    """
    def __init__(self, Parameter, *args, **kwargs):
        # Store current task (U_wall) and structure info for re-computation
        self.U_wall = list(Parameter['U_Wall'])
        self.roomnum = Parameter['roomnum']
        self.building_structure = Parameter['building_structure']
        self.original_Parameter = Parameter.copy()

        super().__init__(Parameter, *args, **kwargs)

        # Index-to-name mapping for U_Wall elements
        self.dyn_ind_to_name = {
            0: 'intwall',
            1: 'floor',
            2: 'outwall',
            3: 'roof',
            4: 'ceiling',
            5: 'groundfloor',
            6: 'window'
        }

        # Task-related metadata
        self.original_task = np.copy(self.get_task())
        self.nominal_values = np.copy(self.original_task)
        self.task_dim = len(self.original_task)

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.set_task_search_bounds()

    def get_task(self):
        """Returns the current U_Wall values as a task vector."""
        return np.array(self.U_wall)

    def set_task(self, *task):
        """Sets a new U_Wall vector and recomputes dynamics."""
        task = np.array(task).flatten()
        assert len(task) == 7
        self.U_wall = list(task)
        self.recompute_RC()

    def recompute_RC(self):
        """
        Recomputes RCtable, connectmap, weightCmap, and nonlinear terms
        based on the current U_Wall task.
        """
        self.RCtable, self.connectmap, self.weightCmap, self.nonlinear = compute_RC_tables(
            self.U_wall,
            self.roomnum,
            self.building_structure,
            full_occ=0,
            AC_map=self.acmap,
            max_power=self.maxpower,
            shgc=0.252,
            shgc_weight=0.01,
            ground_weight=0.5
        )

        # Recompute system dynamics matrices A_d and B_d
        Amatrix = self.RCtable[:, :-1]
        diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
        np.fill_diagonal(Amatrix, np.diag(diagvalue))
        Amatrix += self.nonlinear * self.OCCU_COEF9 / self.roomnum
        Bmatrix = self.weightCmap.T
        Bmatrix[2] = self.connectmap[:, -1] * self.RCtable[:, -1]
        Bmatrix = Bmatrix.T
        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix

    def get_search_bounds_mean(self, index):
        """
        Returns reasonable min/max bounds for each dynamic parameter.
        Used for task sampling.
        """
        bounds_mean = {
            'intwall': (0.774, 6.299),
            'floor': (0.386, 3.145),
            'outwall': (0.269, 2.191),
            'roof': (0.160, 1.304),
            'ceiling': (0.386, 3.145),
            'groundfloor': (0.386, 3.145),
            'window': (1.950, 3.622)
        }
        return bounds_mean[self.dyn_ind_to_name[index]]

    def set_task_search_bounds(self):
        """Initializes min/max bounds for each U_Wall parameter."""
        for i in range(self.task_dim):
            low, high = self.get_search_bounds_mean(i)
            self.min_task[i] = low
            self.max_task[i] = high

    def sample_task(self):
        """Samples a random U_Wall task vector."""
        return np.random.uniform(self.min_task, self.max_task)

    def set_random_task(self):
        """Samples and sets a random U_Wall task."""
        task = self.sample_task()
        self.set_task(*task)

    def reset(self, *args, **kwargs):
        """Resets the environment with a new random dynamics task."""
        self.set_random_task()
        return super().reset(*args, **kwargs)
    
    def reset_with_task(self, task, *args, **kwargs):
        """
        Reset the environment with a fixed U_wall setting (instead of random sampling).
        task: 1D array or list of length 7
        """
        self.set_task(*task)
        return super().reset(*args, **kwargs)

    def step(self, action):
        """Steps the environment with the given action (unchanged from base)."""
        return super().step(action)
    
class BuildingEnv_DR_9d(BuildingEnv_9d):
    """
    An extended environment for testing dynamics generalization.
    Allows dynamic sampling of U-Wall parameters and recomputes the RC network.
    """
    def __init__(self, Parameter, *args, **kwargs):
        # Store current task (U_wall) and structure info for re-computation
        self.U_wall = list(Parameter['U_Wall'])
        self.roomnum = Parameter['roomnum']
        self.building_structure = Parameter['building_structure']
        self.original_Parameter = Parameter.copy()

        super().__init__(Parameter, *args, **kwargs)

        # Index-to-name mapping for U_Wall elements
        self.dyn_ind_to_name = {
            0: 'intwall',
            1: 'floor',
            2: 'outwall',
            3: 'roof',
            4: 'ceiling',
            5: 'groundfloor',
            6: 'window'
        }

        # Task-related metadata
        self.original_task = np.copy(self.get_task())
        self.nominal_values = np.copy(self.original_task)
        self.task_dim = len(self.original_task)

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.set_task_search_bounds()

    def get_task(self):
        """Returns the current U_Wall values as a task vector."""
        return np.array(self.U_wall)

    def set_task(self, *task):
        """Sets a new U_Wall vector and recomputes dynamics."""
        task = np.array(task).flatten()
        assert len(task) == 7
        self.U_wall = list(task)
        self.recompute_RC()

    def recompute_RC(self):
        """
        Recomputes RCtable, connectmap, weightCmap, and nonlinear terms
        based on the current U_Wall task.
        """
        self.RCtable, self.connectmap, self.weightCmap, self.nonlinear = compute_RC_tables(
            self.U_wall,
            self.roomnum,
            self.building_structure,
            full_occ=0,
            AC_map=self.acmap,
            max_power=self.maxpower,
            shgc=0.252,
            shgc_weight=0.01,
            ground_weight=0.5
        )

        # Recompute system dynamics matrices A_d and B_d
        Amatrix = self.RCtable[:, :-1]
        diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
        np.fill_diagonal(Amatrix, np.diag(diagvalue))
        Amatrix += self.nonlinear * self.OCCU_COEF9 / self.roomnum
        Bmatrix = self.weightCmap.T
        Bmatrix[2] = self.connectmap[:, -1] * self.RCtable[:, -1]
        Bmatrix = Bmatrix.T
        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix

    def get_search_bounds_mean(self, index):
        """
        Returns reasonable min/max bounds for each dynamic parameter.
        Used for task sampling.
        """
        bounds_mean = {
            'intwall': (0.774, 6.299),
            'floor': (0.386, 3.145),
            'outwall': (0.269, 2.191),
            'roof': (0.160, 1.304),
            'ceiling': (0.386, 3.145),
            'groundfloor': (0.386, 3.145),
            'window': (1.950, 3.622)
        }
        return bounds_mean[self.dyn_ind_to_name[index]]

    def set_task_search_bounds(self):
        """Initializes min/max bounds for each U_Wall parameter."""
        for i in range(self.task_dim):
            low, high = self.get_search_bounds_mean(i)
            self.min_task[i] = low
            self.max_task[i] = high

    def sample_task(self):
        """Samples a random U_Wall task vector."""
        return np.random.uniform(self.min_task, self.max_task)

    def set_random_task(self):
        """Samples and sets a random U_Wall task."""
        task = self.sample_task()
        self.set_task(*task)

    def reset(self, *args, **kwargs):
        """Resets the environment with a new random dynamics task."""
        self.set_random_task()
        return super().reset(*args, **kwargs)
    
    def reset_with_task(self, task, *args, **kwargs):
        """
        Reset the environment with a fixed U_wall setting (instead of random sampling).
        task: 1D array or list of length 7
        """
        self.set_task(*task)
        return super().reset(*args, **kwargs)

    def step(self, action):
        """Steps the environment with the given action (unchanged from base)."""
        return super().step(action)


