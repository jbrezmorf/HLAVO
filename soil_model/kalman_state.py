from typing import Any, Dict, Tuple, Union, List, Callable
import attrs
from scipy import stats
import numpy as np
from scipy import linalg
"""
Representation of Kalman state and associated variables:
- initial mean and covariance: x, P
- process noise covariance: Q (but its discutable to associate it with state as it describes noise in the model)
- reference value: ref; used to compute syntetic measurements
===
- decoding the state vector to state dictionary and encoding back to vector
- Support for lornormal transform.
- Basic support for correlated fields.

"""

#############################
# Transforms
#############################
OneWayTransform = Callable[[np.ndarray], np.ndarray]
TwoWayTransform = Tuple[OneWayTransform, OneWayTransform]
transforms = dict(
    lognormal = (np.log, np.exp),   # to gauss, from gauss
    identity = (np.array, np.array)
)
#############################
# Variable Classes
#############################

@attrs.define
class GVar:
    """
    Normaly distributed variable.
    TODO:
    Could possibly be unified with the field to support vector variables.
    Difference is just specific kind of correlation matrix.
    """
    mean: float
    std: float
    Q: float
    ref: None
    transform: TwoWayTransform = transforms['identity']

    @classmethod
    def from_dict(cls, n_nodes, data: Dict[str, Any]) -> 'KalmanVar':
        if 'mean_std' in data:
            mean, std = data['mean_std']
        elif 'conf_int' in data:
            conf_int: List[float] = data['conf_int']
            if len(conf_int) == 2:
                conf_int.append(0.95)   # Default confidence level (95%)
            l, u, p = conf_int

            # Compute the mean as the midpoint of the interval
            mean = (l + u) / 2.0
            # Compute the standard deviation from the confidence interval
            z = stats.norm.ppf(1.0 - (1.0 - p) / 2.0)  # z-score for the confidence level
            std = (u - l) / (2.0 * z)
        transform = transforms[data.get('transform', 'identity')]
        return cls(mean, std, data['Q'], data['ref'], transform)

    @property
    def size(self) -> int:
        return 1

    @property
    def transform_to_gauss(self):
        return self.transform[0]

    @property
    def transform_from_gauss(self):
        return self.transform[1]

    @property
    def Q_full(self) -> np.ndarray:
        return np.diag([self.Q])

    def init_state(self, nodes_z):
        return np.array([self.mean]), np.array([[self.std ** 2]])

    def encode(self, value: float) -> np.ndarray:
        return self.transform_to_gauss(np.array([value]))

    def decode(self, value: np.ndarray) -> float:
        return self.transform_from_gauss(value)[0]

MeanField = Union['FieldMeanLinear']
@attrs.define
class FieldMeanLinear:
    top: float
    bottom: float

    def make(self, node_z):
        return np.linspace(self.top, self.bottom, len(node_z))

CovField = Union['FieldCovExponential']
@attrs.define
class FieldCovExponential:
    std: float
    corr_length: float = 0.0
    exp: float = 2.0

    def make(self, node_z):
        z_range = np.max(node_z) - np.min(node_z)
        if self.corr_length < 1.0e-10 * z_range:
            return np.diag(self.std ** 2 * np.ones(len(node_z)))
        s = np.abs(node_z[:, None]-node_z[:,None]) / self.corr_length
        cov = self.std ** 2 * np.exp(- s ** self.exp)
        return cov

@attrs.define
class GField:
    """
    Gaussian correlation field (1D).
    """

    mean: MeanField
    cov: CovField
    ref: None
    Q: float
    _size: int
    #transform: Callable[[float], float] = None

    @classmethod
    def from_dict(cls, size, data: Dict[str, Any]) -> 'GField':
        # JB TODO: check form wring config keyword make more structured config
        # to simplify implementation
        if 'mean_linear' in data:
            mean = FieldMeanLinear(**data['mean_linear'])
        if 'cov_exponential' in data:
            cov = FieldCovExponential(**data['cov_exponential'])
        return cls(mean, cov, data.get('ref', None), data['Q'], size)

    @property
    def size(self) -> int:
        return self._size

    @property
    def Q_full(self) -> np.ndarray:
        return np.diag(self.size * [self.Q])

    def init_state(self, nodes_z):
        assert len(nodes_z) == self.size
        return self.mean.make(nodes_z), self.cov.make(nodes_z)

    def encode(self, value: np.ndarray) -> np.ndarray:
        return value

    def decode(self, value: np.ndarray) -> np.ndarray:
        return value

@attrs.define
class Measure:
    """
    Auxiliary representing measurement in the state.
    """
    z_pos: np.ndarray
    #transform: Callable[[float], float] = None

    @classmethod
    def from_dict(cls, size, data: Dict[str, Any]) -> 'GField':
        # JB TODO: check form wring config keyword make more structured config
        # to simplify implementation
        return cls(data['z_pos'])

    @property
    def size(self) -> int:
        return len(self.z_pos)

    @property
    def ref(self) -> np.ndarray:
        return np.zeros(self.size)

    @property
    def Q_full(self) -> np.ndarray:
        return np.diag(self.size * [1e-8])

    def init_state(self, nodes_z):
        return np.zeros(self.size), 1e-8 * np.eye(self.size)

    def encode(self, value: np.ndarray) -> np.ndarray:
        return value

    def decode(self, value: np.ndarray) -> np.ndarray:
        return value

#############################
# Dictionary for declaring state structure and properties.
#############################
class StateStructure:
    @staticmethod
    def _resolve_var_class(key):
        if key.endswith('_field'):
            return GField
        elif key.endswith('_meas'):
            return Measure
        else:
            return GVar

    def __init__(self, n_nodes, var_cfg: Dict[str, Dict[str, Any]]):
        self.var_dict : Dict[str, Union[GVar, GField,Measure]] = {
            key: self._resolve_var_class(key).from_dict(n_nodes, val)
            for key, val in var_cfg.items()
        }

    def __getitem__(self, item):
        return self.var_dict[item]

    def size(self):
        return sum(var.size for var in self.var_dict.values())

    def compose_Q(self) -> np.ndarray:
        Q_blocks = (var.Q_full for var in self.var_dict.values())
        return linalg.block_diag(*Q_blocks)

    def compose_ref_dict(self) -> np.ndarray:
        ref_dict = {key: var.ref for key, var in  self.var_dict.items()}
        return ref_dict

    def compose_init_state(self, nodes_z) -> np.ndarray:
        mean_list, cov_list = zip(*(var.init_state(nodes_z) for var in  self.var_dict.values()))
        mean = np.concatenate(mean_list)
        cov = linalg.block_diag(*cov_list)
        return mean, cov

    def encode_state(self, value_dict):
        """
        Encodes a dict of GVariable objects into a single 1D state vector.

        :param var_dict: Dict of {variable_name: GVariable}
        :return: A 1D numpy array representing the concatenation of all variables' fields.
        """
        # Decide on a consistent order. Here we use the insertion or .values() order.
        # You could also sort by name for consistency if needed.
        components = [var.encode(value_dict[key]) for key, var in self.var_dict.items()]
        return np.concatenate(components)


    def decode_state(self, state_vector):
        """
        Decodes a 1D state vector into the existing GVariable objects.

        :param var_dict: Dict of {variable_name: GVariable}
        :param state_vector: 1D numpy array containing all the data in the same order used by encode_state.
        """
        offset = np.cumsum([var.size for var in self.var_dict.values()])
        state_dict = {
            key: var.decode(state_vector[var_off-var.size: var_off])
            for var_off, (key, var) in zip(offset, self.var_dict.items())
        }
        return state_dict



