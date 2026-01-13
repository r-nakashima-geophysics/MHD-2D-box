"""A Python script to simulate two-dimensional (2D) incompressible
magnetohydrodynamic (MHD) flows in a box or on a beta plane with the
finite difference and the vorticity-stream function methods.

This script creates a movie showing the time evolution of the stream
function, vorticity, vector potential and electric current.

Notes
-----
All other parameters aside from command line arguments are described
within the script.

Examples
--------
Run the script:
    $ python3 mhd2dbox.py
"""

import sys

import numpy as np
from matplotlib import animation
from numba import njit, prange

from package_common.common_types import ArrayFloat, Final, Self
from package_common.default_logger import DefaultLogger
from package_common.default_plotter import DefaultPlotter, create_plotter
from package_common.default_timer import DefaultTimer
from package_common.progress_bar import ProgressBar
from package_common.utils_name import create_function_name_logger

# ========== Parameters ========== #

# Reynolds number
RE: Final[float] = 1000

# Magnetic Reynolds number
RM: Final[float] = np.inf

# Alfven Mach number
MA: Final[float] = np.inf

# Nondimensional beta
BETA_PARAM: Final[float] = 0

# The boundary condition for velocity ('free-slip', 'no-slip', or
# 'periodic')
# BC_VELOCITY[0]: bottom and top boundaries
# BC_VELOCITY[1]: left and right boundaries
BC_VELOCITY: Final[tuple[str, str]] = ('free-slip', 'free-slip')

# The boundary condition for magnetic field ('perfect-conductor' or
# 'periodic')
# BC_MAGNETIC[0]: bottom and top boundaries
# BC_MAGNETIC[1]: left and right boundaries
BC_MAGNETIC: Final[tuple[str, str]] = ('periodic', 'periodic')

# The initial condition for velocity
# IC_VELOCITY[0]: type ('zero', 'x-linear-shear' or 'vortex')
# IC_VELOCITY[1]: sign ('+' or '-')
IC_VELOCITY: Final[tuple[str, str]] = ('vortex', '+')

# The initial condition for magnetic field
# IC_MAGNETIC[0]: type ('zero', 'x-uniform' or 'x-linear-shear')
# IC_MAGNETIC[1]: sign ('+' or '-')
IC_MAGNETIC: Final[tuple[str, str]] = ('zero', '+')

# Aspect ratio (x/y)
ASPECT_RATIO: Final[float] = 1
# The number of grid points in the x and y directions
NUM_Y: Final[int] = 101
NUM_X: Final[int] = int(ASPECT_RATIO * (NUM_Y-1)) + 1

# The forcing terms in the vorticity and vector potential equations
# strength
STRENGTH_FORCING_OMEGA: Final[float] = 0
STRENGTH_FORCING_POTENTIAL: Final[float] = 0
# x-coordinate of the center of the forcing
X_FORCING: Final[float] = ASPECT_RATIO / 2
# y-coordinate of the center of the forcing
Y_FORCING: Final[float] = 0.5
# standard deviation of the Gaussian function
SD_FORCING: Final[float] = 0.05
# frequency
FREQUENCY_FORCING: Final[float] = 2 * np.pi / np.inf
# wave number of the forcing
K_FORCING: Final[tuple[float, float]] = (
    2 * np.pi / np.inf,
    2 * np.pi / np.inf
)

# The interval of a time step
DT: Final[float] = 10**(-3)
# The number of time steps
NUM_STEP: Final[int] = 10000
# The number of frames in the movie
NUM_FRAME: Final[int] = 50

# The parameters for the SOR method
# relaxation parameter (0 < OMEGA_SOR < 2)
OMEGA_SOR: Final[float] = 1.8
# tolerance (relative error)
TOL_SOR: Final[float] = 10**(-3)
# the number of maximum iterations
MAX_ITER_SOR: Final[int] = 100

# The boolean value to switch whether to use parallelization in Numba
SWITCH_NUMBA_PARALLEL: Final[bool] = False

# ================================ #

# The number of time steps between plots
STEP_INTERVAL_PLOT: Final[int] = NUM_STEP // NUM_FRAME


class Variable:
    """Class to handle physical quantities on grid points.

    Attributes
    ----------
    value : ArrayFloat
        The values of a physical quantity on grid points.
    value_prev : ArrayFloat
        The previous values of a physical quantity on grid points.

    Notes
    -----
    Before constructing an instance of this class, it is necessary to
    execute set_class_variables class method.

    Examples
    --------
    >>> Variable.set_class_variables(101, 101, 1)
    >>> grid_x: Variable = Variable('grid_x')
    >>> grid_y: Variable = Variable('grid_y')
    """

    __num_x: int
    __num_y: int
    __aspect_ratio: float
    __dx: float
    __dy: float
    __bc_velocity: tuple[str, str]
    __bc_magnetic: tuple[str, str]
    __type_ic_velocity: str
    __sign_ic_velocity: int
    __type_ic_magnetic: str
    __sign_ic_magnetic: int

    __flag_set_class_variables: bool = False

    @classmethod
    def set_class_variables(cls,
                            num_x: int,
                            num_y: int,
                            aspect_ratio: float,
                            *,
                            bc_velocity: tuple[str, str],
                            bc_magnetic: tuple[str, str],
                            ic_velocity: tuple[str, str],
                            ic_magnetic: tuple[str, str],) -> None:
        """Set class variables of Variable class.

        Parameters
        ----------
        num_x : int
            The number of grid points in the x direction.
        num_y : int
            The number of grid points in the y direction.
        aspect_ratio : float
            Aspect ratio (x/y).
        bc_velocity : tuple[str, str]
            The boundary condition for velocity.
        bc_magnetic : tuple[str, str]
            The boundary condition for magnetic field.
        ic_velocity : tuple[str, str]
            The initial condition for velocity.
        ic_magnetic : tuple[str, str]
            The initial condition for magnetic field.

        Warnings
        --------
        Invalid boundary condition
            If the boundary conditions are invalid.
        Invalid initial condition
            If the initial conditions are invalid.
        Incompatible initial and boundary conditions
            If the initial and boundary conditions are incompatible.
        """

        logger: DefaultLogger = create_function_name_logger()

        cls.__num_x = num_x
        cls.__num_y = num_y
        cls.__aspect_ratio = aspect_ratio
        cls.__dx = aspect_ratio / (num_x-1)
        cls.__dy = 1 / (num_y-1)

        if not (all((bc in ['free-slip',
                            'no-slip',
                            'periodic']) for bc in bc_velocity)
                and all((bc in ['perfect-conductor',
                                'periodic']) for bc in bc_magnetic)):
            logger.error('Invalid boundary condition')
            sys.exit(1)
        cls.__bc_velocity = bc_velocity
        cls.__bc_magnetic = bc_magnetic

        if not ((ic_velocity[0] in ['zero',
                                    'x-linear-shear',
                                    'vortex'])
                and (ic_magnetic[0] in ['zero',
                                        'x-uniform',
                                        'x-linear-shear'])):
            logger.error('Invalid initial condition')
            sys.exit(1)
        cls.__type_ic_velocity = ic_velocity[0]
        cls.__type_ic_magnetic = ic_magnetic[0]

        if ((ic_velocity[0] == 'x-linear-shear')
                and (bc_velocity[1] != 'periodic')) \
            or ((ic_magnetic[0] == 'x-uniform')
                and (bc_magnetic[1] != 'periodic')) \
            or ((ic_magnetic[0] == 'x-linear-shear')
                and (bc_magnetic[1] != 'periodic')):
            logger.error(
                'Incompatible initial and boundary conditions')
            sys.exit(1)

        if ((ic_velocity[0] == 'x-linear-shear')
                and (bc_velocity[0] != 'no-slip')) \
            or ((ic_magnetic[0] == 'x-uniform')
                and (bc_magnetic[0] != 'perfect-conductor')) \
            or ((ic_magnetic[0] == 'x-linear-shear')
                and (bc_magnetic[0] != 'perfect-conductor')):
            logger.error(
                'Incompatible initial and boundary conditions')
            sys.exit(1)

        if ic_velocity[1] == '+':
            cls.__sign_ic_velocity = 1
        elif ic_velocity[1] == '-':
            cls.__sign_ic_velocity = -1
        else:
            logger.error('Invalid initial condition')
            sys.exit(1)

        if ic_magnetic[1] == '+':
            cls.__sign_ic_magnetic = 1
        elif ic_magnetic[1] == '-':
            cls.__sign_ic_magnetic = -1
        else:
            logger.error('Invalid initial condition')
            sys.exit(1)

        cls.__flag_set_class_variables = True

    def __init__(self,
                 name: str) -> None:
        """Initialize an instance of Variable class.

        Parameters
        ----------
        name : str
            'grid_x', 'grid_y', 'psi', 'omega', 'potential' or
            'current'.

        Warnings
        --------
        set_class_variables has not been executed yet
            If set_class_variables class method has not been executed
            yet.
        Invalid argument
            If the argument is neither 'grid_x', 'grid_y', 'psi',
            'omega', 'potential' nor 'current'.
        """

        logger: DefaultLogger = create_function_name_logger()

        if not Variable.__flag_set_class_variables:
            logger.error(
                'set_class_variables has not been executed yet')
            sys.exit(1)

        if name not in ['grid_x', 'grid_y',
                        'psi', 'omega', 'potential', 'current']:
            logger.error('Invalid argument')
            sys.exit(1)
        self.__name: str = name

        num_x: int = Variable.__num_x
        num_y: int = Variable.__num_y
        self.value: ArrayFloat \
            = np.empty((num_x, num_y), dtype=np.float64)
        self.value_prev: ArrayFloat \
            = np.empty((num_x, num_y), dtype=np.float64)

        dx: float = Variable.__dx
        dy: float = Variable.__dy
        x: float
        y: float
        for ix in range(num_x):
            x = dx * ix
            for iy in range(num_y):
                y = dy * iy
                self.value[ix, iy] = self.__set_initial_condition(x, y)
                self.value_prev[ix, iy] = self.value[ix, iy]

    def __set_initial_condition(self,
                                x: float,
                                y: float) -> float:
        """Set the (initial) value at a grid point (x, y).

        Parameters
        ----------
        x : float
            x-coordinate of a grid point.
        y : float
            y-coordinate of a grid point.

        Returns
        -------
        float
            The (initial) value at the grid point (x, y).
        """

        aspect_ratio: float = Variable.__aspect_ratio
        type_ic_velocity: str = Variable.__type_ic_velocity
        sign_ic_velocity: int = Variable.__sign_ic_velocity
        type_ic_magnetic: str = Variable.__type_ic_magnetic
        sign_ic_magnetic: int = Variable.__sign_ic_magnetic

        bc_velocity: tuple[str, str] = Variable.__bc_velocity
        n: float
        m: float
        if (bc_velocity[0] == 'free-slip') \
                or (bc_velocity[0] == 'periodic'):
            m = 1
        elif (bc_velocity[0] == 'no-slip'):
            m = 2
        if (bc_velocity[1] == 'free-slip') \
                or (bc_velocity[1] == 'periodic'):
            n = 1
        elif (bc_velocity[1] == 'no-slip'):
            n = 2

        pi_x: float = np.pi * x / aspect_ratio
        pi_y: float = np.pi * y

        if self.__name == 'grid_x':
            return x

        elif self.__name == 'grid_y':
            return y

        elif self.__name == 'psi':

            if type_ic_velocity == 'zero':
                return sign_ic_velocity * 0
            elif type_ic_velocity == 'x-linear-shear':
                return sign_ic_velocity * ((y - 0.5)**2)
            elif type_ic_velocity == 'vortex':
                return sign_ic_velocity \
                    * (np.sqrt(aspect_ratio) / np.pi) \
                    * (np.sin(pi_x)**n) * (np.sin(pi_y)**m)

        elif self.__name == 'omega':

            if type_ic_velocity == 'zero':
                return sign_ic_velocity * 0
            elif type_ic_velocity == 'x-linear-shear':
                return sign_ic_velocity * -2
            elif type_ic_velocity == 'vortex':
                return sign_ic_velocity * -(
                    np.pi / np.sqrt(aspect_ratio)) * (
                    (n / aspect_ratio) * (
                        (n-1)*(np.cos(pi_x)**2)
                        - (np.sin(pi_x)**(n-1)) * np.sin(pi_x)
                    ) * (np.sin(pi_y)**m)
                    + m * aspect_ratio * (np.sin(pi_x)**n) * (
                        (m-1)*(np.cos(pi_y)**2)
                        - (np.sin(pi_y)**(m-1)) * np.sin(pi_y)
                    )
                )

        elif self.__name == 'potential':

            if type_ic_magnetic == 'zero':
                return sign_ic_magnetic * 0
            elif type_ic_magnetic == 'x-uniform':
                return sign_ic_magnetic * y
            elif type_ic_magnetic == 'x-linear-shear':
                return sign_ic_magnetic * (y**2)

        elif self.__name == 'current':

            if type_ic_magnetic == 'zero':
                return sign_ic_magnetic * 0
            elif type_ic_magnetic == 'x-uniform':
                return sign_ic_magnetic * 0
            elif type_ic_magnetic == 'x-linear-shear':
                return sign_ic_magnetic * -2

    def set_boundary_condition(self,
                               psi: Self | None = None,
                               potential: Self | None = None) -> None:
        """Set the boundary condition.

        Parameters
        ----------
        psi : Self | None, optional, default None
            The instance of Variable class (psi).
        potential : Self | None, optional, default None
            The instance of Variable class (potential).

        Warnings
        --------
        Invalid argument
            If the arguments are invalid.
        """

        logger: DefaultLogger = create_function_name_logger()

        bc_velocity: tuple[str, str] = Variable.__bc_velocity
        bc_magnetic: tuple[str, str] = Variable.__bc_magnetic
        sign_ic_velocity: int = Variable.__sign_ic_velocity
        sign_ic_magnetic: int = Variable.__sign_ic_magnetic

        psi_top: float
        psi_bottom: float
        u_top: float
        u_bottom: float
        if Variable.__type_ic_velocity == 'x-linear-shear':
            psi_top = sign_ic_velocity * (1 / 4)
            psi_bottom = sign_ic_velocity * (1 / 4)
            u_top = sign_ic_velocity * 1
            u_bottom = sign_ic_velocity * -1
        else:
            psi_top = sign_ic_velocity * 0
            psi_bottom = sign_ic_velocity * 0
            u_top = sign_ic_velocity * 0
            u_bottom = sign_ic_velocity * 0

        potential_top: float
        potential_bottom: float
        b_top: float
        b_bottom: float
        if Variable.__type_ic_magnetic == 'x-uniform':
            potential_top = sign_ic_magnetic * 1
            potential_bottom = sign_ic_magnetic * 0
            b_top = sign_ic_magnetic * 1
            b_bottom = sign_ic_magnetic * 1
        elif Variable.__type_ic_magnetic == 'x-linear-shear':
            potential_top = sign_ic_magnetic * 1
            potential_bottom = sign_ic_magnetic * 0
            b_top = sign_ic_magnetic * 2
            b_bottom = sign_ic_magnetic * 0
        else:
            potential_top = sign_ic_magnetic * 0
            potential_bottom = sign_ic_magnetic * 0
            b_top = sign_ic_magnetic * 0
            b_bottom = sign_ic_magnetic * 0

        dx: float = Variable.__dx
        dy: float = Variable.__dy

        if self.__name == 'psi':

            # Bottom and top boundaries
            if (bc_velocity[0] == 'free-slip') \
                    or (bc_velocity[0] == 'no-slip'):
                self.value[:, 0] = psi_bottom
                self.value[:, -1] = psi_top
            elif bc_velocity[0] == 'periodic':
                self.value[:, 0] = self.value[:, -2]
                self.value[:, -1] = self.value[:, 1]

            # Left and right boundaries
            if (bc_velocity[1] == 'free-slip') \
                    or (bc_velocity[1] == 'no-slip'):
                self.value[0, :] = 0
                self.value[-1, :] = 0
            elif bc_velocity[1] == 'periodic':
                self.value[0, :] = self.value[-2, :]
                self.value[-1, :] = self.value[1, :]

        elif self.__name == 'omega':

            psi_name: str = ''
            if isinstance(psi, Variable):
                psi_name = psi.__name
            if (psi_name != 'psi'):
                logger.error('Invalid argument')
                sys.exit(1)

            # Bottom and top boundaries
            if bc_velocity[0] == 'free-slip':
                self.value[:, 0] = 0
                self.value[:, -1] = 0
            elif bc_velocity[0] == 'no-slip':
                self.value[:, 0] \
                    = -2 * (psi.value[:, 1]
                            - psi.value[:, 0]) / (dy**2) \
                    + 2 * u_bottom / dy
                self.value[:, -1] \
                    = -2 * (psi.value[:, -2]
                            - psi.value[:, -1]) / (dy**2) \
                    - 2 * u_top / dy
            elif bc_velocity[0] == 'periodic':
                self.value[:, 0] = self.value[:, -2]
                self.value[:, -1] = self.value[:, 1]

            # Left and right boundaries
            if bc_velocity[1] == 'free-slip':
                self.value[0, :] = 0
                self.value[-1, :] = 0
            elif bc_velocity[1] == 'no-slip':
                self.value[0, :] \
                    = -2 * (psi.value[1, :]-psi.value[0, :]) / (dx**2)
                self.value[-1, :] \
                    = -2 * (psi.value[-2, :]-psi.value[-1, :]) / (dx**2)
            elif bc_velocity[1] == 'periodic':
                self.value[0, :] = self.value[-2, :]
                self.value[-1, :] = self.value[1, :]

        elif self.__name == 'potential':

            # Bottom and top boundaries
            if bc_magnetic[0] == 'perfect-conductor':
                self.value[:, 0] = potential_bottom
                self.value[:, -1] = potential_top
            elif bc_magnetic[0] == 'periodic':
                self.value[:, 0] = self.value[:, -2]
                self.value[:, -1] = self.value[:, 1]

            # Left and right boundaries
            if bc_magnetic[1] == 'perfect-conductor':
                self.value[0, :] = 0
                self.value[-1, :] = 0
            elif bc_magnetic[1] == 'periodic':
                self.value[0, :] = self.value[-2, :]
                self.value[-1, :] = self.value[1, :]

        elif self.__name == 'current':

            potential_name: str = ''
            if isinstance(potential, Variable):
                potential_name = potential.__name
            if (potential_name != 'potential'):
                logger.error('Invalid argument')
                sys.exit(1)

            # Bottom and top boundaries
            if bc_magnetic[0] == 'perfect-conductor':
                self.value[:, 0] \
                    = -2 * (potential.value[:, 1]
                            - potential.value[:, 0]) / (dy**2) \
                    + 2 * b_bottom / dy
                self.value[:, -1] \
                    = -2 * (potential.value[:, -2]
                            - potential.value[:, -1]) / (dy**2) \
                    - 2 * b_top / dy
            elif bc_magnetic[0] == 'periodic':
                self.value[:, 0] = self.value[:, -2]
                self.value[:, -1] = self.value[:, 1]

            # Left and right boundaries
            if bc_magnetic[1] == 'perfect-conductor':
                self.value[0, :] \
                    = -2 * (potential.value[1, :]
                            - potential.value[0, :]) / (dx**2)
                self.value[-1, :] \
                    = -2 * (potential.value[-2, :]
                            - potential.value[-1, :]) / (dx**2)
            elif bc_magnetic[1] == 'periodic':
                self.value[0, :] = self.value[-2, :]
                self.value[-1, :] = self.value[1, :]

    def advection(self,
                  psi_or_potential: Self,
                  *,
                  prev: bool = False) -> ArrayFloat:
        """The wrapper of the function to calculate the advection term.

        Parameters
        ----------
        psi_or_potential : Self
            The instance of Variable class (psi or potential).
        prev : bool, optional, default False
            The boolean value to switch between the current and previous
            values.

        Returns
        -------
        ArrayFloat
            The advection term.

        Warnings
        --------
        Invalid argument
            If the arguments are invalid.
        """

        logger: DefaultLogger = create_function_name_logger()

        psi_or_potential_name: str = ''
        if isinstance(psi_or_potential, Variable):
            psi_or_potential_name = psi_or_potential.__name
        if not (psi_or_potential_name in ('psi', 'potential')):
            logger.error('Invalid argument')
            sys.exit(1)

        self_value: ArrayFloat
        psi_or_potential_value: ArrayFloat
        if prev:
            self_value = self.value_prev
            psi_or_potential_value = psi_or_potential.value_prev
        else:
            self_value = self.value
            psi_or_potential_value = psi_or_potential.value

        dx: float = Variable.__dx
        dy: float = Variable.__dy

        return calc_advection(
            self_value, dx, dy,
            psi_or_potential_value, psi_or_potential_name == 'psi')

    def beta(self,
             *,
             prev: bool = False) -> ArrayFloat:
        """The wrapper of the function to calculate the beta term.

        Parameters
        ----------
        prev : bool, optional, default False
            The boolean value to switch between the current and previous
            values.

        Returns
        -------
        ArrayFloat
            The beta term.
        """

        self_value: ArrayFloat
        if prev:
            self_value = self.value_prev
        else:
            self_value = self.value

        dx: float = Variable.__dx

        return calc_beta(self_value, dx)

    def laplacian(self,
                  *,
                  prev: bool = False) -> ArrayFloat:
        """The wrapper of the function to calculate the Laplacian.

        Parameters
        ----------
        prev : bool, optional, default False
            The boolean value to switch between the current and previous
            values.

        Returns
        -------
        ArrayFloat
            The Laplacian.
        """

        self_value: ArrayFloat
        if prev:
            self_value = self.value_prev
        else:
            self_value = self.value

        dx: float = Variable.__dx
        dy: float = Variable.__dy

        return calc_laplacian(self_value, dx, dy)

    def poisson_solver(self,
                       source: ArrayFloat,
                       *,
                       omega_sor: float,
                       tol_sor: float,
                       max_iter_sor: int) -> None:
        """Solve the Poisson equation using the SOR method.

        Parameters
        ----------
        source : ArrayFloat
            The source term of the Poisson equation.
        omega_sor : float
            The relaxation factor for the SOR method.
        tol_sor : float
            The tolerance for the SOR method.
        max_iter_sor : int
            The maximum number of iterations for the SOR method.
        """

        dx: float = Variable.__dx
        dy: float = Variable.__dy
        max_err: float
        for _ in range(max_iter_sor):
            self.value, max_err = sor_one_iter(
                self.value, source, dx, dy, omega_sor=omega_sor)
            self.set_boundary_condition()
            if max_err < tol_sor:
                break


@njit(cache=True, parallel=SWITCH_NUMBA_PARALLEL)
def calc_advection(variable: ArrayFloat,
                   dx: float,
                   dy: float,
                   psi_or_potential_value: ArrayFloat,
                   is_psi: bool) -> ArrayFloat:
    """Calculate the advection term.

    Parameters
    ----------
    variable : ArrayFloat
        The variable to calculate the advection term.
    dx : float
        The interval of grid points in the x direction.
    dy : float
        The interval of grid points in the y direction.
    psi_or_potential_value : ArrayFloat
        The values of psi or potential on grid points.
    is_psi : bool
        The boolean value to switch between psi and potential.

    Returns
    -------
    advection : ArrayFloat
        The advection term.
    """

    num_x: int
    num_y: int
    num_x, num_y = variable.shape

    advection: ArrayFloat \
        = np.zeros((num_x, num_y), dtype=np.float64)

    inv_dx: float = 1 / dx
    inv_dy: float = 1 / dy
    inv_2dx: float = 1 / (2 * dx)
    inv_2dy: float = 1 / (2 * dy)

    if is_psi:

        for ix in prange(1, num_x-1):
            for iy in range(1, num_y-1):
                vx = (psi_or_potential_value[ix, iy+1]
                      - psi_or_potential_value[ix, iy-1]) * inv_2dy
                vy = - (psi_or_potential_value[ix+1, iy]
                        - psi_or_potential_value[ix-1, iy]) * inv_2dx
                advection[ix, iy] = (
                    max(vx, 0)
                    * (variable[ix, iy] - variable[ix-1, iy]) * inv_dx
                    + min(vx, 0)
                    * (variable[ix+1, iy] - variable[ix, iy]) * inv_dx
                    + max(vy, 0)
                    * (variable[ix, iy] - variable[ix, iy-1]) * inv_dy
                    + min(vy, 0)
                    * (variable[ix, iy+1] - variable[ix, iy]) * inv_dy
                )

    else:

        for ix in prange(1, num_x-1):
            for iy in range(1, num_y-1):
                bx = (psi_or_potential_value[ix, iy+1]
                      - psi_or_potential_value[ix, iy-1]) * inv_2dy
                by = - (psi_or_potential_value[ix+1, iy]
                        - psi_or_potential_value[ix-1, iy]) * inv_2dx
                advection[ix, iy] = (
                    bx * (variable[ix+1, iy]
                          - variable[ix-1, iy]) * inv_2dx
                    + by * (variable[ix, iy+1]
                            - variable[ix, iy-1]) * inv_2dy
                )

    return advection


@njit(cache=True, parallel=SWITCH_NUMBA_PARALLEL)
def calc_beta(variable: ArrayFloat,
              dx: float) -> ArrayFloat:
    """Calculate the beta term.

    Parameters
    ----------
    variable : ArrayFloat
        The variable to calculate the beta term.
    dx : float
        The interval of grid points in the x direction.

    Returns
    -------
    beta : ArrayFloat
        The beta term.
    """

    num_x: int
    num_y: int
    num_x, num_y = variable.shape

    beta: ArrayFloat = np.zeros((num_x, num_y), dtype=np.float64)

    inv_2dx: float = 1 / (2 * dx)

    for ix in prange(1, num_x-1):
        for iy in range(1, num_y-1):
            beta[ix, iy] = - (variable[ix+1, iy]
                              - variable[ix-1, iy]) * inv_2dx

    return beta


@njit(cache=True, parallel=SWITCH_NUMBA_PARALLEL)
def calc_laplacian(variable: ArrayFloat,
                   dx: float,
                   dy: float) -> ArrayFloat:
    """Calculate the Laplacian.

    Parameters
    ----------
    variable : ArrayFloat
        The variable to calculate the Laplacian.
    dx : float
        The interval of grid points in the x direction.
    dy : float
        The interval of grid points in the y direction.

    Returns
    -------
    laplacian : ArrayFloat
        The Laplacian.
    """

    num_x: int
    num_y: int
    num_x, num_y = variable.shape

    laplacian: ArrayFloat = np.zeros((num_x, num_y), dtype=np.float64)

    inv_dx2: float = 1 / (dx**2)
    inv_dy2: float = 1 / (dy**2)

    for ix in prange(1, num_x-1):
        for iy in range(1, num_y-1):
            laplacian[ix, iy] = (
                (variable[ix+1, iy] - 2 * variable[ix, iy]
                 + variable[ix-1, iy]) * inv_dx2
                + (variable[ix, iy+1] - 2 * variable[ix, iy]
                   + variable[ix, iy-1]) * inv_dy2
            )

    return laplacian


@njit(cache=True, parallel=SWITCH_NUMBA_PARALLEL)
def sor_one_iter(solution: ArrayFloat,
                 source: ArrayFloat,
                 dx: float,
                 dy: float,
                 *,
                 omega_sor: float) -> tuple[ArrayFloat, float]:
    """Perform one iteration of the Red-Black SOR method.

    Parameters
    ----------
    solution : ArrayFloat
        The unknown function of the Poisson equation.
    source : ArrayFloat
        The source term of the Poisson equation.
    dx : float
        The interval of grid points in the x direction
    dy : float
        The interval of grid points in the y direction
    omega_sor : float
        The relaxation factor for the SOR method.

    Returns
    -------
    solution : ArrayFloat
        The unknown function of the Poisson equation.
    max_err : float
        The maximum error.
    """

    num_x: int
    num_y: int
    num_x, num_y = solution.shape

    inv_dx2: float = 1 / (dx**2)
    inv_dy2: float = 1 / (dy**2)

    max_err: float = 0
    inv_denominator: float = 1 / (2 * inv_dx2 + 2 * inv_dy2)

    old: float
    new: float
    err: float

    for ix in prange(1, num_x-1):
        for iy in range(1, num_y-1):
            if (ix + iy) % 2 == 0:
                old = solution[ix, iy]
                new = (
                    (solution[ix+1, iy] + solution[ix-1, iy])
                    * inv_dx2
                    + (solution[ix, iy+1] + solution[ix, iy-1])
                    * inv_dy2
                    - source[ix, iy]
                ) * inv_denominator
                solution[ix, iy] = (1-omega_sor) * old + omega_sor * new

                if new != 0:
                    err = abs((new - old) / new)
                    max_err = max(max_err, err)

    for ix in prange(1, num_x-1):
        for iy in range(1, num_y-1):
            if (ix + iy) % 2 == 1:
                old = solution[ix, iy]
                new = (
                    (solution[ix+1, iy] + solution[ix-1, iy])
                    * inv_dx2
                    + (solution[ix, iy+1] + solution[ix, iy-1])
                    * inv_dy2
                    - source[ix, iy]
                ) * inv_denominator
                solution[ix, iy] = (1-omega_sor) * old + omega_sor * new

                if new != 0:
                    err = abs((new - old) / new)
                    max_err = max(max_err, err)

    return solution, max_err


def rhs_omega_equation(psi: Variable,
                       omega: Variable,
                       potential: Variable,
                       current: Variable,
                       re: float,
                       ma: float,
                       beta_param: float,
                       *,
                       prev: bool = False) -> ArrayFloat:
    """Calculate the right-hand side of the equation for the vorticity.

    Parameters
    ----------
    psi : Variable
        The instance of Variable class (psi).
    omega : Variable
        The instance of Variable class (omega).
    potential : Variable
        The instance of Variable class (potential).
    current : Variable
        The instance of Variable class (current).
    re : float
        The Reynolds number.
    ma : float
        The Alfven Mach number.
    beta_param : float
        The nondimensional beta.
    prev : bool, optional, default False
        The boolean value to switch between the current and previous
        values.

    Returns
    -------
    ArrayFloat
        The right-hand side of the equation for the vorticity.
    """

    return -omega.advection(psi, prev=prev) \
        - beta_param * psi.beta(prev=prev) \
        + (1/re) * omega.laplacian(prev=prev) \
        + (1/(ma**2)) * current.advection(potential, prev=prev)


def rhs_potential_equation(psi: Variable,
                           potential: Variable,
                           rm: float,
                           *,
                           ic_magnetic: tuple[str, str],
                           prev: bool = False) -> ArrayFloat:
    """Calculate the right-hand side of the equation for the vector potential.

    Parameters
    ----------
    psi : Variable
        The instance of Variable class (psi).
    potential : Variable
        The instance of Variable class (potential).
    rm : float
        The Magnetic Reynolds number.
    ic_magnetic : tuple[str, str]
            The initial condition for magnetic field.
    prev : bool, optional, default False
        The boolean value to switch between the current and previous
        values.

    Returns
    -------
    ArrayFloat
        The right-hand side of the equation for the vector potential.
    """

    gauge: float = 0
    if ic_magnetic[0] == 'x-linear-shear':
        if ic_magnetic[1] == '+':
            gauge = -2
        elif ic_magnetic[1] == '-':
            gauge = 2

    return -potential.advection(psi, prev=prev) \
        + (1/rm) * (potential.laplacian(prev=prev) + gauge)


@njit(cache=True, parallel=SWITCH_NUMBA_PARALLEL)
def calc_forcing(grid_x_value: ArrayFloat,
                 grid_y_value: ArrayFloat,
                 t: float,
                 strength_forcing: float,
                 x0: float,
                 y0: float,
                 sigma: float,
                 frequency: float,
                 kx: float,
                 ky: float) -> ArrayFloat:
    """Calculate the forcing term.

    Parameters
    ----------
    grid_x_value : ArrayFloat
        The x coordinates of grid points.
    grid_y_value : ArrayFloat
        The y coordinates of grid points.
    t : float
        Time.
    strength_forcing : float
        The strength of the forcing term.
    x0 : float
        The x-coordinate of the center of the Gaussian function.
    y0 : float
        The y-coordinate of the center of the Gaussian function.
    sigma : float
        The standard deviation of the Gaussian function.
    frequency : float
        The frequency of the forcing term.
    kx : float
        The wave number in the x direction of the forcing term.
    ky : float
        The wave number in the y direction of the forcing term.

    Returns
    -------
    ArrayFloat
        The forcing term.
    """

    if strength_forcing == 0:
        return np.zeros_like(grid_x_value)

    return strength_forcing * np.exp(
        - ((grid_x_value - x0)**2 + (grid_y_value - y0)**2)
        / (2 * sigma**2)
    ) * np.cos(kx * grid_x_value
               + ky * grid_y_value - frequency * t)


def create_plot(frame: int, *fargs) -> tuple:
    """Create contour plots for each time step.

    Parameters
    ----------
    frame : int
        Frame number
    fargs : tuple
        A tuple containing the x coordinates of grid points, y
        coordinates of grid points, time, results of psi, omega,
        potential and current.
    """

    plotter: DefaultPlotter
    grid_x_value: ArrayFloat
    grid_y_value: ArrayFloat
    results_t: list[float]
    results_psi: list[ArrayFloat]
    results_omega: list[ArrayFloat]
    results_potential: list[ArrayFloat]
    results_current: list[ArrayFloat]
    num_axes: int
    plotter, \
        grid_x_value, grid_y_value, results_t, \
        results_psi, results_omega, \
        results_potential, results_current, \
        num_axes = fargs

    if num_axes == 2:
        plotter.axes[0].cla()
        plotter.axes[1].cla()
    elif num_axes == 4:
        plotter.axes[0, 0].cla()
        plotter.axes[0, 1].cla()
        plotter.axes[1, 0].cla()
        plotter.axes[1, 1].cla()

    # psi
    psi_value: ArrayFloat = results_psi[frame]
    vmin: float = np.min(results_psi)
    vmax: float = np.max(results_psi)
    if num_axes == 2:
        axis = plotter.axes[0]
    elif num_axes == 4:
        axis = plotter.axes[0, 0]
    im1 = axis.contourf(grid_x_value, grid_y_value, psi_value,
                        vmin=vmin, vmax=vmax, levels=20, cmap='jet')
    axis.contour(im1, colors='k', linewidths=0.5)
    axis.set_xlabel(r'$x$', fontsize=16)
    axis.set_ylabel(r'$y$', fontsize=16)
    axis.set_title('Stream function', fontsize=14)
    axis.set_aspect('equal')
    axis.tick_params(labelsize=12)

    # omega
    omega_value: ArrayFloat = results_omega[frame]
    vmin = np.min(results_omega)
    vmax = np.max(results_omega)
    if num_axes == 2:
        axis = plotter.axes[1]
    elif num_axes == 4:
        axis = plotter.axes[0, 1]
    im2 = axis.contourf(grid_x_value, grid_y_value, omega_value,
                        vmin=vmin, vmax=vmax, levels=20, cmap='jet')
    axis.contour(im2, colors='k', linewidths=0.5)
    axis.set_xlabel(r'$x$', fontsize=16)
    axis.set_ylabel(r'$y$', fontsize=16)
    axis.set_title('Vorticity', fontsize=14)
    axis.set_aspect('equal')
    axis.tick_params(labelsize=12)

    if num_axes == 2:
        im3 = None
        im4 = None
    elif num_axes == 4:
        # potential
        potential_value: ArrayFloat = results_potential[frame]
        vmin = np.min(results_potential)
        vmax = np.max(results_potential)
        im3 = plotter.axes[1, 0].contourf(
            grid_x_value, grid_y_value, potential_value,
            vmin=vmin, vmax=vmax, levels=20, cmap='jet')
        plotter.axes[1, 0].contour(im3, colors='k', linewidths=0.5)
        plotter.axes[1, 0].set_xlabel(r'$x$', fontsize=16)
        plotter.axes[1, 0].set_ylabel(r'$y$', fontsize=16)
        plotter.axes[1, 0].set_title('Vector potential', fontsize=14)
        plotter.axes[1, 0].set_aspect('equal')
        plotter.axes[1, 0].tick_params(labelsize=12)

        # current
        current_value: ArrayFloat = results_current[frame]
        vmin = np.min(results_current)
        vmax = np.max(results_current)
        im4 = plotter.axes[1, 1].contourf(
            grid_x_value, grid_y_value, current_value,
            vmin=vmin, vmax=vmax, levels=20, cmap='jet')
        plotter.axes[1, 1].contour(im4, colors='k', linewidths=0.5)
        plotter.axes[1, 1].set_xlabel(r'$x$', fontsize=16)
        plotter.axes[1, 1].set_ylabel(r'$y$', fontsize=16)
        plotter.axes[1, 1].set_title('Electric current', fontsize=14)
        plotter.axes[1, 1].set_aspect('equal')
        plotter.axes[1, 1].tick_params(labelsize=12)

    plotter.fig.suptitle(
        r'$t =$' + f' {results_t[frame]:.1f}', fontsize=20)

    return im1, im2, im3, im4


if __name__ == '__main__':
    timer: DefaultTimer = DefaultTimer(__name__)
    timer.start()

    logger: DefaultLogger = DefaultLogger(__name__)

    list_params: list[str] = [
        f'{RE=}',
    ]
    if (IC_MAGNETIC[0] != 'zero') or (STRENGTH_FORCING_POTENTIAL != 0):
        list_params += [
            f'{RM=}',
            f'{MA=}',
        ]
    list_params += [
        f'{BETA_PARAM=}',
        f'{BC_VELOCITY=}',
    ]
    if (IC_MAGNETIC[0] != 'zero') or (STRENGTH_FORCING_POTENTIAL != 0):
        list_params += [
            f'{BC_MAGNETIC=}',
        ]
    list_params += [
        f'{IC_VELOCITY=}',
        f'{IC_MAGNETIC=}',
        f'{ASPECT_RATIO=}',
        f'{NUM_X=}',
        f'{NUM_Y=}',
        f'{STRENGTH_FORCING_OMEGA=}',
        f'{STRENGTH_FORCING_POTENTIAL=}',
    ]
    if (STRENGTH_FORCING_OMEGA != 0) \
            or (STRENGTH_FORCING_POTENTIAL != 0):
        list_params += [
            f'{X_FORCING=}',
            f'{Y_FORCING=}',
            f'{SD_FORCING=}',
            f'{FREQUENCY_FORCING=}',
            f'{K_FORCING=}',
        ]
    list_params += [
        f'{DT=}',
        f'{NUM_STEP=}',
    ]
    logger.show_params(*list_params)

    Variable.set_class_variables(
        NUM_X, NUM_Y, ASPECT_RATIO,
        bc_velocity=BC_VELOCITY, bc_magnetic=BC_MAGNETIC,
        ic_velocity=IC_VELOCITY, ic_magnetic=IC_MAGNETIC)

    grid_x: Variable = Variable('grid_x')
    grid_y: Variable = Variable('grid_y')
    psi: Variable = Variable('psi')
    omega: Variable = Variable('omega')
    potential: Variable = Variable('potential')
    current: Variable = Variable('current')

    t: float = 0

    figsize: tuple[float, float]
    plotter: DefaultPlotter
    num_axes: int
    if (IC_MAGNETIC[0] == 'zero') and (STRENGTH_FORCING_POTENTIAL == 0):
        figsize = (int(ASPECT_RATIO * 8), 4)
        plotter = create_plotter(1, 2, figsize=figsize)
        num_axes = 2
    else:
        figsize = (int(ASPECT_RATIO * 8), 8)
        plotter = create_plotter(2, 2, figsize=figsize)
        num_axes = 4

    results_t: list[float] = [t]
    results_psi: list[ArrayFloat] = [psi.value.copy()]
    results_omega: list[ArrayFloat] = [omega.value.copy()]
    results_potential: list[ArrayFloat] = [potential.value.copy()]
    results_current: list[ArrayFloat] = [current.value.copy()]

    progress_bar: ProgressBar = ProgressBar(
        'time_integration', NUM_STEP)
    progress_bar.start()

    delta_prev: ArrayFloat
    delta: ArrayFloat
    for it in range(NUM_STEP):

        forcing_omega: ArrayFloat = calc_forcing(
            grid_x.value, grid_y.value, t,
            STRENGTH_FORCING_OMEGA,
            X_FORCING, Y_FORCING, SD_FORCING,
            FREQUENCY_FORCING, K_FORCING[0], K_FORCING[1])
        forcing_potential: ArrayFloat = calc_forcing(
            grid_x.value, grid_y.value, t,
            STRENGTH_FORCING_POTENTIAL,
            X_FORCING, Y_FORCING, SD_FORCING,
            FREQUENCY_FORCING, K_FORCING[0], K_FORCING[1])

        # Predictor (2nd order Adams-Bashforth method)
        delta_prev = rhs_omega_equation(
            psi, omega, potential, current,
            RE, MA, BETA_PARAM, prev=True) \
            + forcing_omega
        delta = rhs_omega_equation(
            psi, omega, potential, current,
            RE, MA, BETA_PARAM) \
            + forcing_omega
        omega.value_prev = omega.value.copy()
        omega.value += ((3/2) * delta - (1/2) * delta_prev) * DT

        delta_prev = rhs_potential_equation(
            psi, potential, RM, ic_magnetic=IC_MAGNETIC, prev=True) \
            + forcing_potential
        delta = rhs_potential_equation(
            psi, potential, RM, ic_magnetic=IC_MAGNETIC) \
            + forcing_potential
        potential.value_prev = potential.value.copy()
        potential.value += ((3/2) * delta - (1/2) * delta_prev) * DT
        potential.set_boundary_condition()

        psi.value_prev = psi.value.copy()
        psi.poisson_solver(-omega.value,
                           omega_sor=OMEGA_SOR,
                           tol_sor=TOL_SOR,
                           max_iter_sor=MAX_ITER_SOR)
        omega.set_boundary_condition(psi=psi)

        current.value_prev = current.value.copy()
        current.value = - potential.laplacian()
        current.set_boundary_condition(potential=potential)

        # Corrector (2nd order Adams-Moulton method)
        delta_prev = rhs_omega_equation(
            psi, omega, potential, current,
            RE, MA, BETA_PARAM, prev=True) \
            + forcing_omega
        delta = rhs_omega_equation(
            psi, omega, potential, current,
            RE, MA, BETA_PARAM) \
            + forcing_omega
        omega.value = omega.value_prev \
            + ((1/2) * delta + (1/2) * delta_prev) * DT

        delta_prev = rhs_potential_equation(
            psi, potential, RM, ic_magnetic=IC_MAGNETIC, prev=True) \
            + forcing_potential
        delta = rhs_potential_equation(
            psi, potential, RM, ic_magnetic=IC_MAGNETIC) \
            + forcing_potential
        potential.value = potential.value_prev \
            + ((1/2) * delta + (1/2) * delta_prev) * DT
        potential.set_boundary_condition()

        psi.poisson_solver(-omega.value,
                           omega_sor=OMEGA_SOR,
                           tol_sor=TOL_SOR,
                           max_iter_sor=MAX_ITER_SOR)
        omega.set_boundary_condition(psi=psi)

        current.value = - potential.laplacian()
        current.set_boundary_condition(potential=potential)

        t += DT
        if it % STEP_INTERVAL_PLOT == STEP_INTERVAL_PLOT - 1:
            results_t.append(t)
            results_psi.append(psi.value.copy())
            results_omega.append(omega.value.copy())
            results_potential.append(potential.value.copy())
            results_current.append(current.value.copy())

        progress_bar.update(it)

    # colorbar
    id_frame: int = int(NUM_FRAME * 0.1)
    if (STRENGTH_FORCING_OMEGA != 0) \
            or (STRENGTH_FORCING_POTENTIAL != 0):
        id_frame = NUM_FRAME // 2
    im1, im2, im3, im4 = create_plot(
        id_frame, plotter,
        grid_x.value, grid_y.value, results_t,
        results_psi, results_omega,
        results_potential, results_current,
        num_axes)
    if num_axes == 2:
        plotter.fig.colorbar(im1, ax=plotter.axes[0])
        plotter.fig.colorbar(im2, ax=plotter.axes[1])
    elif num_axes == 4:
        plotter.fig.colorbar(im1, ax=plotter.axes[0, 0])
        plotter.fig.colorbar(im2, ax=plotter.axes[0, 1])
        plotter.fig.colorbar(im3, ax=plotter.axes[1, 0])
        plotter.fig.colorbar(im4, ax=plotter.axes[1, 1])

    plotter.fig.tight_layout()
    anim = animation.FuncAnimation(
        plotter.fig, create_plot, range(len(results_t)),
        fargs=(plotter,
               grid_x.value, grid_y.value, results_t,
               results_psi, results_omega,
               results_potential, results_current,
               num_axes))

    anim.save('mhd2dbox.webp', writer='pillow', fps=10)
    anim.save('mhd2dbox.mp4', writer='ffmpeg', fps=10)

    timer.end()
